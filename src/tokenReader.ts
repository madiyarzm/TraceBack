import * as fs from 'fs';

/**
 * Reads REAL token usage from a Claude Code transcript (JSONL).
 *
 * Every hook payload carries `transcript_path`; each assistant message line
 * in that file embeds `message.usage`. The latest one gives the session's
 * current context size: input + cache_read + cache_creation + output.
 *
 * Only the tail of the file is read (transcripts grow to MBs), so a refresh
 * is O(1) in transcript size.
 */

const TAIL_BYTES = 256 * 1024;

/**
 * The transcript path arrives in a hook payload, so treat it as untrusted:
 * a real Claude Code transcript is always a `.jsonl` file. Refusing anything
 * else stops a crafted payload from pointing the reader at, say, /etc/passwd.
 */
export function isTranscriptPath(p: string | undefined): p is string {
  return typeof p === 'string' && p.endsWith('.jsonl');
}

export async function readContextTokens(transcriptPath: string): Promise<number | null> {
  if (!isTranscriptPath(transcriptPath)) return null;
  try {
    const stat  = await fs.promises.stat(transcriptPath);
    const start = Math.max(0, stat.size - TAIL_BYTES);
    const fh    = await fs.promises.open(transcriptPath, 'r');
    const buf   = Buffer.alloc(stat.size - start);
    await fh.read(buf, 0, buf.length, start);
    await fh.close();

    const lines = buf.toString('utf-8').split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      if (!lines[i].includes('"usage"')) continue;
      try {
        const usage = JSON.parse(lines[i])?.message?.usage;
        if (usage && typeof usage.input_tokens === 'number') {
          return (
            (usage.input_tokens ?? 0) +
            (usage.cache_read_input_tokens ?? 0) +
            (usage.cache_creation_input_tokens ?? 0) +
            (usage.output_tokens ?? 0)
          );
        }
      } catch {
        // partial line at the chunk boundary or non-JSON content — keep scanning
      }
    }
    return null;
  } catch {
    return null; // file missing/unreadable — caller falls back to estimates
  }
}

// ─── Intent extraction ────────────────────────────────────────────────────────
//
// Before each tool call, Claude writes a text block explaining what it is
// about to do. That block lives in the same transcript JSONL, in the
// assistant message right before the tool_use entry. After PostToolUse we
// look back, grab the first sentence, and surface it as the call's "intent".

/** sessionId → transcript path, learned from hook payloads (traceStore). */
const _transcriptPaths = new Map<string, string>();

export function noteTranscript(sessionId: string, transcriptPath: string): void {
  _transcriptPaths.set(sessionId, transcriptPath);
}

const INTENT_MAX_CHARS = 220;
const INTENT_MIN_CHARS = 30;

/** Reaction interjections: as a punctuated opener ("Perfect! …") they get
 *  stripped so the substance after them survives; as the first word of the
 *  sentence proper ("Good progress on…") the sentence is commentary, not
 *  intent. "Now let me…" / "I'll…" are NOT here — announcing the next action
 *  is exactly the intent worth showing on the card. */
const REACTION_WORDS = 'perfect|excellent|good|great|nice|awesome|done|right|okay|ok|yes|hmm|interesting';
const REACTION_OPENER = new RegExp(`^(?:(?:${REACTION_WORDS})\\s*[!.,:;…—–-]+\\s*)+`, 'i');
const REACTION_SENTENCE = new RegExp(`^(?:${REACTION_WORDS})\\b`, 'i');

/** First sentence of a text block, trimmed to INTENT_MAX_CHARS. */
export function firstSentence(text: string): string | null {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (!clean) return null;
  const m = clean.match(/^.*?[.!?](?=\s|$)/);
  const sentence = (m ? m[0] : clean).trim();
  if (!sentence) return null;
  return sentence.length > INTENT_MAX_CHARS
    ? sentence.slice(0, INTENT_MAX_CHARS - 1) + '…'
    : sentence;
}

/**
 * The first substantive sentence of a text block: reaction openers
 * ("Perfect! …", "Right — …") are stripped rather than nuking the whole
 * block, pure-commentary sentences and stubs are dropped. Null when nothing
 * substantive remains — no intent beats a meaningless one.
 */
export function meaningfulIntent(text: string): string | null {
  const stripped = text.replace(/\s+/g, ' ').trim().replace(REACTION_OPENER, '');
  const sentence = firstSentence(stripped);
  if (!sentence) return null;
  if (REACTION_SENTENCE.test(sentence)) return null;
  if (sentence.length < INTENT_MIN_CHARS) return null;
  return sentence;
}

interface TranscriptItem {
  kind: 'text' | 'tool_use';
  text?: string;
  /** tool_use only — lets the reader align by tool name, not global position. */
  name?: string;
}

// ─── Decision & assumption ledger ────────────────────────────────────────────
//
// Agents make silent judgment calls in prose — "I'll assume X", "instead of
// Y" — that scroll past unread. This mines the transcript's assistant text
// for those sentences and surfaces them as first-class objects. Regex-based
// by design: it must work offline, and judgment markers are surprisingly
// regular in agent output.

export interface LedgerItem {
  text:      string;
  kind:      'decision' | 'assumption';
  /** Timestamp of the transcript line the sentence came from. */
  timestamp: number;
}

const ASSUMPTION_MARKER = /\b(i(?:'|’)ll assume|assuming|i(?:'|’)m assuming|presumably|i suspect)\b/i;
const DECISION_MARKER   = new RegExp(
  '\\b(instead of|rather than|i chose|chose to|opted (?:for|to)|went with|' +
  'i(?:\'|’)ll use|decided (?:to|on|against)|deliberately|for simplicity|' +
  'to keep it simple|for now)\\b', 'i'
);

const LEDGER_MIN_CHARS = 25;
const LEDGER_MAX_CHARS = 400;

/** Truncate at a word boundary — a mid-word chop reads as broken formatting. */
function clip(s: string): string {
  if (s.length <= LEDGER_MAX_CHARS) return s;
  const cut = s.slice(0, LEDGER_MAX_CHARS);
  const atWord = cut.slice(0, Math.max(cut.lastIndexOf(' '), LEDGER_MAX_CHARS - 40));
  return atWord.trimEnd() + '…';
}

/** Sentences of a text block that read like judgment calls. Exported for tests. */
export function mineDecisions(text: string, timestamp: number): LedgerItem[] {
  const out: LedgerItem[] = [];
  const sentences = text.replace(/\s+/g, ' ').split(/(?<=[.!?])\s+/);
  for (const raw of sentences) {
    // Inline markdown (**bold**, `code`) is authoring noise in a UI card.
    const s = raw.trim().replace(/\*\*|__|`/g, '');
    if (s.length < LEDGER_MIN_CHARS) continue;
    const isAssumption = ASSUMPTION_MARKER.test(s);
    if (!isAssumption && !DECISION_MARKER.test(s)) continue;
    // Markdown headers / bullets are structure, not prose judgment.
    if (/^[#>\-*|\d]/.test(s)) continue;
    out.push({
      text: clip(s),
      kind: isAssumption ? 'assumption' : 'decision',
      timestamp,
    });
  }
  return out;
}

/**
 * All judgment sentences in the transcript tail, deduped, oldest first.
 * Tail-only like everything else here — a merged, session-long ledger is the
 * caller's job (traceStore keeps items that scroll out of the window).
 */
export async function extractDecisions(transcriptPath: string): Promise<LedgerItem[]> {
  if (!isTranscriptPath(transcriptPath)) return [];
  try {
    const stat  = await fs.promises.stat(transcriptPath);
    const start = Math.max(0, stat.size - TAIL_BYTES);
    const fh    = await fs.promises.open(transcriptPath, 'r');
    const buf   = Buffer.alloc(stat.size - start);
    await fh.read(buf, 0, buf.length, start);
    await fh.close();

    const items: LedgerItem[] = [];
    for (const line of buf.toString('utf-8').split('\n')) {
      if (!line.includes('"content"')) continue;
      try {
        const parsed = JSON.parse(line);
        if (parsed?.type !== 'assistant') continue;
        const ts = typeof parsed.timestamp === 'string' ? Date.parse(parsed.timestamp) : NaN;
        const content = parsed?.message?.content;
        if (!Array.isArray(content)) continue;
        for (const block of content) {
          if (block?.type !== 'text' || typeof block.text !== 'string') continue;
          items.push(...mineDecisions(block.text, Number.isFinite(ts) ? ts : Date.now()));
        }
      } catch {
        // partial line at the chunk boundary — skip
      }
    }

    const seen = new Set<string>();
    return items.filter((it) => {
      const key = it.text.toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  } catch {
    return [];
  }
}

/**
 * The assistant reasoning behind a tool call: the nearest preceding text
 * block's first sentence, ≤INTENT_MAX_CHARS. Returns null on any miss — the intent
 * field is best-effort everywhere and must never block rendering.
 *
 * `fromEnd` identifies the target tool_use counted BACKWARD from the last
 * matching one in the transcript (0 = the most recent). Counting from the end
 * is stable even when only the transcript tail is read: the tail always ends
 * at the live edge of the session, so from-end offsets line up regardless of
 * how much history was truncated. When `toolName` is given, both sides count
 * only calls of that tool — a global position drifts whenever the transcript
 * carries calls the hook stream didn't see (or vice versa).
 *
 * The intent is the DIRECTLY preceding text block. If another tool_use sits
 * between the text and the target, the target is a sibling of a group whose
 * first call already carries that sentence — it gets null, not a duplicate
 * subtitle on every card of the burst.
 */
export async function extractIntentForTool(
  sessionId: string,
  fromEnd: number,
  toolName?: string,
): Promise<string | null> {
  const transcriptPath = _transcriptPaths.get(sessionId);
  if (!isTranscriptPath(transcriptPath)) return null;

  try {
    const stat  = await fs.promises.stat(transcriptPath);
    const start = Math.max(0, stat.size - TAIL_BYTES);
    const fh    = await fs.promises.open(transcriptPath, 'r');
    const buf   = Buffer.alloc(stat.size - start);
    await fh.read(buf, 0, buf.length, start);
    await fh.close();

    const items: TranscriptItem[] = [];
    for (const line of buf.toString('utf-8').split('\n')) {
      if (!line.includes('"content"')) continue;
      try {
        const parsed = JSON.parse(line);
        if (parsed?.type !== 'assistant') continue;
        // Subagent (sidechain) chatter shares the transcript file but not this
        // session's hook stream — counting it misattributes everything after.
        if (parsed?.isSidechain === true) continue;
        const content = parsed?.message?.content;
        if (!Array.isArray(content)) continue;
        for (const block of content) {
          if (block?.type === 'text' && typeof block.text === 'string') {
            items.push({ kind: 'text', text: block.text });
          } else if (block?.type === 'tool_use') {
            items.push({ kind: 'tool_use', name: typeof block.name === 'string' ? block.name : undefined });
          }
        }
      } catch {
        // partial line at the chunk boundary or non-JSON content — skip
      }
    }

    // Locate the target tool_use, counting matching calls backward from the end.
    let target = -1;
    let seen = 0;
    for (let i = items.length - 1; i >= 0; i--) {
      if (items[i].kind !== 'tool_use') continue;
      if (toolName && items[i].name !== toolName) continue;
      if (seen === fromEnd) { target = i; break; }
      seen++;
    }
    if (target === -1) return null;

    // Intent belongs to the group's FIRST call only: a tool_use between the
    // text and the target means a sibling already carries this sentence.
    for (let i = target - 1; i >= 0; i--) {
      if (items[i].kind === 'tool_use') return null;
      if (items[i].kind === 'text') return meaningfulIntent(items[i].text ?? '');
    }
    return null;
  } catch {
    return null;
  }
}
