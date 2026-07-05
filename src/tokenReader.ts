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

export async function readContextTokens(transcriptPath: string): Promise<number | null> {
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

const INTENT_MAX_CHARS = 120;

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

interface TranscriptItem {
  kind: 'text' | 'tool_use';
  text?: string;
}

/**
 * The assistant text immediately preceding the tool_use entry at
 * `toolCallIndex` (0-based, counted across the session), first sentence,
 * ≤120 chars. Returns null on any miss — the intent field is best-effort
 * everywhere and must never block rendering.
 *
 * Only the transcript tail is read. When the tail is truncated (huge
 * transcript), absolute indexing is unreliable, so we fall back to the most
 * recent tool_use — for a call that just completed that is almost always the
 * right entry anyway.
 */
export async function extractIntentForTool(
  sessionId: string,
  toolCallIndex: number,
): Promise<string | null> {
  const transcriptPath = _transcriptPaths.get(sessionId);
  if (!transcriptPath) return null;

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
        const content = parsed?.message?.content;
        if (!Array.isArray(content)) continue;
        for (const block of content) {
          if (block?.type === 'text' && typeof block.text === 'string') {
            items.push({ kind: 'text', text: block.text });
          } else if (block?.type === 'tool_use') {
            items.push({ kind: 'tool_use' });
          }
        }
      } catch {
        // partial line at the chunk boundary or non-JSON content — skip
      }
    }

    // Locate the target tool_use item.
    const wholeFile = start === 0;
    let target = -1;
    if (wholeFile) {
      let seen = 0;
      for (let i = 0; i < items.length; i++) {
        if (items[i].kind !== 'tool_use') continue;
        if (seen === toolCallIndex) { target = i; break; }
        seen++;
      }
    }
    if (target === -1) {
      // Truncated tail (or index out of range): use the latest tool_use.
      for (let i = items.length - 1; i >= 0; i--) {
        if (items[i].kind === 'tool_use') { target = i; break; }
      }
    }
    if (target === -1) return null;

    // The text block immediately preceding it.
    for (let i = target - 1; i >= 0; i--) {
      if (items[i].kind === 'text') return firstSentence(items[i].text ?? '');
      if (items[i].kind === 'tool_use') break; // another call in between — no text for this one
    }
    return null;
  } catch {
    return null;
  }
}
