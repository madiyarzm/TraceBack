import type { TraceEvent } from './traceStore';

/**
 * Anomaly detection engine for live agent sessions.
 *
 * Pure and tail-only by design: every detector walks backward from the end of
 * the event array with a hard scan cap and early exit, so evaluation is O(1)
 * per call regardless of session length. The live state self-clears — it is
 * recomputed on every event, so a success after an error run (or a PostToolUse
 * after a stall) clears the warning without any dismiss bookkeeping. Each
 * ONSET is additionally recorded permanently on the session (traceStore).
 */

export type AnomalyType =
  | 'near_duplicate_loop'
  | 'error_thrash'
  | 'stall'
  | 'context_spiral'
  | 'scope_creep'
  | 'thrash_no_progress';

export type AnomalySeverity = 'high' | 'medium';

export interface AnomalyState {
  isAnomalous:     boolean;
  type?:           AnomalyType;
  severity?:       AnomalySeverity;
  /** Short, plain-English name of the pattern ("Near-duplicate loop"). */
  title?:          string;
  /** One sentence with specifics — kept as `reason` too for older UI paths. */
  description?:    string;
  reason?:         string;
  flaggedEventIds: string[];
}

/**
 * Permanent record of a detection. The live AnomalyState self-clears when the
 * agent recovers; records never do — they are the session's evidence trail.
 */
export interface AnomalyRecord {
  type:            AnomalyType;
  severity:        AnomalySeverity;
  title:           string;
  description:     string;
  /** Mirror of description — legacy field older webview builds read. */
  reason:          string;
  flaggedEventIds: string[];
  detectedAt:      number;
}

const NEAR_DUP_N     = 3;        // same tool+base-command in the window
const NEAR_DUP_WIN   = 10;       // window (events of any kind) for near-dups
const THRASH_N       = 3;        // consecutive failed PostToolUse results
const STALL_MS       = 120_000;  // PreToolUse with no Post for this long
const SPIRAL_N       = 8;        // consecutive reads with no edit/write/bash
const FILE_THRASH_N  = 3;        // read→edit round-trips on one file
const SCAN_CAP       = 64;       // never look further back than this many events

export const NO_ANOMALY: AnomalyState = { isAnomalous: false, flaggedEventIds: [] };

const EDIT_TOOLS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite']);
const READ_TOOLS = new Set(['Read', 'Grep', 'Glob', 'LS', 'NotebookRead']);
const BREAK_SPIRAL_TOOLS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite', 'Bash']);

export function detectAnomaly(events: TraceEvent[], now: number): AnomalyState {
  if (events.length === 0) return NO_ANOMALY;
  // Most acute condition first (a stall means nothing is happening right
  // now), then the high-severity loops, then the medium-severity drifts.
  // thrash_no_progress runs before near_duplicate_loop: its read→edit rounds
  // also contain 3 same-file edits, which would otherwise match the looser
  // near-duplicate rule and mask the more specific diagnosis.
  return (
    detectStall(events, now) ??
    detectErrorThrash(events) ??
    detectThrashNoProgress(events) ??
    detectNearDuplicateLoop(events) ??
    detectScopeCreep(events) ??
    detectContextSpiral(events) ??
    NO_ANOMALY
  );
}

function state(
  type: AnomalyType,
  severity: AnomalySeverity,
  title: string,
  description: string,
  flaggedEventIds: string[],
): AnomalyState {
  return { isAnomalous: true, type, severity, title, description, reason: description, flaggedEventIds };
}

/** Last `n` events of `kind`, newest first, scanning at most SCAN_CAP back. */
function tailOfKind(events: TraceEvent[], kind: TraceEvent['kind'], n: number): TraceEvent[] {
  const out: TraceEvent[] = [];
  const stop = Math.max(0, events.length - SCAN_CAP);
  for (let i = events.length - 1; i >= stop && out.length < n; i--) {
    if (events[i].kind === kind) out.push(events[i]);
  }
  return out;
}

function filePathOf(e: TraceEvent): string | undefined {
  const input = e.toolInput ?? {};
  const p = input.file_path ?? input.path ?? input.notebook_path;
  return typeof p === 'string' ? p : undefined;
}

// ── 3. Stall (medium) ────────────────────────────────────────────────────────

function detectStall(events: TraceEvent[], now: number): AnomalyState | null {
  const last = events[events.length - 1];
  if (last.kind !== 'pre_tool_use') return null;
  if (now - last.timestamp <= STALL_MS) return null;
  const secs = Math.round((now - last.timestamp) / 1000);
  return state(
    'stall', 'medium', 'Stall',
    `Stalled: ${last.toolName ?? 'tool'} unresponsive for ${secs}s.`,
    [last.id],
  );
}

// ── 2. Error thrash (high) ───────────────────────────────────────────────────

function detectErrorThrash(events: TraceEvent[]): AnomalyState | null {
  const posts = tailOfKind(events, 'post_tool_use', THRASH_N);
  if (posts.length < THRASH_N || !posts.every((e) => e.isError)) return null;
  return state(
    'error_thrash', 'high', 'Error loop',
    `${THRASH_N} consecutive tool failures — the agent is not recovering.`,
    posts.map((e) => e.id),
  );
}

// ── 1. Near-duplicate loop (high) ────────────────────────────────────────────

/**
 * Loosened call signature: tool name + base command/target with arguments
 * stripped. "python test.py" retried with different flags, or the same file
 * re-read over and over, counts as the same action — exact-match repetition
 * proved too strict in field data.
 */
export function callSignature(e: TraceEvent): string {
  const input = e.toolInput ?? {};
  if (e.toolName === 'Bash' && typeof input.command === 'string') {
    const base = input.command
      .trim()
      .split(/\s+/)
      .filter((t) => !t.startsWith('-'))
      .slice(0, 2)
      .join(' ');
    return `Bash:${base}`;
  }
  const target =
    filePathOf(e) ??
    (typeof input.url === 'string' ? input.url : undefined) ??
    (typeof input.query === 'string' ? input.query : undefined) ??
    (typeof input.pattern === 'string' ? input.pattern : undefined) ??
    JSON.stringify(input);
  return `${e.toolName}:${target}`;
}

/** Plan bookkeeping repeats legitimately — never a "loop". */
const PLAN_TOOL_NAMES = new Set(['TodoWrite', 'TaskCreate', 'TaskUpdate', 'TodoRead']);

function detectNearDuplicateLoop(events: TraceEvent[]): AnomalyState | null {
  const start = Math.max(0, events.length - NEAR_DUP_WIN);
  const window = events.slice(start).filter(
    (e) => e.kind === 'pre_tool_use' && !PLAN_TOOL_NAMES.has(e.toolName ?? '')
  );
  if (window.length === 0) return null;

  const newest = window[window.length - 1];
  const sig = callSignature(newest);
  const matches = window.filter((e) => e.toolName === newest.toolName && callSignature(e) === sig);
  if (matches.length < NEAR_DUP_N) return null;

  const what = sig.slice(sig.indexOf(':') + 1) || newest.toolName || 'call';
  return state(
    'near_duplicate_loop', 'high', 'Near-duplicate loop',
    `${newest.toolName ?? 'Tool'} "${what}" ran ${matches.length}× in the last ${NEAR_DUP_WIN} events.`,
    matches.map((e) => e.id),
  );
}

// ── 6. Thrash without progress (high) ────────────────────────────────────────

/**
 * Same file READ then EDIT, 3+ round-trips in a row, with no plan-tool call
 * in between (a TodoWrite/TaskUpdate between rounds breaks the alternation,
 * which is exactly the "active task changed" reset the rule wants).
 */
function detectThrashNoProgress(events: TraceEvent[]): AnomalyState | null {
  const pres = tailOfKind(events, 'pre_tool_use', FILE_THRASH_N * 2);
  if (pres.length < FILE_THRASH_N * 2) return null;

  // pres is newest-first: expect Edit(f), Read(f), Edit(f), Read(f), …
  const file = filePathOf(pres[0]);
  if (!file || !EDIT_TOOLS.has(pres[0].toolName ?? '')) return null;

  for (let i = 0; i < FILE_THRASH_N * 2; i++) {
    const e = pres[i];
    const wantEdit = i % 2 === 0;
    const okTool = wantEdit ? EDIT_TOOLS.has(e.toolName ?? '') : e.toolName === 'Read';
    if (!okTool || filePathOf(e) !== file) return null;
  }

  const flagged = pres.slice(0, FILE_THRASH_N * 2);
  return state(
    'thrash_no_progress', 'high', 'Thrash without progress',
    `${shortPath(file)} read and edited ${FILE_THRASH_N}× in a row without the active task changing.`,
    flagged.map((e) => e.id),
  );
}

// ── 5. Scope creep (medium) ──────────────────────────────────────────────────

function detectScopeCreep(events: TraceEvent[]): AnomalyState | null {
  const cwd = initialCwd(events);
  if (!cwd) return null;

  const pres = tailOfKind(events, 'pre_tool_use', 1);
  const last = pres[0];
  if (!last || !EDIT_TOOLS.has(last.toolName ?? '')) return null;

  const file = filePathOf(last);
  if (!file || !file.startsWith('/')) return null;              // relative = inside cwd
  if (file === cwd || file.startsWith(cwd.endsWith('/') ? cwd : cwd + '/')) return null;

  return state(
    'scope_creep', 'medium', 'Scope creep',
    `${last.toolName === 'Write' ? 'Wrote' : 'Edited'} ${shortPath(file)} — outside the session's working directory.`,
    [last.id],
  );
}

/** The session's initial cwd — hook payloads carry it on (almost) every event. */
function initialCwd(events: TraceEvent[]): string | undefined {
  const cap = Math.min(events.length, 8);
  for (let i = 0; i < cap; i++) {
    if (events[i].cwd) return events[i].cwd;
  }
  return undefined;
}

// ── 4. Context spiral (medium) ───────────────────────────────────────────────

function detectContextSpiral(events: TraceEvent[]): AnomalyState | null {
  const flagged: string[] = [];
  const stop = Math.max(0, events.length - SCAN_CAP);
  for (let i = events.length - 1; i >= stop; i--) {
    const e = events[i];
    if (e.kind !== 'pre_tool_use') continue;
    const tool = e.toolName ?? '';
    if (BREAK_SPIRAL_TOOLS.has(tool)) break;   // acted on something — no spiral
    if (READ_TOOLS.has(tool)) flagged.push(e.id);
    // Other tools (WebSearch, TodoWrite, …) neither count nor break the run.
  }
  if (flagged.length < SPIRAL_N) return null;
  return state(
    'context_spiral', 'medium', 'Context spiral',
    `${flagged.length} consecutive reads with no edit, write, or command — exploring without acting.`,
    flagged,
  );
}

function shortPath(p: string): string {
  const parts = p.split('/').filter(Boolean);
  return parts.length <= 2 ? p : parts.slice(-2).join('/');
}
