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
  // No longer detected (read→edit iteration is normal); kept so archived
  // sessions with old records still typecheck.
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

const FAILED_CMD_N   = 3;        // exact same bash command, non-zero exit every time
const READ_LOOP_N    = 5;        // exact same read path with no edit/write between
const THRASH_N       = 3;        // consecutive failed PostToolUse results
const STALL_MS       = 120_000;  // PreToolUse with no Post for this long
const SPIRAL_N       = 8;        // consecutive reads with no edit/write/bash
const SCAN_CAP       = 64;       // never look further back than this many events

export const NO_ANOMALY: AnomalyState = { isAnomalous: false, flaggedEventIds: [] };

const EDIT_TOOLS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite']);
const READ_TOOLS = new Set(['Read', 'Grep', 'Glob', 'LS', 'NotebookRead']);
const BREAK_SPIRAL_TOOLS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite', 'Bash']);

export function detectAnomaly(events: TraceEvent[], now: number): AnomalyState {
  if (events.length === 0) return NO_ANOMALY;
  // Most acute condition first (a stall means nothing is happening right
  // now), then the high-severity loops, then the medium-severity drifts.
  return (
    detectStall(events, now) ??
    detectErrorThrash(events) ??
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

/**
 * A pending call with no result for a long time. NOT agent misbehavior — in
 * practice this is almost always Claude waiting on the USER (a permission
 * prompt, manual-mode approval, an unread question), so the UI renders it as
 * a quiet "waiting" notice and traceStore never records it as an anomaly.
 */
function detectStall(events: TraceEvent[], now: number): AnomalyState | null {
  const last = events[events.length - 1];
  if (last.kind !== 'pre_tool_use') return null;
  if (now - last.timestamp <= STALL_MS) return null;
  const secs = Math.round((now - last.timestamp) / 1000);
  return state(
    'stall', 'medium', 'Waiting',
    `${last.toolName ?? 'A tool'} call has been pending for ${secs}s — ` +
    'Claude may be waiting for your approval in the terminal.',
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
 * Deliberately strict — read→edit→re-read verification and iterative editing
 * are normal agent behavior, never a loop. Fires on exactly two patterns:
 *
 *  a) The EXACT same bash command (args included) completed FAILED_CMD_N+
 *     times, non-zero exit every time (one success clears it).
 *  b) The exact same file was Read READ_LOOP_N+ times with no Edit/Write to
 *     that file in between (an edit means the re-reads were verification).
 */
function detectNearDuplicateLoop(events: TraceEvent[]): AnomalyState | null {
  return detectFailedCommandLoop(events) ?? detectReadLoop(events);
}

function detectFailedCommandLoop(events: TraceEvent[]): AnomalyState | null {
  // Anchor on the newest completed Bash call — must itself be a failure.
  const posts = tailOfKind(events, 'post_tool_use', SCAN_CAP);
  const newest = posts.find((e) => e.toolName === 'Bash');
  if (!newest || !newest.isError) return null;
  const cmd = typeof newest.toolInput?.command === 'string' ? newest.toolInput.command : undefined;
  if (!cmd) return null;

  // Walk older completed runs of the identical command; a success clears it.
  const matches: TraceEvent[] = [];
  for (const e of posts) {
    if (e.toolName !== 'Bash' || e.toolInput?.command !== cmd) continue;
    if (!e.isError) break;
    matches.push(e);
  }
  if (matches.length < FAILED_CMD_N) return null;

  return state(
    'near_duplicate_loop', 'high', 'Near-duplicate loop',
    `"${cmd}" ran ${matches.length}× and failed every time.`,
    matches.map((e) => e.id),
  );
}

function detectReadLoop(events: TraceEvent[]): AnomalyState | null {
  // Anchor on the newest Read; count identical reads walking backward, and
  // stop at any Edit/Write touching that same file.
  const stop = Math.max(0, events.length - SCAN_CAP);
  let file: string | undefined;
  const matches: TraceEvent[] = [];
  for (let i = events.length - 1; i >= stop; i--) {
    const e = events[i];
    if (e.kind !== 'pre_tool_use') continue;
    if (file === undefined) {
      if (e.toolName !== 'Read') return null;
      file = filePathOf(e);
      if (!file) return null;
    }
    if (EDIT_TOOLS.has(e.toolName ?? '') && filePathOf(e) === file) break;
    if (e.toolName === 'Read' && filePathOf(e) === file) matches.push(e);
  }
  if (matches.length < READ_LOOP_N) return null;

  return state(
    'near_duplicate_loop', 'high', 'Near-duplicate loop',
    `${shortPath(file!)} read ${matches.length}× with no edit in between.`,
    matches.map((e) => e.id),
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
  // Claude's own config/memory lives outside every workspace by design —
  // writing there is normal agent behavior, not scope creep.
  if (file.includes('/.claude/')) return null;

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
