import type { TraceEvent } from './traceStore';

/**
 * Anomaly detection engine for live agent sessions.
 *
 * Pure and tail-only by design: every trigger walks backward from the end of
 * the event array with a hard scan cap and early exit, so evaluation is O(1)
 * per call regardless of session length. State self-clears — it is recomputed
 * on every event, so a success after an error run (or a PostToolUse after a
 * stall) clears the warning without any dismiss bookkeeping.
 */

export type AnomalyType = 'repeater' | 'error_thrash' | 'stall';

export interface AnomalyState {
  isAnomalous:     boolean;
  type?:           AnomalyType;
  reason?:         string;
  flaggedEventIds: string[];
}

const REPEAT_N = 4;       // identical PreToolUse calls in a row
const THRASH_N = 3;       // consecutive failed PostToolUse results
const STALL_MS = 60_000;  // PreToolUse with no Post for this long
const SCAN_CAP = 64;      // never look further back than this many events

export const NO_ANOMALY: AnomalyState = { isAnomalous: false, flaggedEventIds: [] };

export function detectAnomaly(events: TraceEvent[], now: number): AnomalyState {
  if (events.length === 0) return NO_ANOMALY;
  // Most acute condition wins: a stall means nothing is happening right now.
  return (
    detectStall(events, now) ??
    detectErrorThrash(events) ??
    detectRepeater(events) ??
    NO_ANOMALY
  );
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

function detectStall(events: TraceEvent[], now: number): AnomalyState | null {
  const last = events[events.length - 1];
  if (last.kind !== 'pre_tool_use') return null;
  if (now - last.timestamp <= STALL_MS) return null;
  const secs = Math.round((now - last.timestamp) / 1000);
  return {
    isAnomalous: true,
    type: 'stall',
    reason: `Stalled: ${last.toolName ?? 'tool'} unresponsive for ${secs}s.`,
    flaggedEventIds: [last.id],
  };
}

function detectErrorThrash(events: TraceEvent[]): AnomalyState | null {
  const posts = tailOfKind(events, 'post_tool_use', THRASH_N);
  if (posts.length < THRASH_N || !posts.every((e) => e.isError)) return null;
  return {
    isAnomalous: true,
    type: 'error_thrash',
    reason: `Error loop: ${THRASH_N} consecutive tool failures.`,
    flaggedEventIds: posts.map((e) => e.id),
  };
}

function detectRepeater(events: TraceEvent[]): AnomalyState | null {
  const pres = tailOfKind(events, 'pre_tool_use', REPEAT_N);
  if (pres.length < REPEAT_N) return null;
  const sig = callSignature(pres[0]);
  if (!pres.every((e) => callSignature(e) === sig)) return null;
  return {
    isAnomalous: true,
    type: 'repeater',
    reason: `Repeating action: ${pres[0].toolName ?? 'tool'} executed ${REPEAT_N} times without progress.`,
    flaggedEventIds: pres.map((e) => e.id),
  };
}

/**
 * Exact-match call signature. Deliberately strict: re-editing a file with
 * different content is legitimate iteration; the IDENTICAL call repeated is
 * the pathology. Loosen here (e.g. compare only file_path/command) if field
 * data shows missed loops.
 */
function callSignature(e: TraceEvent): string {
  return `${e.toolName} ${JSON.stringify(e.toolInput ?? {})}`;
}
