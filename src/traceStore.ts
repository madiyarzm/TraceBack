import * as vscode from 'vscode';
import { AnomalyRecord, AnomalyState, detectAnomaly } from './anomalyDetector';
import { readContextTokens } from './tokenReader';

export type EventKind =
  | 'pre_tool_use'
  | 'post_tool_use'
  | 'stop'
  | 'notification';

export type ToolName =
  | 'Read'
  | 'Edit'
  | 'Write'
  | 'Bash'
  | 'WebSearch'
  | 'WebFetch'
  | 'TodoRead'
  | 'TodoWrite'
  | 'Agent'
  | '__thinking__'
  | string;

export type NodeStatus = 'pending' | 'success' | 'error' | 'thinking';

export interface TraceEvent {
  id: string;
  sessionId: string;
  kind: EventKind;
  toolName?: ToolName;
  toolInput?: Record<string, unknown>;
  toolResponse?: string;
  isError?: boolean;
  transcriptPath?: string;
  /** Working directory of the agent session (from the hook payload). */
  cwd?: string;
  timestamp: number;
}

export interface BatchItem {
  label: string;
  detail?: string;
  status: NodeStatus;
  durationMs?: number;
}

export interface TraceNode {
  id: string;
  toolName: ToolName;
  status: NodeStatus;
  label: string;
  count: number;
  eventIds: string[];
  detail?: string;
  toolInput?: Record<string, unknown>;
  timestamp: number;
  /** Wall-clock ms between PreToolUse and PostToolUse for this call. */
  durationMs?: number;
  isBatch?: boolean;
  batchItems?: BatchItem[];
}

export interface TraceSession {
  id: string;
  label: string;
  startedAt: number;
  stopped: boolean;
  events: TraceEvent[];
  nodes: TraceNode[];
  anomaly?: AnomalyState;
  /** Every detection ever made in this session — never cleared on recovery. */
  anomalyHistory: AnomalyRecord[];
  aiSummary?: string;
  transcriptPath?: string;
  /** Real token usage read from the Claude Code transcript (not estimated). */
  contextTokens?: number;
  /** Working directory of the agent; its tail segment becomes the label. */
  cwd?: string;
  /** Breakpoint: when true, the server parks the next PreToolUse call. */
  paused?: boolean;
}

class TraceStore {
  private _sessions: Map<string, TraceSession> = new Map();
  private _activeSessionId: string | null = null;
  private _onDidUpdate = new vscode.EventEmitter<TraceSession>();
  private _onAnomaly = new vscode.EventEmitter<{ session: TraceSession; record: AnomalyRecord }>();

  readonly onDidUpdate = this._onDidUpdate.event;
  /** Fires once per anomaly ONSET (not per re-evaluation of an ongoing one). */
  readonly onAnomaly = this._onAnomaly.event;

  startSession(sessionId: string): TraceSession {
    const count = this._sessions.size + 1;
    const session: TraceSession = {
      id: sessionId,
      label: `Session ${count}`,
      startedAt: Date.now(),
      stopped: false,
      events: [],
      nodes: [],
      anomalyHistory: [],
    };
    this._sessions.set(sessionId, session);
    this._activeSessionId = sessionId;
    // Verification log: each distinct agent session shows up once here.
    console.log(`[TraceBack] tracking new session ${sessionId} (${this._sessions.size} total)`);
    return session;
  }

  getActiveSession(): TraceSession | null {
    if (!this._activeSessionId) return null;
    return this._sessions.get(this._activeSessionId) ?? null;
  }

  getOrCreateActiveSession(): TraceSession {
    if (this._activeSessionId) {
      const s = this._sessions.get(this._activeSessionId);
      if (s) return s;
    }
    const id = `session-${Date.now()}`;
    return this.startSession(id);
  }

  /**
   * Route an incoming event to the session named by its own sessionId,
   * creating that session if it does not exist yet. The most recently
   * active session id is updated so the single-session view follows the
   * latest agent, while all sessions remain individually addressable for
   * the swimlane view.
   */
  getOrCreateSession(sessionId: string): TraceSession {
    let s = this._sessions.get(sessionId);
    if (!s) s = this.startSession(sessionId);
    this._activeSessionId = sessionId;
    return s;
  }

  getAllSessions(): TraceSession[] {
    return Array.from(this._sessions.values()).sort((a, b) => b.startedAt - a.startedAt);
  }

  getSession(id: string): TraceSession | undefined {
    return this._sessions.get(id);
  }

  addEvent(event: TraceEvent): void {
    const session = this.getOrCreateSession(event.sessionId);
    // Claude Code fires Stop at the end of every TURN, not the session — so
    // any new activity after a Stop means the session is live again. Without
    // this reset, sessions were stuck "DONE" (hiding the pause button) after
    // their first completed turn.
    if (event.kind === 'stop') session.stopped = true;
    else if (session.stopped) session.stopped = false;
    session.events.push(event);
    const raw = buildNodes(session.events, session.stopped);
    session.nodes = applyBatchGrouping(raw);

    // Recompute anomaly state on every event (O(1), tail-only). A finished
    // session can't be anomalous; otherwise the state self-clears as soon as
    // the triggering condition stops holding.
    this._setAnomaly(
      session,
      session.stopped ? undefined : detectAnomaly(session.events, Date.now())
    );

    if (event.transcriptPath) session.transcriptPath = event.transcriptPath;
    if (event.cwd && session.cwd !== event.cwd) {
      session.cwd   = event.cwd;
      session.label = event.cwd.split('/').filter(Boolean).pop() ?? session.label;
    }
    this._maybeRefreshTokens(session);

    this._onDidUpdate.fire(session);
  }

  /**
   * Throttled async refresh of real token usage from the session transcript.
   * At most one tail-read per session per 2s; fires an update only when the
   * number actually changes.
   */
  private _tokenReadAt = new Map<string, number>();

  private _maybeRefreshTokens(session: TraceSession): void {
    const path = session.transcriptPath;
    if (!path) return;
    const now  = Date.now();
    const last = this._tokenReadAt.get(session.id) ?? 0;
    if (now - last < 2000) return;
    this._tokenReadAt.set(session.id, now);

    void readContextTokens(path).then((tokens) => {
      if (tokens === null || tokens === session.contextTokens) return;
      session.contextTokens = tokens;
      this._onDidUpdate.fire(session);
    });
  }

  /**
   * Periodic re-evaluation for time-based anomalies (the Silent Stall can
   * only trip via the clock, never via an incoming event). Fires an update
   * only when the anomaly state actually changes.
   */
  checkStalls(): void {
    for (const session of this._sessions.values()) {
      // A paused session is intentionally frozen — not a stall.
      if (session.stopped || session.paused) continue;
      const next = detectAnomaly(session.events, Date.now());
      const prev = session.anomaly;
      const changed =
        next.isAnomalous !== (prev?.isAnomalous ?? false) ||
        next.reason !== prev?.reason;
      if (changed) {
        this._setAnomaly(session, next);
        this._onDidUpdate.fire(session);
      }
    }
  }

  /**
   * Single write-path for anomaly state. The live state self-clears on
   * recovery, but each ONSET (none→anomalous, or the type changing) appends a
   * permanent AnomalyRecord and fires onAnomaly exactly once — re-evaluations
   * of an ongoing anomaly (e.g. a stall's seconds counter) do not re-fire.
   */
  private _setAnomaly(session: TraceSession, next: AnomalyState | undefined): void {
    const prev = session.anomaly;
    session.anomaly = next;
    const isOnset =
      next?.isAnomalous && next.type &&
      (!prev?.isAnomalous || prev.type !== next.type);
    if (isOnset) {
      const record: AnomalyRecord = {
        type:            next.type!,
        reason:          next.reason ?? 'Anomaly detected',
        flaggedEventIds: next.flaggedEventIds,
        detectedAt:      Date.now(),
      };
      session.anomalyHistory.push(record);
      this._onAnomaly.fire({ session, record });
    }
  }

  setPaused(sessionId: string, paused: boolean): void {
    const session = this._sessions.get(sessionId);
    if (!session) return;
    session.paused = paused;
    this._onDidUpdate.fire(session);
  }

  setAiSummary(sessionId: string, summary: string): void {
    const session = this._sessions.get(sessionId);
    if (!session) return;
    session.aiSummary = summary;
    this._onDidUpdate.fire(session);
  }

  clearActive(): void {
    // Wipe every session so Clear gives a genuinely empty canvas. Leaving old
    // sessions behind kept the view locked in swimlane mode and leaked stale
    // (e.g. previous-project) nodes into new runs.
    this._sessions.clear();
    const id = `session-${Date.now()}`;
    this.startSession(id);
    this._onDidUpdate.fire(this.getActiveSession()!);
  }

  dispose(): void {
    this._onDidUpdate.dispose();
    this._onAnomaly.dispose();
  }
}

// ─── Pass 1: build flat node list ────────────────────────────────────────────

function buildNodes(events: TraceEvent[], stopped: boolean): TraceNode[] {
  const nodes: TraceNode[] = [];

  for (const event of events) {
    if (event.kind === 'pre_tool_use') {
      const label = buildLabel(event.toolName ?? 'Unknown', event.toolInput);

      // Remove the trailing "Thinking…" node — a real tool is starting now
      const last = nodes[nodes.length - 1];
      if (last?.toolName === '__thinking__') nodes.pop();

      // Collapse consecutive identical tool+label pairs into one node with a count badge
      const prev = nodes[nodes.length - 1];
      if (
        prev &&
        prev.status === 'success' &&
        prev.toolName === event.toolName &&
        prev.label === label
      ) {
        prev.count += 1;
        prev.eventIds.push(event.id);
      } else {
        nodes.push({
          id: event.id,
          toolName: event.toolName ?? 'Unknown',
          status: 'pending',
          label,
          count: 1,
          eventIds: [event.id],
          timestamp: event.timestamp,
          toolInput: event.toolInput,
        });
      }
      continue;
    }

    if (event.kind === 'post_tool_use') {
      const pending = [...nodes].reverse().find(
        (n) => n.toolName === event.toolName && n.status === 'pending'
      );
      if (pending) {
        pending.status = event.isError ? 'error' : 'success';
        pending.detail = event.toolResponse;
        pending.durationMs = Math.max(0, event.timestamp - pending.timestamp);
        pending.eventIds.push(event.id);
      }

      if (!event.isError && !stopped) {
        nodes.push({
          id: `thinking-${event.id}`,
          toolName: '__thinking__',
          status: 'thinking',
          label: 'Thinking…',
          count: 1,
          eventIds: [],
          timestamp: event.timestamp + 1,
        });
      }
      continue;
    }

    if (event.kind === 'stop') {
      const last = nodes[nodes.length - 1];
      if (last?.toolName === '__thinking__') nodes.pop();
      nodes.forEach((n) => {
        if (n.status === 'pending') n.status = 'success';
      });
    }
  }

  return nodes;
}

// ─── Pass 2: batch-group consecutive same-tool runs ──────────────────────────

const BATCH_THRESHOLD = 3;

function applyBatchGrouping(nodes: TraceNode[]): TraceNode[] {
  const result: TraceNode[] = [];
  let i = 0;

  while (i < nodes.length) {
    const current = nodes[i];

    if (current.toolName === '__thinking__') {
      result.push(current);
      i++;
      continue;
    }

    let runEnd = i + 1;
    while (
      runEnd < nodes.length &&
      nodes[runEnd].toolName === current.toolName &&
      nodes[runEnd].toolName !== '__thinking__'
    ) {
      runEnd++;
    }

    const runLength = runEnd - i;

    if (runLength >= BATCH_THRESHOLD) {
      const batchNodes = nodes.slice(i, runEnd);
      const hasError   = batchNodes.some((n) => n.status === 'error');
      const hasPending = batchNodes.some((n) => n.status === 'pending');

      result.push({
        id: `batch-${current.id}`,
        toolName: current.toolName,
        status: hasError ? 'error' : hasPending ? 'pending' : 'success',
        label: `${runLength} steps`,
        count: runLength,
        eventIds: batchNodes.flatMap((n) => n.eventIds),
        isBatch: true,
        durationMs: batchNodes.reduce((sum, n) => sum + (n.durationMs ?? 0), 0) || undefined,
        batchItems: batchNodes.map((n) => ({
          label: n.label,
          detail: n.detail,
          status: n.status,
          durationMs: n.durationMs,
        })),
        detail: batchNodes
          .map((n, idx) => `── [${idx + 1}/${runLength}] ${n.label}\n${n.detail ?? '(no output)'}`)
          .join('\n\n'),
        timestamp: current.timestamp,
      });

      i = runEnd;
    } else {
      result.push(current);
      i++;
    }
  }

  return result;
}

// ─── Label extraction ─────────────────────────────────────────────────────────

function buildLabel(toolName: ToolName, input?: Record<string, unknown>): string {
  const desc = input?.description as string | undefined;

  switch (toolName) {
    case 'Read':
      return desc ? truncate(desc, 55) : shortenPath(input?.file_path as string);

    case 'Edit':
    case 'Write':
      return desc ? truncate(desc, 55) : shortenPath((input?.file_path ?? input?.path) as string);

    case 'Bash':
      return desc ? truncate(desc, 55) : truncate(input?.command as string, 45);

    case 'WebSearch':
      return desc ? truncate(desc, 55) : truncate(input?.query as string, 50);

    case 'WebFetch':
      return desc ? truncate(desc, 55) : truncate(input?.url as string, 50);

    case 'Agent':
      return desc ? truncate(desc, 55) : 'Subagent';

    default:
      return desc ? truncate(desc, 55) : toolName;
  }
}

function shortenPath(p?: string): string {
  if (!p) return 'file';
  const parts = p.split('/');
  return parts.slice(-2).join('/');
}

function truncate(s?: string, max = 40): string {
  if (!s) return '';
  return s.length > max ? s.slice(0, max) + '…' : s;
}

export const traceStore = new TraceStore();
