import * as vscode from 'vscode';
import { AnomalyRecord, AnomalyState, detectAnomaly } from './anomalyDetector';
import { extractIntentForTool, noteTranscript, readContextTokens } from './tokenReader';

export type EventKind =
  | 'pre_tool_use'
  | 'post_tool_use'
  | 'stop'
  | 'notification'
  | 'user_prompt';

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
  /** First sentence of the assistant text preceding this call (transcript).
   *  Best-effort and always optional — attached async after PostToolUse. */
  intent?: string;
}

export interface BatchItem {
  label: string;
  detail?: string;
  status: NodeStatus;
  durationMs?: number;
  toolInput?: Record<string, unknown>;
  intent?: string;
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
  /** The in-progress plan item this call ran under (from TodoWrite state). */
  objective?: string;
  /** TodoWrite call — rendered as a quiet "plan updated" divider, not a card. */
  isPlanUpdate?: boolean;
  /** Why the agent made this call — from the transcript, may arrive late. */
  intent?: string;
}

export interface PlanItem {
  /** Task id for TaskCreate/TaskUpdate flows ("1", "2", …); absent for TodoWrite. */
  id?: string;
  content: string;
  status: 'pending' | 'in_progress' | 'completed';
  activeForm?: string;
}

export interface SessionPlan {
  items: PlanItem[];
  updatedAt: number;
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
  /** Set while Claude waits on the user (permission prompt / idle input);
   *  holds the notification message. Cleared by any subsequent activity. */
  awaitingInput?: string;
  /** Live task plan from the agent's TodoWrite calls. */
  plan?: SessionPlan;
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

    // "Waiting for you" state: a Notification means Claude is blocked on the
    // user (permission prompt, idle). Any other activity clears it.
    if (event.kind === 'notification') {
      session.awaitingInput = event.toolResponse || 'Waiting for your input';
    } else {
      session.awaitingInput = undefined;
    }

    session.events.push(event);
    this._rebuildNodes(session);

    // Recompute anomaly state on every event (O(1), tail-only). A finished
    // session can't be anomalous; otherwise the state self-clears as soon as
    // the triggering condition stops holding.
    this._setAnomaly(
      session,
      session.stopped ? undefined : detectAnomaly(session.events, Date.now())
    );

    if (event.transcriptPath) {
      session.transcriptPath = event.transcriptPath;
      noteTranscript(session.id, event.transcriptPath);
    }
    if (event.cwd && session.cwd !== event.cwd) {
      session.cwd   = event.cwd;
      session.label = event.cwd.split('/').filter(Boolean).pop() ?? session.label;
    }
    this._maybeRefreshTokens(session);
    if (event.kind === 'post_tool_use') this._attachIntent(session, event);

    this._onDidUpdate.fire(session);
  }

  private _rebuildNodes(session: TraceSession): void {
    const { nodes: raw, plan } = buildNodes(session.events, session.stopped);
    session.nodes = applyBatchGrouping(raw);
    session.plan  = plan;
  }

  /**
   * Best-effort intent extraction for the call that just completed: the
   * transcript now contains the assistant text written right before this
   * tool_use. Attached to the PRE event (nodes are rebuilt from events, so
   * anything stored on a node alone would be lost on the next rebuild).
   * Never blocks — the UI renders with or without it.
   */
  private _attachIntent(session: TraceSession, post: TraceEvent): void {
    if (PLAN_TOOLS.has(post.toolName ?? '')) return;

    let preIndex = -1;
    let toolCallIndex = -1;
    for (let i = session.events.length - 1; i >= 0; i--) {
      const e = session.events[i];
      if (e.kind === 'pre_tool_use' && e.toolName === post.toolName && !e.intent) {
        preIndex = i;
        break;
      }
    }
    if (preIndex === -1) return;
    // 0-based index of this call among all the session's tool calls.
    toolCallIndex = session.events
      .slice(0, preIndex + 1)
      .filter((e) => e.kind === 'pre_tool_use').length - 1;

    const preEvent = session.events[preIndex];
    void extractIntentForTool(session.id, toolCallIndex).then((intent) => {
      if (!intent || preEvent.intent) return;
      preEvent.intent = intent;
      this._rebuildNodes(session);
      this._onDidUpdate.fire(session);
    });
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
        severity:        next.severity ?? 'medium',
        title:           next.title ?? 'Anomaly',
        description:     next.description ?? next.reason ?? 'Anomaly detected',
        reason:          next.reason ?? next.description ?? 'Anomaly detected',
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

/** Extract a validated plan from TodoWrite tool input; null when malformed. */
export function parsePlanItems(input?: Record<string, unknown>): PlanItem[] | null {
  if (!input || !Array.isArray(input.todos)) return null;
  const items: PlanItem[] = [];
  for (const raw of input.todos as Record<string, unknown>[]) {
    if (!raw || typeof raw.content !== 'string') continue;
    const status = raw.status === 'in_progress' || raw.status === 'completed'
      ? raw.status
      : 'pending';
    items.push({
      content: raw.content,
      status,
      activeForm: typeof raw.activeForm === 'string' ? raw.activeForm : undefined,
    });
  }
  return items.length > 0 ? items : null;
}

/** Tool names that mutate the agent's plan — two generations of Claude Code:
 *  TodoWrite (full-list snapshots) and TaskCreate/TaskUpdate (incremental). */
const PLAN_TOOLS = new Set(['TodoWrite', 'TaskCreate', 'TaskUpdate']);

interface BuildResult {
  nodes: TraceNode[];
  plan?: SessionPlan;
}

function buildNodes(events: TraceEvent[], stopped: boolean): BuildResult {
  const nodes: TraceNode[] = [];
  // The in-progress task at each point in the stream — new tool nodes are
  // stamped with it so the UI can group effort by objective.
  let activeObjective: string | undefined;
  let plan: SessionPlan | undefined;
  let taskCounter = 0; // TaskCreate ids are assigned sequentially per session

  function refreshObjective() {
    const current = plan?.items.find((i) => i.status === 'in_progress');
    if (current) activeObjective = current.activeForm ?? current.content;
  }

  function applyPlanTool(event: TraceEvent) {
    const input = event.toolInput;
    if (event.toolName === 'TodoWrite') {
      const items = parsePlanItems(input);
      if (items) plan = { items, updatedAt: event.timestamp };
    } else if (event.toolName === 'TaskCreate') {
      if (typeof input?.subject !== 'string') return;
      taskCounter++;
      const item: PlanItem = {
        id:         String(taskCounter),
        content:    input.subject,
        status:     'pending',
        activeForm: typeof input.activeForm === 'string' ? input.activeForm : undefined,
      };
      plan = { items: [...(plan?.items ?? []), item], updatedAt: event.timestamp };
    } else if (event.toolName === 'TaskUpdate') {
      const rawId = input?.taskId;
      const id =
        typeof rawId === 'string' ? rawId.replace(/^#/, '')
        : typeof rawId === 'number' ? String(rawId)
        : null;
      if (!id || !plan) return;
      const items = plan.items
        .filter((it) => !(it.id === id && input?.status === 'deleted'))
        .map((it) => it.id !== id ? it : {
          ...it,
          status:
            input?.status === 'in_progress' || input?.status === 'completed'
              ? input.status
              : it.status,
          activeForm: typeof input?.activeForm === 'string' ? input.activeForm : it.activeForm,
          content:    typeof input?.subject === 'string' ? input.subject : it.content,
        });
      plan = { items, updatedAt: event.timestamp };
    }
    refreshObjective();
  }

  for (const event of events) {
    if (event.kind === 'pre_tool_use') {
      if (PLAN_TOOLS.has(event.toolName ?? '')) applyPlanTool(event);
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
          objective: activeObjective,
          isPlanUpdate: PLAN_TOOLS.has(event.toolName ?? '') || undefined,
          intent: event.intent,
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

    if (event.kind === 'user_prompt' && event.toolResponse) {
      const last = nodes[nodes.length - 1];
      if (last?.toolName === '__thinking__') nodes.pop();
      nodes.push({
        id: event.id,
        toolName: '__prompt__',
        status: 'success',
        label: truncate(event.toolResponse.replace(/\s+/g, ' '), 120),
        detail: event.toolResponse,
        count: 1,
        eventIds: [event.id],
        timestamp: event.timestamp,
      });
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

  return { nodes, plan };
}

// ─── Pass 2: batch-group consecutive same-tool runs ──────────────────────────

const BATCH_THRESHOLD = 3;

function applyBatchGrouping(nodes: TraceNode[]): TraceNode[] {
  const result: TraceNode[] = [];
  let i = 0;

  while (i < nodes.length) {
    const current = nodes[i];

    if (current.toolName.startsWith('__') || current.isPlanUpdate) {
      result.push(current);
      i++;
      continue;
    }

    let runEnd = i + 1;
    while (
      runEnd < nodes.length &&
      nodes[runEnd].toolName === current.toolName &&
      !nodes[runEnd].toolName.startsWith('__') &&
      !nodes[runEnd].isPlanUpdate
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
          toolInput: n.toolInput,
          intent: n.intent,
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
