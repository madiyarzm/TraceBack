import * as vscode from 'vscode';

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
  timestamp: number;
}

export interface BatchItem {
  label: string;
  detail?: string;
  status: NodeStatus;
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
  isLooping?: boolean;
  timestamp: number;
  isBatch?: boolean;
  batchItems?: BatchItem[];
}

export interface StumbleAlert {
  type: 'loop' | 'timeout';
  toolName: string;
  nodeId: string;
  label: string;
  detectedAt: number;
}

export interface TraceSession {
  id: string;
  label: string;
  startedAt: number;
  stopped: boolean;
  events: TraceEvent[];
  nodes: TraceNode[];
  stumbleAlert?: StumbleAlert;
  aiSummary?: string;
}

class TraceStore {
  private _sessions: Map<string, TraceSession> = new Map();
  private _activeSessionId: string | null = null;
  private _onDidUpdate = new vscode.EventEmitter<TraceSession>();

  readonly onDidUpdate = this._onDidUpdate.event;

  startSession(sessionId: string): TraceSession {
    const count = this._sessions.size + 1;
    const session: TraceSession = {
      id: sessionId,
      label: `Session ${count}`,
      startedAt: Date.now(),
      stopped: false,
      events: [],
      nodes: [],
    };
    this._sessions.set(sessionId, session);
    this._activeSessionId = sessionId;
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

  getAllSessions(): TraceSession[] {
    return Array.from(this._sessions.values()).sort((a, b) => b.startedAt - a.startedAt);
  }

  getSession(id: string): TraceSession | undefined {
    return this._sessions.get(id);
  }

  addEvent(event: TraceEvent): void {
    const session = this.getOrCreateActiveSession();
    if (event.kind === 'stop') session.stopped = true;
    session.events.push(event);
    const raw = buildNodes(session.events, session.stopped);
    session.nodes = applyBatchGrouping(raw);

    // Loop detection: same tool + same label repeated 3× in sequence
    if (event.kind === 'pre_tool_use' && !session.stumbleAlert) {
      const alert = detectLoop(raw);
      if (alert) {
        session.stumbleAlert = alert;
        // Mark the three looping nodes so the UI can apply a halo animation
        const real = raw.filter((n) => n.toolName !== '__thinking__');
        for (const n of real.slice(-3)) n.isLooping = true;
      }
    }
    // Clear stumble alert when session finishes cleanly
    if (session.stopped) session.stumbleAlert = undefined;

    this._onDidUpdate.fire(session);
  }

  checkTimeouts(): void {
    const TIMEOUT_MS = 45_000;
    for (const session of this._sessions.values()) {
      if (session.stopped || session.stumbleAlert) continue;
      const pendingNode = session.nodes.find(
        (n) => n.status === 'pending' && n.toolName !== '__thinking__'
      );
      if (pendingNode && Date.now() - pendingNode.timestamp > TIMEOUT_MS) {
        session.stumbleAlert = {
          type: 'timeout',
          toolName: pendingNode.toolName,
          nodeId: pendingNode.id,
          label: pendingNode.label,
          detectedAt: Date.now(),
        };
        this._onDidUpdate.fire(session);
      }
    }
  }

  setAiSummary(sessionId: string, summary: string): void {
    const session = this._sessions.get(sessionId);
    if (!session) return;
    session.aiSummary = summary;
    this._onDidUpdate.fire(session);
  }

  clearActive(): void {
    const id = `session-${Date.now()}`;
    this.startSession(id);
    this._onDidUpdate.fire(this.getActiveSession()!);
  }

  dispose(): void {
    this._onDidUpdate.dispose();
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
        batchItems: batchNodes.map((n) => ({
          label: n.label,
          detail: n.detail,
          status: n.status,
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

// ─── Loop detection ───────────────────────────────────────────────────────────

function detectLoop(nodes: TraceNode[]): StumbleAlert | undefined {
  const real = nodes.filter((n) => n.toolName !== '__thinking__');
  if (real.length < 3) return undefined;
  const last3 = real.slice(-3);
  if (
    last3[0].toolName === last3[1].toolName &&
    last3[1].toolName === last3[2].toolName &&
    last3[0].label === last3[1].label &&
    last3[1].label === last3[2].label
  ) {
    return {
      type: 'loop',
      toolName: last3[2].toolName,
      nodeId: last3[2].id,
      label: last3[2].label,
      detectedAt: Date.now(),
    };
  }
  return undefined;
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
