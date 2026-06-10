import * as http from 'http';
import * as vscode from 'vscode';
import { traceStore, TraceEvent, EventKind } from './traceStore';

let _server: http.Server | null = null;
let _outputChannel: vscode.OutputChannel;

export function startServer(outputChannel: vscode.OutputChannel, port: number): void {
  _outputChannel = outputChannel;

  if (_server) {
    _outputChannel.appendLine(`[TraceBack] Server already running on :${port}`);
    return;
  }

  _server = http.createServer((req, res) => {
    // Allow CORS for any local webview requests
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    if (req.method === 'POST' && req.url === '/event') {
      let body = '';
      req.on('data', (chunk) => (body += chunk));
      req.on('end', () => {
        try {
          const payload = JSON.parse(body);
          const event = handleHookPayload(payload);

          // Tripwires: static policy rules checked BEFORE execution. A match
          // denies the call immediately, fleet-wide, no human in the loop.
          if (event && event.kind === 'pre_tool_use') {
            const tripped = checkTripwires(event);
            if (tripped) {
              denyCall(event, res, tripped);
              return;
            }
          }

          // Breakpoint gate: while a session is paused, PreToolUse responses
          // are held open — Claude Code waits on the curl, freezing the agent
          // at the boundary of its next tool call. resume/redirect resolves it.
          if (
            event &&
            event.kind === 'pre_tool_use' &&
            traceStore.getSession(event.sessionId)?.paused
          ) {
            parkCall(event, res);
            return;
          }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true }));
        } catch (err) {
          _outputChannel.appendLine(`[TraceBack] Failed to parse hook payload: ${err}`);
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      });
      return;
    }

    // Health check endpoint — useful to verify the server is up
    if (req.method === 'GET' && req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', version: '0.1.0' }));
      return;
    }

    res.writeHead(404);
    res.end();
  });

  _server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      vscode.window.showErrorMessage(
        `TraceBack: Port ${port} is already in use. Change it in settings (traceback.port).`
      );
    } else {
      _outputChannel.appendLine(`[TraceBack] Server error: ${err.message}`);
    }
  });

  _server.listen(port, '127.0.0.1', () => {
    _outputChannel.appendLine(`[TraceBack] Listening on http://127.0.0.1:${port}`);
    vscode.window.setStatusBarMessage(`$(pulse) TraceBack listening on :${port}`, 4000);
  });
}

export function stopServer(): void {
  // Release any frozen agents before going down.
  for (const sessionId of Array.from(_parked.keys())) {
    flushSession(sessionId, 'allow');
  }
  if (!_server) return;
  _server.close(() => {
    _outputChannel.appendLine('[TraceBack] Server stopped.');
  });
  _server = null;
}

// ─── Breakpoint gate ──────────────────────────────────────────────────────────
// Held PreToolUse responses, keyed by session. While parked, the agent's curl
// is blocked and the agent is frozen at its next tool call.

interface ParkedCall {
  event: TraceEvent;
  res:   http.ServerResponse;
}

const _parked = new Map<string, ParkedCall[]>();

function parkCall(event: TraceEvent, res: http.ServerResponse): void {
  const list = _parked.get(event.sessionId) ?? [];
  list.push({ event, res });
  _parked.set(event.sessionId, list);
  _outputChannel.appendLine(
    `[TraceBack] ⏸ holding ${event.toolName ?? 'tool'} call (session ${event.sessionId.slice(0, 8)})`
  );
}

/**
 * Resolve all held calls for a session.
 *  - 'allow': respond neutrally — the tool call proceeds as normal.
 *  - 'deny':  respond with a PreToolUse deny decision; `reason` is fed back
 *             into Claude's context verbatim (the human-in-the-loop redirect).
 */
export function flushSession(sessionId: string, decision: 'allow' | 'deny', reason?: string): number {
  const parked = _parked.get(sessionId) ?? [];
  _parked.delete(sessionId);

  for (const { event, res } of parked) {
    try {
      if (decision === 'allow') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
      } else {
        sendDeny(event, res, reason ?? 'Intercepted by user via TraceBack.', 'Intercepted by user');
      }
    } catch (err) {
      _outputChannel.appendLine(`[TraceBack] Failed to flush parked call: ${err}`);
    }
  }
  if (parked.length > 0) {
    _outputChannel.appendLine(
      `[TraceBack] ▶ released ${parked.length} held call(s) (${decision}) for ${sessionId.slice(0, 8)}`
    );
  }
  return parked.length;
}

// Maps the raw Claude Code hook payload into our internal TraceEvent shape.
// Claude Code sends different shapes for PreToolUse vs PostToolUse vs Stop.
function handleHookPayload(payload: Record<string, unknown>): TraceEvent | null {
  const hookType = payload.hook_event_name as string | undefined;

  if (!hookType) {
    _outputChannel.appendLine('[TraceBack] Received payload with no hook_event_name, skipping.');
    return null;
  }

  _outputChannel.appendLine(`[TraceBack] Hook received: ${hookType}`);

  const kind = hookTypeToKind(hookType);
  if (!kind) return null;

  const event: TraceEvent = {
    id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    sessionId: (payload.session_id as string) ?? 'default',
    kind,
    toolName: payload.tool_name as string | undefined,
    toolInput: payload.tool_input as Record<string, unknown> | undefined,
    toolResponse:
      typeof payload.tool_response === 'string'
        ? payload.tool_response
        : JSON.stringify(payload.tool_response),
    isError: hookType === 'PostToolUseFailure' || isErrorResponse(payload.tool_response),
    transcriptPath: payload.transcript_path as string | undefined,
    cwd: payload.cwd as string | undefined,
    timestamp: Date.now(),
  };

  traceStore.addEvent(event);
  return event;
}

/**
 * Defensive error sniffing for PostToolUse payloads. Verified against real
 * transcripts: failed Bash calls return tool_response as a plain string
 * starting with "Error:", while successes return an {stdout, stderr, ...}
 * object. The flag checks cover other tools / future payload shapes.
 */
function isErrorResponse(resp: unknown): boolean {
  if (typeof resp === 'string') return /^Error[:\s]/i.test(resp);
  if (resp && typeof resp === 'object') {
    const r = resp as Record<string, unknown>;
    if (r.is_error === true || r.isError === true) return true;
    if (r.success === false) return true;
    if (typeof r.exitCode === 'number' && r.exitCode !== 0) return true;
    if (typeof r.exit_code === 'number' && r.exit_code !== 0) return true;
    if (r.interrupted === true) return true;
  }
  return false;
}

// ─── Tripwires ────────────────────────────────────────────────────────────────

interface TripwireRule {
  /** Regex matched against the tool name (e.g. "Bash", "Edit|Write"). Empty = any tool. */
  tool?: string;
  /** Regex matched against toolName + serialized tool input. Required. */
  pattern: string;
  /** Custom message fed back to the agent. */
  reason?: string;
}

/** Returns the deny reason if any configured tripwire matches this call. */
function checkTripwires(event: TraceEvent): string | null {
  const rules = vscode.workspace
    .getConfiguration('traceback')
    .get<TripwireRule[]>('tripwires') ?? [];

  for (const rule of rules) {
    if (!rule?.pattern) continue;
    try {
      if (rule.tool && !new RegExp(rule.tool).test(event.toolName ?? '')) continue;
      const haystack = `${event.toolName ?? ''} ${JSON.stringify(event.toolInput ?? {})}`;
      if (new RegExp(rule.pattern, 'i').test(haystack)) {
        return rule.reason ??
          `Blocked by TraceBack tripwire (pattern: ${rule.pattern}). Do not retry this exact call.`;
      }
    } catch (err) {
      _outputChannel.appendLine(`[TraceBack] Invalid tripwire regex "${rule.pattern}": ${err}`);
    }
  }
  return null;
}

function denyCall(event: TraceEvent, res: http.ServerResponse, reason: string): void {
  sendDeny(event, res, reason, 'Tripwire blocked');
  _outputChannel.appendLine(`[TraceBack] ⛔ tripwire blocked ${event.toolName ?? 'tool'}: ${reason}`);
  vscode.window.showWarningMessage(`TraceBack tripwire blocked ${event.toolName ?? 'a tool call'}: ${reason}`);
}

/** Shared deny response (used by tripwires and breakpoint redirects). */
function sendDeny(event: TraceEvent, res: http.ServerResponse, reason: string, label: string): void {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  // Both schemas for compatibility: legacy top-level decision and the
  // current hookSpecificOutput.permissionDecision form.
  res.end(JSON.stringify({
    decision: 'block',
    reason,
    hookSpecificOutput: {
      hookEventName:            'PreToolUse',
      permissionDecision:       'deny',
      permissionDecisionReason: reason,
    },
  }));
  // Synthetic completion so the timeline shows the block instead of an
  // eternally-pending node (a denied call emits no PostToolUse).
  traceStore.addEvent({
    id:           `evt-deny-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    sessionId:    event.sessionId,
    kind:         'post_tool_use',
    toolName:     event.toolName,
    toolResponse: `⛔ ${label}: ${reason}`,
    isError:      true,
    timestamp:    Date.now(),
  });
}

function hookTypeToKind(hookType: string): EventKind | null {
  switch (hookType) {
    case 'PreToolUse':         return 'pre_tool_use';
    case 'PostToolUse':        return 'post_tool_use';
    // Failures arrive on a dedicated event — PostToolUse only fires on success
    case 'PostToolUseFailure': return 'post_tool_use';
    case 'Stop':               return 'stop';
    case 'Notification':       return 'notification';
    default:
      return null;
  }
}
