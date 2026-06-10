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
          handleHookPayload(payload);
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
  if (!_server) return;
  _server.close(() => {
    _outputChannel.appendLine('[TraceBack] Server stopped.');
  });
  _server = null;
}

// Maps the raw Claude Code hook payload into our internal TraceEvent shape.
// Claude Code sends different shapes for PreToolUse vs PostToolUse vs Stop.
function handleHookPayload(payload: Record<string, unknown>): void {
  const hookType = payload.hook_event_name as string | undefined;

  if (!hookType) {
    _outputChannel.appendLine('[TraceBack] Received payload with no hook_event_name, skipping.');
    return;
  }

  _outputChannel.appendLine(`[TraceBack] Hook received: ${hookType}`);

  const kind = hookTypeToKind(hookType);
  if (!kind) return;

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
