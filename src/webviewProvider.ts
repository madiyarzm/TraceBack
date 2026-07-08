import * as fs from 'fs';
import * as vscode from 'vscode';
import { TraceSession, traceStore } from './traceStore';
import { flushSession } from './server';
import { listSessions, loadSession } from './sessionArchive';
import {
  addCustomGuard, BuiltinGuardKey, getGuardsState, removeCustomGuard, setBuiltinGuard,
} from './guardsManager';

export interface LLMQueryEvent {
  question:    string;
  nodeContext: string;
}

export class TracebackWebviewProvider implements vscode.WebviewViewProvider {
  private _view?:  vscode.WebviewView;
  private _panel?: vscode.WebviewPanel;

  private _onLlmQuery = new vscode.EventEmitter<LLMQueryEvent>();
  readonly onLlmQuery = this._onLlmQuery.event;

  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly _outputChannel: vscode.OutputChannel,
    private readonly _archiveDir: string
  ) {}

  resolveWebviewView(
    webviewView:  vscode.WebviewView,
    _context:     vscode.WebviewViewResolveContext,
    _token:       vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this._extensionUri, 'webview', 'dist'),
      ],
    };

    webviewView.webview.html = this._getHtml(webviewView.webview, 'sidebar');

    webviewView.webview.onDidReceiveMessage((message) => {
      this._handleWebviewMessage(message);
    });

    this._outputChannel.appendLine('[TraceBack] Webview sidebar resolved.');
  }

  postSessionUpdate(session: TraceSession): void {
    const payload = {
      type: 'session_update',
      // The webview only needs which session updated (it renders from
      // `allSessions`); shipping the full object would leak whole-file
      // baselines into every update. Send just the id.
      session: { id: session.id },
      // Only surface sessions that actually have activity. Empty sessions
      // (e.g. the fresh session created by Clear, or a session that errored
      // before any tool ran) must not spawn ghost swimlanes or trip the
      // swimlane-mode threshold.
      allSessions: traceStore.getAllSessions()
        .filter((s) => s.nodes.some((n) => !n.toolName.startsWith('__')))
        .map((s) => ({
        id:           s.id,
        label:        s.label,
        startedAt:    s.startedAt,
        nodeCount:    s.nodes.filter((n) => !n.toolName.startsWith('__')).length,
        stopped:      s.stopped,
        nodes:        s.nodes,
        anomaly:        s.anomaly,
        anomalyHistory: s.anomalyHistory,
        aiSummary:      s.aiSummary,
        contextTokens: s.contextTokens,
        cwd:           s.cwd,
        paused:        s.paused,
        awaitingInput: s.awaitingInput,
        plan:          s.plan,
        ledger:        s.ledger,
        })),
      history: listSessions(this._archiveDir),
    };
    this._view?.webview.postMessage(payload);
    this._panel?.webview.postMessage(payload);
  }

  postLlmResponse(answer: string): void {
    const msg = { type: 'llm_response', answer };
    this._view?.webview.postMessage(msg);
    this._panel?.webview.postMessage(msg);
  }

  openFullPanel(extensionUri: vscode.Uri): void {
    if (this._panel) {
      this._panel.reveal(vscode.ViewColumn.One);
      return;
    }

    this._panel = vscode.window.createWebviewPanel(
      'traceback.fullMap',
      'TraceBack — Action Map',
      vscode.ViewColumn.One,
      {
        enableScripts:          true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(extensionUri, 'webview', 'dist'),
        ],
      }
    );

    this._panel.webview.html = this._getHtml(this._panel.webview, 'panel');

    this._panel.webview.onDidReceiveMessage((message) => {
      this._handleWebviewMessage(message);
    });

    this._panel.onDidDispose(() => {
      this._panel = undefined;
    });
  }

  private _handleWebviewMessage(message: { type: string; [key: string]: unknown }): void {
    switch (message.type) {
      case 'ready': {
        this._postGuards();
        const active = traceStore.getActiveSession();
        if (active) {
          this.postSessionUpdate(active);
        } else {
          // No live session yet — still surface archived history in the panel.
          const msg = { type: 'history_update', history: listSessions(this._archiveDir) };
          this._view?.webview.postMessage(msg);
          this._panel?.webview.postMessage(msg);
        }
        break;
      }
      case 'open_full_panel':
        vscode.commands.executeCommand('traceback.openMap');
        break;
      case 'load_archived': {
        if (typeof message.sessionId === 'string') {
          const archived = loadSession(this._archiveDir, message.sessionId);
          if (archived) {
            const msg = { type: 'archived_session', session: archived };
            this._view?.webview.postMessage(msg);
            this._panel?.webview.postMessage(msg);
          }
        }
        break;
      }
      case 'open_file':
        if (typeof message.filePath === 'string') {
          vscode.workspace.openTextDocument(message.filePath).then((doc) => {
            vscode.window.showTextDocument(doc, { preview: true });
          });
        }
        break;
      case 'switch_session':
        if (typeof message.sessionId === 'string') {
          const session = traceStore.getSession(message.sessionId);
          if (session) this.postSessionUpdate(session);
        }
        break;
      case 'clear_session':
        traceStore.clearActive();
        break;
      case 'pause_session':
        if (typeof message.sessionId === 'string') {
          traceStore.setPaused(message.sessionId, true);
        }
        break;
      case 'resume_session':
        if (typeof message.sessionId === 'string') {
          traceStore.setPaused(message.sessionId, false);
          flushSession(message.sessionId, 'allow');
        }
        break;
      case 'redirect_session':
        if (typeof message.sessionId === 'string') {
          const reason = typeof message.message === 'string' && message.message.trim()
            ? message.message.trim()
            : undefined;
          flushSession(message.sessionId, 'deny', reason);
          traceStore.setPaused(message.sessionId, false);
        }
        break;
      case 'llm_query':
        this._onLlmQuery.fire({
          question:    (message.question as string) ?? '',
          nodeContext: (message.nodeContext as string) ?? '',
        });
        break;
      case 'set_guard':
        if (typeof message.key === 'string') {
          void setBuiltinGuard(message.key as BuiltinGuardKey, message.enabled === true)
            .then(() => this._postGuards());
        }
        break;
      case 'add_custom_guard':
        if (typeof message.pattern === 'string') {
          addCustomGuard(message.pattern)
            .then(() => this._postGuards())
            .catch((err) => {
              vscode.window.showWarningMessage(`TraceBack: ${err.message ?? err}`);
              this._postGuards();
            });
        }
        break;
      case 'remove_custom_guard':
        if (typeof message.pattern === 'string') {
          void removeCustomGuard(message.pattern).then(() => this._postGuards());
        }
        break;
      case 'get_guards':
        this._postGuards();
        break;
      case 'request_review':
        if (typeof message.sessionId === 'string') {
          void this._postReviewData(message.sessionId);
        }
        break;
      default:
        this._outputChannel.appendLine(`[TraceBack] Unknown webview message: ${message.type}`);
    }
  }

  private _postGuards(): void {
    const msg = { type: 'guards_update', guards: getGuardsState() };
    this._view?.webview.postMessage(msg);
    this._panel?.webview.postMessage(msg);
  }

  /**
   * Net-change review data: for every file baselined during the session, pair
   * the pre-edit snapshot with the file's CURRENT content on disk. Sent on
   * demand only — baselines are whole files and must not ride every
   * session_update.
   */
  private async _postReviewData(sessionId: string): Promise<void> {
    const session = traceStore.getSession(sessionId);
    const baselines = session?.baselines ?? {};

    const files = await Promise.all(
      Object.entries(baselines).map(async ([path, baseline]) => {
        let current: string | null = null;
        try {
          current = await fs.promises.readFile(path, 'utf8');
        } catch {
          current = null; // deleted since (or unreadable)
        }
        return { path, baseline: baseline.content, current };
      })
    );

    const msg = { type: 'review_data', sessionId, files };
    this._view?.webview.postMessage(msg);
    this._panel?.webview.postMessage(msg);
  }

  private _getHtml(webview: vscode.Webview, mode: 'sidebar' | 'panel'): string {
    const distUri   = vscode.Uri.joinPath(this._extensionUri, 'webview', 'dist');
    // Cache-bust: the bundle filename is stable (no content hash), so without a
    // changing query string the webview serves a stale cached copy across reloads.
    const v         = Date.now();
    const scriptUri = `${webview.asWebviewUri(vscode.Uri.joinPath(distUri, 'assets', 'index.js'))}?v=${v}`;
    const styleUri  = `${webview.asWebviewUri(vscode.Uri.joinPath(distUri, 'assets', 'index.css'))}?v=${v}`;

    const nonce = getNonce();
    const csp = [
      `default-src 'none'`,
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `script-src 'nonce-${nonce}'`,
      `img-src ${webview.cspSource} data: blob:`,
      `font-src ${webview.cspSource}`,
    ].join('; ');

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="${styleUri}" />
  <title>TraceBack</title>
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}">window.__TB_MODE__ = '${mode}';</script>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }

  dispose(): void {
    this._onLlmQuery.dispose();
  }
}

function getNonce(): string {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
