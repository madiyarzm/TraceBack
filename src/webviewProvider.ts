import * as vscode from 'vscode';
import { TraceSession, traceStore } from './traceStore';

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
    private readonly _outputChannel: vscode.OutputChannel
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

    webviewView.webview.html = this._getHtml(webviewView.webview);

    webviewView.webview.onDidReceiveMessage((message) => {
      this._handleWebviewMessage(message);
    });

    this._outputChannel.appendLine('[TraceBack] Webview sidebar resolved.');
  }

  postSessionUpdate(session: TraceSession): void {
    const payload = {
      type: 'session_update',
      session,
      // Only surface sessions that actually have activity. Empty sessions
      // (e.g. the fresh session created by Clear, or a session that errored
      // before any tool ran) must not spawn ghost swimlanes or trip the
      // swimlane-mode threshold.
      allSessions: traceStore.getAllSessions()
        .filter((s) => s.nodes.some((n) => n.toolName !== '__thinking__'))
        .map((s) => ({
        id:           s.id,
        label:        s.label,
        startedAt:    s.startedAt,
        nodeCount:    s.nodes.filter((n) => n.toolName !== '__thinking__').length,
        stopped:      s.stopped,
        nodes:        s.nodes,
        anomaly:       s.anomaly,
        aiSummary:     s.aiSummary,
        contextTokens: s.contextTokens,
        cwd:           s.cwd,
        })),
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

    this._panel.webview.html = this._getHtml(this._panel.webview);

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
        const active = traceStore.getActiveSession();
        if (active) this.postSessionUpdate(active);
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
      case 'llm_query':
        this._onLlmQuery.fire({
          question:    (message.question as string) ?? '',
          nodeContext: (message.nodeContext as string) ?? '',
        });
        break;
      default:
        this._outputChannel.appendLine(`[TraceBack] Unknown webview message: ${message.type}`);
    }
  }

  private _getHtml(webview: vscode.Webview): string {
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
