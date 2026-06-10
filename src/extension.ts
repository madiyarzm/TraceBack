import * as vscode from 'vscode';
import { startServer, stopServer } from './server';
import { installHooks, removeHooks } from './hookManager';
import { traceStore } from './traceStore';
import { TracebackWebviewProvider } from './webviewProvider';
import { callLLM, LLMConfig } from './llmClient';

let outputChannel: vscode.OutputChannel;

function getLLMConfig(): LLMConfig | null {
  const cfg      = vscode.workspace.getConfiguration('traceback');
  const provider = cfg.get<string>('llmProvider') ?? 'disabled';
  if (provider === 'disabled') return null;
  return {
    provider:   provider as 'groq' | 'ollama',
    apiKey:     cfg.get<string>('groqApiKey') ?? '',
    model:      provider === 'groq'
      ? (cfg.get<string>('groqModel')   ?? 'llama-3.1-8b-instant')
      : (cfg.get<string>('ollamaModel') ?? 'llama3.2'),
    ollamaPort: cfg.get<number>('ollamaPort') ?? 11434,
  };
}

const SUMMARY_SYSTEM =
  'You analyze AI coding agent timelines. Respond in 1–2 short, jargon-free sentences. ' +
  'Be specific about what was done and what is likely happening next.';

const CHAT_SYSTEM =
  'You are a debugging assistant analyzing an AI coding agent\'s action timeline. ' +
  'Answer questions about what happened and why, suggest fixes, and be concise and actionable.';

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  outputChannel = vscode.window.createOutputChannel('TraceBack');
  outputChannel.appendLine('[TraceBack] Extension activating...');

  const config      = vscode.workspace.getConfiguration('traceback');
  const port: number       = config.get('port') ?? 7777;
  const autoInstall: boolean = config.get('autoInstallHooks') ?? true;

  startServer(outputChannel, port);

  if (autoInstall) {
    await installHooks(port, outputChannel);
  }

  const provider = new TracebackWebviewProvider(context.extensionUri, outputChannel);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('traceback.mapView', provider, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  // Debounced LLM auto-summary
  let summaryTimer: ReturnType<typeof setTimeout> | null = null;

  context.subscriptions.push(
    traceStore.onDidUpdate((session) => {
      provider.postSessionUpdate(session);

      if (summaryTimer) clearTimeout(summaryTimer);
      summaryTimer = setTimeout(async () => {
        const cfg = getLLMConfig();
        if (!cfg) return;
        const realNodes = session.nodes.filter((n) => n.toolName !== '__thinking__');
        if (realNodes.length === 0) return;
        const nodeLines = realNodes
          .map((n) => `${n.toolName}: ${n.label} [${n.status}]`)
          .join('\n');
        try {
          const summary = await callLLM(cfg, SUMMARY_SYSTEM, `Agent actions:\n${nodeLines}`);
          traceStore.setAiSummary(session.id, summary);
        } catch (err) {
          outputChannel.appendLine(`[TraceBack] LLM summary error: ${err}`);
        }
      }, 2500);
    })
  );

  // On-demand chat queries from the webview
  context.subscriptions.push(
    provider.onLlmQuery(async ({ question, nodeContext }) => {
      const cfg = getLLMConfig();
      if (!cfg) {
        provider.postLlmResponse(
          'LLM is not configured. Add traceback.llmProvider (groq or ollama) in VS Code settings.'
        );
        return;
      }
      try {
        const answer = await callLLM(
          cfg,
          CHAT_SYSTEM,
          `Agent timeline:\n${nodeContext}\n\nQuestion: ${question}`
        );
        provider.postLlmResponse(answer);
      } catch (err) {
        outputChannel.appendLine(`[TraceBack] LLM chat error: ${err}`);
        provider.postLlmResponse(`Error calling LLM: ${err}`);
      }
    })
  );

  // Time-based anomaly detection (silent stalls) — check every 5 seconds
  const anomalyTimer = setInterval(() => traceStore.checkStalls(), 5000);
  context.subscriptions.push({ dispose: () => clearInterval(anomalyTimer) });

  context.subscriptions.push(
    vscode.commands.registerCommand('traceback.openMap', () => {
      provider.openFullPanel(context.extensionUri);
    }),
    vscode.commands.registerCommand('traceback.clearSession', () => {
      traceStore.clearActive();
      vscode.window.showInformationMessage('TraceBack: Session cleared.');
    }),
    vscode.commands.registerCommand('traceback.startListening', async () => {
      startServer(outputChannel, port);
      if (autoInstall) await installHooks(port, outputChannel);
    }),
    vscode.commands.registerCommand('traceback.stopListening', async () => {
      stopServer();
      await removeHooks(outputChannel);
    })
  );

  outputChannel.appendLine('[TraceBack] Ready.');
}

export async function deactivate(): Promise<void> {
  stopServer();
  traceStore.dispose();
}
