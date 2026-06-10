import * as vscode from 'vscode';
import { startServer, stopServer } from './server';
import { installHooks, removeHooks } from './hookManager';
import { traceStore } from './traceStore';
import { TracebackWebviewProvider } from './webviewProvider';
import { callLLM, LLMConfig } from './llmClient';
import { loadEnvFromAllKnownLocations } from './envLoader';

let outputChannel: vscode.OutputChannel;

/**
 * Resolves the LLM backend in this priority order, per knob:
 *   1. VS Code setting (explicit, per-machine override)
 *   2. process.env (.env file or shell env)
 *   3. hard-coded default
 *
 * Provider auto-detects from the env: if the user only set GROQ_API_KEY (or
 * OLLAMA_MODEL) without flipping traceback.llmProvider, we still light up
 * the Narrative Engine instead of silently doing nothing.
 */
function getLLMConfig(): LLMConfig | null {
  const cfg = vscode.workspace.getConfiguration('traceback');

  let provider = cfg.get<string>('llmProvider') ?? 'disabled';
  if (provider === 'disabled') {
    if (process.env.GROQ_API_KEY)  provider = 'groq';
    else if (process.env.OLLAMA_MODEL) provider = 'ollama';
    else return null;
  }

  return {
    provider: provider as 'groq' | 'ollama',
    apiKey:
      cfg.get<string>('groqApiKey') ||
      process.env.GROQ_API_KEY ||
      '',
    model: provider === 'groq'
      ? (cfg.get<string>('groqModel') ||
         process.env.GROQ_MODEL ||
         'llama-3.1-8b-instant')
      : (cfg.get<string>('ollamaModel') ||
         process.env.OLLAMA_MODEL ||
         'llama3.2'),
    ollamaPort:
      cfg.get<number>('ollamaPort') ||
      Number(process.env.OLLAMA_PORT) ||
      11434,
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

  const workspaceFolders =
    vscode.workspace.workspaceFolders?.map((f) => f.uri.fsPath) ?? [];
  const loadedKeys = loadEnvFromAllKnownLocations(
    context.extensionUri.fsPath,
    workspaceFolders,
  );
  if (loadedKeys.length) {
    outputChannel.appendLine(
      `[TraceBack] Loaded .env keys: ${loadedKeys.join(', ')}`,
    );
  }

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
