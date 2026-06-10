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
  'You are a pragmatic, ultra-concise senior engineering monitor watching an AI coding agent work. ' +
  'Describe what the agent just did and its current state in plain, punchy English. ' +
  'HARD LIMITS: maximum 2 short sentences. No greetings, no preamble, no hedging. ' +
  'NO markdown of any kind: no **, no #, no bullets, no code fences. Plain text only.';

const CHAT_SYSTEM =
  'You are a pragmatic senior engineer answering questions about an AI coding agent\'s action timeline. ' +
  'Be direct: maximum 2 short sentences OR 3 short bullet lines starting with "- ". ' +
  'Never use greetings or preamble ("Sure...", "Looking at this..."). ' +
  'Never use headings (#) or section labels like "Possible Context:". Bold (**term**) and `code` are allowed, nothing else. ' +
  'Name the exact tools/files involved. If the timeline does not contain the answer, say so in one sentence.';

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

  // LLM auto-summary: fires only after a 4s quiet period (no agent events),
  // and only when the timeline actually changed since the last summary —
  // token refreshes and stall re-checks must not regenerate the text.
  let summaryTimer: ReturnType<typeof setTimeout> | null = null;
  let lastSummarySig = '';

  context.subscriptions.push(
    traceStore.onDidUpdate((session) => {
      provider.postSessionUpdate(session);

      if (summaryTimer) clearTimeout(summaryTimer);
      summaryTimer = setTimeout(async () => {
        const cfg = getLLMConfig();
        if (!cfg) return;
        const realNodes = session.nodes.filter((n) => n.toolName !== '__thinking__');
        if (realNodes.length === 0) return;

        const sig = `${session.id}|${realNodes.length}|${realNodes[realNodes.length - 1]?.status}|${session.stopped}`;
        if (sig === lastSummarySig) return;

        const nodeLines = realNodes
          .map((n) => `${n.toolName}: ${n.label} [${n.status}]`)
          .join('\n');
        try {
          const summary = await callLLM(cfg, SUMMARY_SYSTEM, `Agent actions:\n${nodeLines}`, 90);
          traceStore.setAiSummary(session.id, summary);
          lastSummarySig = sig;
        } catch (err) {
          outputChannel.appendLine(`[TraceBack] LLM summary error: ${err}`);
        }
      }, 4000);
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

  // OS-level alert on anomaly onset — works even when the sidebar is hidden.
  // Fires once per detection (traceStore dedupes re-evaluations).
  context.subscriptions.push(
    traceStore.onAnomaly(({ session, record }) => {
      vscode.window
        .showWarningMessage(`TraceBack — ${session.label}: ${record.reason}`, 'Show')
        .then((choice) => {
          if (choice === 'Show') {
            vscode.commands.executeCommand('traceback.mapView.focus');
          }
        });
    })
  );

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
