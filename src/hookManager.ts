import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as vscode from 'vscode';

const HOOK_MARKER = '# traceback-managed';

interface ClaudeSettings {
  hooks?: {
    PreToolUse?: HookEntry[];
    PostToolUse?: HookEntry[];
    // Fires when a tool call FAILS — PostToolUse only fires on success.
    PostToolUseFailure?: HookEntry[];
    Stop?: HookEntry[];
    Notification?: HookEntry[];
  };
  [key: string]: unknown;
}

interface HookEntry {
  matcher: string;
  hooks: { type: string; command: string; timeout?: number }[];
}

// Generous per-hook timeout (seconds) so the breakpoint feature can hold a
// PreToolUse connection open while the user inspects a paused agent. Without
// this, Claude Code kills the curl at its default timeout and proceeds.
const HOOK_TIMEOUT_S = 600;

function getSettingsPath(): string {
  return path.join(os.homedir(), '.claude', 'settings.json');
}

function buildHookEntry(port: number): HookEntry {
  return {
    matcher: '',
    hooks: [
      {
        type: 'command',
        // Pipes the hook JSON payload (stdin) directly to our local server.
        // The HOOK_MARKER comment helps us identify and remove our entries later.
        command: `curl -s -X POST http://127.0.0.1:${port}/event -H 'Content-Type: application/json' -d @- ${HOOK_MARKER}`,
        timeout: HOOK_TIMEOUT_S,
      },
    ],
  };
}

function isManagedEntry(entry: HookEntry): boolean {
  return entry.hooks.some((h) => h.command.includes(HOOK_MARKER));
}

export async function installHooks(port: number, outputChannel: vscode.OutputChannel): Promise<void> {
  const settingsPath = getSettingsPath();

  let settings: ClaudeSettings = {};

  if (fs.existsSync(settingsPath)) {
    try {
      const raw = fs.readFileSync(settingsPath, 'utf-8');
      settings = JSON.parse(raw);
    } catch {
      vscode.window.showWarningMessage(
        'TraceBack: Could not parse ~/.claude/settings.json. Hooks not installed.'
      );
      return;
    }
  } else {
    // Create the directory if it doesn't exist
    fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
  }

  if (!settings.hooks) settings.hooks = {};

  const hookEntry = buildHookEntry(port);
  const hookTypes: (keyof NonNullable<ClaudeSettings['hooks']>)[] = [
    'PreToolUse',
    'PostToolUse',
    'PostToolUseFailure',
    'Stop',
  ];

  let changed = false;

  for (const hookType of hookTypes) {
    if (!settings.hooks[hookType]) {
      settings.hooks[hookType] = [];
    }
    const existing = settings.hooks[hookType]!;
    const alreadyInstalled = existing.some(isManagedEntry);

    if (!alreadyInstalled) {
      existing.push(hookEntry);
      changed = true;
    } else {
      // Migrate older managed entries that predate the breakpoint timeout.
      for (const entry of existing.filter(isManagedEntry)) {
        for (const h of entry.hooks) {
          if (h.command.includes(HOOK_MARKER) && h.timeout !== HOOK_TIMEOUT_S) {
            h.timeout = HOOK_TIMEOUT_S;
            changed = true;
          }
        }
      }
    }
  }

  if (changed) {
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf-8');
    outputChannel.appendLine('[TraceBack] Hooks installed into ~/.claude/settings.json');
    vscode.window.showInformationMessage(
      'TraceBack: Claude Code hooks installed. Start a new claude session to begin tracing.'
    );
  } else {
    outputChannel.appendLine('[TraceBack] Hooks already present, skipping install.');
  }
}

export async function removeHooks(outputChannel: vscode.OutputChannel): Promise<void> {
  const settingsPath = getSettingsPath();
  if (!fs.existsSync(settingsPath)) return;

  let settings: ClaudeSettings;
  try {
    const raw = fs.readFileSync(settingsPath, 'utf-8');
    settings = JSON.parse(raw);
  } catch {
    return;
  }

  if (!settings.hooks) return;

  const hookTypes: (keyof NonNullable<ClaudeSettings['hooks']>)[] = [
    'PreToolUse',
    'PostToolUse',
    'PostToolUseFailure',
    'Stop',
  ];

  let changed = false;

  for (const hookType of hookTypes) {
    const entries = settings.hooks[hookType];
    if (!entries) continue;
    const filtered = entries.filter((e) => !isManagedEntry(e));
    if (filtered.length !== entries.length) {
      settings.hooks[hookType] = filtered;
      changed = true;
    }
  }

  if (changed) {
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf-8');
    outputChannel.appendLine('[TraceBack] Hooks removed from ~/.claude/settings.json');
  }
}
