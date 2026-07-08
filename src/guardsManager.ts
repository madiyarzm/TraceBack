import * as vscode from 'vscode';
import type { TraceEvent } from './traceStore';

/**
 * Guards: policy rules checked against every PreToolUse call BEFORE it
 * executes. A match auto-denies the call via the hook response, with the
 * guard's name fed back to the agent as context (so Claude knows WHY and can
 * route around the rule instead of blindly retrying).
 *
 * Two kinds:
 *  - Built-in guards: named, human-toggleable rules with hand-written logic.
 *  - Custom guards: user regexes matched against the full tool call string.
 *
 * State persists in vscode.workspace.getConfiguration('traceback'):
 *  - traceback.guards        { [builtinKey]: boolean }
 *  - traceback.customGuards  string[]
 */

export type BuiltinGuardKey =
  | 'never_delete'
  | 'stay_in_project'
  | 'protect_secrets'
  | 'no_push_main';

export interface GuardHit {
  /** Guard name — sent to the agent as part of the deny reason. */
  name:   string;
  reason: string;
}

export interface BuiltinGuardInfo {
  key:         BuiltinGuardKey;
  label:       string;
  description: string;
  enabled:     boolean;
}

export interface GuardsState {
  builtins: BuiltinGuardInfo[];
  custom:   string[];
}

const EDIT_WRITE_TOOLS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite']);
const READ_EDIT_TOOLS  = new Set(['Read', 'Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite']);
const SECRET_PATH      = /(^|\/)\.env(\.|$)|(^|\/)[^/]*\.secret(\.|$)|(^|\/)secrets\//i;

interface BuiltinGuardDef {
  key:         BuiltinGuardKey;
  label:       string;
  description: string;
  matches(event: TraceEvent): boolean;
}

function commandOf(event: TraceEvent): string {
  const c = event.toolInput?.command;
  return typeof c === 'string' ? c : '';
}

function filePathOf(event: TraceEvent): string | undefined {
  const input = event.toolInput ?? {};
  const p = input.file_path ?? input.path ?? input.notebook_path;
  return typeof p === 'string' ? p : undefined;
}

export const BUILTIN_GUARDS: BuiltinGuardDef[] = [
  {
    key:         'never_delete',
    label:       'Never delete files',
    description: 'Blocks any rm command or DeleteFile call.',
    matches: (e) =>
      e.toolName === 'DeleteFile' ||
      (e.toolName === 'Bash' && /\brm\b/.test(commandOf(e))),
  },
  {
    key:         'stay_in_project',
    label:       'Stay in project folder',
    description: 'Blocks edits and writes to paths outside the session working directory.',
    matches: (e) => {
      if (!EDIT_WRITE_TOOLS.has(e.toolName ?? '')) return false;
      const file = filePathOf(e);
      if (!file || !e.cwd || !file.startsWith('/')) return false;
      const cwd = e.cwd.endsWith('/') ? e.cwd : e.cwd + '/';
      return file !== e.cwd && !file.startsWith(cwd);
    },
  },
  {
    key:         'protect_secrets',
    label:       'Protect .env and secrets',
    description: 'Blocks reads and edits of .env, .secret, and secrets/ paths.',
    matches: (e) => {
      if (!READ_EDIT_TOOLS.has(e.toolName ?? '')) return false;
      const file = filePathOf(e);
      return !!file && SECRET_PATH.test(file);
    },
  },
  {
    key:         'no_push_main',
    label:       'No git push to main',
    description: 'Blocks bash commands that push to the main branch.',
    matches: (e) => e.toolName === 'Bash' && /git push.*main/.test(commandOf(e)),
  },
];

function config() {
  return vscode.workspace.getConfiguration('traceback');
}

function enabledBuiltins(): Record<string, boolean> {
  return config().get<Record<string, boolean>>('guards') ?? {};
}

function customPatterns(): string[] {
  const raw = config().get<string[]>('customGuards') ?? [];
  return raw.filter((p): p is string => typeof p === 'string' && p.length > 0);
}

export function getGuardsState(): GuardsState {
  const enabled = enabledBuiltins();
  return {
    builtins: BUILTIN_GUARDS.map((g) => ({
      key:         g.key,
      label:       g.label,
      description: g.description,
      enabled:     enabled[g.key] === true,
    })),
    custom: customPatterns(),
  };
}

/** Workspace-level persistence; falls back to global when no folder is open. */
async function update(key: string, value: unknown): Promise<void> {
  try {
    await config().update(key, value, vscode.ConfigurationTarget.Workspace);
  } catch {
    await config().update(key, value, vscode.ConfigurationTarget.Global);
  }
}

export async function setBuiltinGuard(key: BuiltinGuardKey, enabled: boolean): Promise<void> {
  await update('guards', { ...enabledBuiltins(), [key]: enabled });
}

export async function addCustomGuard(pattern: string): Promise<void> {
  const trimmed = pattern.trim();
  if (!trimmed) return;
  try {
    new RegExp(trimmed); // reject invalid regexes at entry, not at match time
  } catch {
    throw new Error(`Invalid regex: ${trimmed}`);
  }
  const current = customPatterns();
  if (current.includes(trimmed)) return;
  await update('customGuards', [...current, trimmed]);
}

export async function removeCustomGuard(pattern: string): Promise<void> {
  await update('customGuards', customPatterns().filter((p) => p !== pattern));
}

/**
 * Returns the first guard matching this PreToolUse call, or null. The deny
 * reason names the guard so it lands in Claude's context as actionable
 * feedback rather than an opaque rejection.
 */
export function checkGuards(event: TraceEvent): GuardHit | null {
  const enabled = enabledBuiltins();

  for (const guard of BUILTIN_GUARDS) {
    if (enabled[guard.key] !== true) continue;
    if (guard.matches(event)) {
      return {
        name:   guard.key,
        reason:
          `Blocked by TraceBack guard "${guard.key}" (${guard.description}) ` +
          `Do not retry this exact call; adjust your approach or ask the user.`,
      };
    }
  }

  const haystack = `${event.toolName ?? ''} ${JSON.stringify(event.toolInput ?? {})}`;
  for (const pattern of customPatterns()) {
    try {
      if (new RegExp(pattern, 'i').test(haystack)) {
        return {
          name:   pattern,
          reason:
            `Blocked by TraceBack custom guard /${pattern}/. ` +
            `Do not retry this exact call; adjust your approach or ask the user.`,
        };
      }
    } catch {
      // invalid regex saved by hand-editing settings — never crash the hook path
    }
  }

  return null;
}
