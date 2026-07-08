/**
 * Deterministic tool-payload inspector. Pure TypeScript, no LLM, no deps:
 * classifies a tool call and extracts the few metrics worth showing, so the
 * UI can mask raw JSON/log dumps behind curated panels.
 */

export type PayloadKind = 'bash' | 'file-read' | 'file-write' | 'web' | 'generic';

export interface ParsedPayload {
  kind:      PayloadKind;
  /** bash */
  command?:  string;
  /** file ops */
  filePath?: string;
  lines?:    number;
  bytes?:    number;
  /** web */
  query?:    string;
  domain?:   string;
  url?:      string;
}

const BASH_TOOLS  = new Set(['Bash', 'ExecuteCommand', 'Shell']);
const READ_TOOLS  = new Set(['Read', 'FileRead', 'NotebookRead']);
const WRITE_TOOLS = new Set(['Write', 'Edit', 'MultiEdit', 'FileWrite', 'NotebookEdit']);

export function parseToolPayload(
  toolName: string,
  input?: Record<string, unknown>,
  output?: string
): ParsedPayload {
  if (BASH_TOOLS.has(toolName)) {
    return { kind: 'bash', command: str(input?.command) ?? str(input?.cmd) };
  }

  if (READ_TOOLS.has(toolName)) {
    return {
      kind:     'file-read',
      filePath: str(input?.file_path) ?? str(input?.path),
      ...sizeOf(output),
    };
  }

  if (WRITE_TOOLS.has(toolName)) {
    const written =
      str(input?.content) ?? str(input?.new_string) ??
      (Array.isArray(input?.edits)
        ? (input!.edits as { new_string?: string }[]).map((e) => e.new_string ?? '').join('\n')
        : undefined);
    return {
      kind:     'file-write',
      filePath: str(input?.file_path) ?? str(input?.path),
      ...sizeOf(written),
    };
  }

  if (toolName === 'WebSearch') {
    return { kind: 'web', query: str(input?.query) };
  }
  if (toolName === 'WebFetch') {
    const url = str(input?.url);
    return { kind: 'web', url, domain: domainOf(url) };
  }

  return { kind: 'generic' };
}

function str(v: unknown): string | undefined {
  return typeof v === 'string' && v.length > 0 ? v : undefined;
}

function sizeOf(text?: string): { lines?: number; bytes?: number } {
  if (!text) return {};
  return { lines: text.split('\n').length, bytes: text.length };
}

function domainOf(url?: string): string | undefined {
  if (!url) return undefined;
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return undefined;
  }
}

/** Does this path look like a directory rather than a file? Heuristic: no
 *  extension on the last segment (dotfiles like .env still count as files). */
function looksLikeDirectory(p: string): boolean {
  const last = p.split('/').filter(Boolean).pop() ?? '';
  return last !== '' && !last.includes('.');
}

/**
 * Deterministic one-line outcome for a tool's output — answers "what happened"
 * without dumping the payload. No LLM: shape-matching only. `node` context
 * (tool + input) lets known failure shapes explain themselves.
 */
export function summarizeOutput(
  detail: string | undefined,
  isError: boolean,
  node?: { toolName?: string; toolInput?: Record<string, unknown> },
): string | null {
  // Known stumble: Read on a directory. Explain it instead of "failed".
  if (isError && node?.toolName === 'Read') {
    const p = (node.toolInput?.file_path ?? node.toolInput?.path) as string | undefined;
    const dirError = detail && /EISDIR|is a directory|illegal operation on a directory/i.test(detail);
    if (dirError || (p && looksLikeDirectory(p) && !detail?.trim())) {
      return 'path is a directory — Read only works on files (the agent usually retries with ls or glob)';
    }
  }

  if (!detail || detail.trim().length === 0) return isError ? 'failed, no output' : null;

  if (isError) {
    // Surface the first meaningful error line(s), not the wall.
    const lines = detail.split('\n').map((l) => l.trim()).filter(Boolean);
    return truncate(lines.slice(0, 2).join(' · '), 110);
  }

  // Structured (JSON) outputs: classify by shape
  if (detail.startsWith('{') || detail.startsWith('[')) {
    try {
      const obj = JSON.parse(detail);
      if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
        if ('stdout' in obj || 'stderr' in obj) {
          const out = String(obj.stdout ?? '').trim();
          const err = String(obj.stderr ?? '').trim();
          if (!out && !err) return 'completed, no output';
          const src   = out || err;
          const first = src.split('\n')[0];
          const n     = src.split('\n').length;
          return truncate(n > 1 ? `${first} (+${n - 1} more lines)` : first, 110);
        }
        const keys = Object.keys(obj);
        return `object · ${keys.slice(0, 4).join(', ')}${keys.length > 4 ? ', …' : ''}`;
      }
      if (Array.isArray(obj)) return `list of ${obj.length} items`;
    } catch { /* not valid JSON — fall through to plain text */ }
  }

  // Plain text: first line + line count
  const lines = detail.split('\n');
  const first = lines.find((l) => l.trim().length > 0) ?? '';
  return truncate(lines.length > 1 ? `${first.trim()} (${lines.length} lines)` : first.trim(), 110);
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + '…' : s;
}

/** Clipboard write with a fallback for restricted webview contexts. */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

export function formatBytes(n?: number): string | null {
  if (n === undefined) return null;
  if (n < 1024)        return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
