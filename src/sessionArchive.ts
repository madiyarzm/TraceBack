import * as fs from 'fs';
import * as path from 'path';
import type { TraceSession } from './traceStore';

/**
 * Disk persistence for finished (and in-flight) sessions. One JSON file per
 * session in <globalStorage>/sessions/, capped at MAX_ARCHIVED — history must
 * survive window reloads, but never grow unbounded.
 */

export const MAX_ARCHIVED = 50;

export interface ArchivedSessionMeta {
  id:           string;
  label:        string;
  startedAt:    number;
  endedAt:      number;
  nodeCount:    number;
  errorCount:   number;
  anomalyCount: number;
  tokens?:      number;
}

/** The serialized shape mirrors what postSessionUpdate sends to the webview —
 *  raw events are dropped to keep files small; nodes are enough to render. */
export interface ArchivedSession extends ArchivedSessionMeta {
  stopped:         boolean;
  nodes:           TraceSession['nodes'];
  anomalyHistory:  TraceSession['anomalyHistory'];
  aiSummary?:      string;
  cwd?:            string;
  plan?:           TraceSession['plan'];
  ledger?:         TraceSession['ledger'];
}

/** Mirror of the webview's isExpectedStumble (payloadParser.ts): Read on a
 *  directory is a probe the agent recovers from itself, not a counted error —
 *  the history-rail badge must agree with the counts inside the session. */
function isExpectedStumble(n: {
  toolName: string; detail?: string; toolInput?: Record<string, unknown>;
}): boolean {
  if (n.toolName !== 'Read') return false;
  if (n.detail && /EISDIR|is a directory|illegal operation on a directory/i.test(n.detail)) return true;
  const p = (n.toolInput?.file_path ?? n.toolInput?.path) as string | undefined;
  const last = p?.split('/').filter(Boolean).pop() ?? '';
  return !!p && last !== '' && !last.includes('.') && !n.detail?.trim();
}

function fileFor(dir: string, id: string): string {
  // Session ids come from Claude Code (uuid-ish) but sanitize defensively.
  return path.join(dir, `${id.replace(/[^\w.-]/g, '_')}.json`);
}

export function serializeSession(session: TraceSession): ArchivedSession {
  const realNodes = session.nodes.filter((n) => !n.toolName.startsWith('__'));
  return {
    id:           session.id,
    label:        session.label,
    startedAt:    session.startedAt,
    endedAt:      Date.now(),
    nodeCount:    realNodes.length,
    errorCount:   realNodes.filter((n) => n.status === 'error' && !isExpectedStumble(n)).length,
    anomalyCount: session.anomalyHistory.length,
    tokens:       session.contextTokens,
    stopped:      session.stopped,
    nodes:        session.nodes,
    anomalyHistory: session.anomalyHistory,
    aiSummary:    session.aiSummary,
    cwd:          session.cwd,
    plan:         session.plan,
    ledger:       session.ledger,
  };
}

export function saveSession(dir: string, session: TraceSession): void {
  const realNodes = session.nodes.filter((n) => !n.toolName.startsWith('__'));
  if (realNodes.length === 0) return; // never archive empty sessions
  fs.mkdirSync(dir, { recursive: true });
  const data = serializeSession(session);
  fs.writeFileSync(fileFor(dir, session.id), JSON.stringify(data), 'utf8');
  pruneOldest(dir, MAX_ARCHIVED);
}

export function listSessions(dir: string): ArchivedSessionMeta[] {
  let files: string[];
  try {
    files = fs.readdirSync(dir).filter((f) => f.endsWith('.json'));
  } catch {
    return [];
  }
  const metas: ArchivedSessionMeta[] = [];
  for (const f of files) {
    try {
      const raw = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')) as ArchivedSession;
      metas.push({
        id:           raw.id,
        label:        raw.label,
        startedAt:    raw.startedAt,
        endedAt:      raw.endedAt,
        nodeCount:    raw.nodeCount,
        errorCount:   raw.errorCount,
        anomalyCount: raw.anomalyCount,
        tokens:       raw.tokens,
      });
    } catch {
      // corrupt file — skip, don't crash the extension
    }
  }
  return metas.sort((a, b) => b.startedAt - a.startedAt);
}

export function loadSession(dir: string, id: string): ArchivedSession | null {
  try {
    return JSON.parse(fs.readFileSync(fileFor(dir, id), 'utf8')) as ArchivedSession;
  } catch {
    return null;
  }
}

export function pruneOldest(dir: string, keep: number): void {
  const metas = listSessions(dir); // newest first
  for (const meta of metas.slice(keep)) {
    try {
      fs.unlinkSync(fileFor(dir, meta.id));
    } catch {
      // already gone — fine
    }
  }
}
