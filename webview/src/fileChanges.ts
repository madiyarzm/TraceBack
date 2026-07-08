import type { TimelineNode } from './components/TimelineCard';

/**
 * Pure summary of "what did the agent do to my codebase" — derived entirely
 * from the node list, no extension-host state. Write counts as created
 * (or overwritten — indistinguishable from hook data), Edit as modified.
 */

export interface FileChange {
  path:         string;
  kind:         'created' | 'modified';
  /** Number of separate write/edit calls that touched this file. */
  edits:        number;
  linesAdded:   number;
  linesRemoved: number;
}

const WRITE_TOOLS = new Set(['Write', 'FileWrite']);
const EDIT_TOOLS  = new Set(['Edit', 'MultiEdit', 'NotebookEdit']);

function countLines(s: string): number {
  if (!s) return 0;
  return s.split('\n').length;
}

interface DiffCounts { added: number; removed: number }

function diffCounts(input: Record<string, unknown>, toolName: string): DiffCounts {
  if (toolName === 'MultiEdit' && Array.isArray(input.edits)) {
    let added = 0, removed = 0;
    for (const e of input.edits as { old_string?: string; new_string?: string }[]) {
      added   += countLines(e.new_string ?? '');
      removed += countLines(e.old_string ?? '');
    }
    return { added, removed };
  }
  if (typeof input.content === 'string') {
    return { added: countLines(input.content), removed: 0 };
  }
  if (typeof input.old_string === 'string' || typeof input.new_string === 'string') {
    return {
      added:   countLines((input.new_string as string) ?? ''),
      removed: countLines((input.old_string as string) ?? ''),
    };
  }
  return { added: 0, removed: 0 };
}

export function computeFileChanges(nodes: TimelineNode[]): FileChange[] {
  const byPath = new Map<string, FileChange>();

  function record(toolName: string, input: Record<string, unknown> | undefined, calls: number): void {
    const isWrite = WRITE_TOOLS.has(toolName);
    const isEdit  = EDIT_TOOLS.has(toolName);
    if ((!isWrite && !isEdit) || !input) return;
    const filePath = (input.file_path ?? input.path ?? input.notebook_path) as string | undefined;
    if (!filePath) return;

    const { added, removed } = diffCounts(input, toolName);
    const existing = byPath.get(filePath);
    if (existing) {
      existing.edits        += calls;
      existing.linesAdded   += added;
      existing.linesRemoved += removed;
      // A Write after Edits (or vice versa) — creation wins as the stronger claim.
      if (isWrite) existing.kind = 'created';
    } else {
      byPath.set(filePath, {
        path:         filePath,
        kind:         isWrite ? 'created' : 'modified',
        edits:        calls,
        linesAdded:   added,
        linesRemoved: removed,
      });
    }
  }

  for (const node of nodes) {
    if (node.isBatch && node.batchItems) {
      for (const item of node.batchItems) record(node.toolName, item.toolInput, 1);
    } else {
      record(node.toolName, node.toolInput, node.count);
    }
  }

  // Stable, presentation-ready order: created first, then by path.
  return Array.from(byPath.values()).sort((a, b) =>
    a.kind !== b.kind ? (a.kind === 'created' ? -1 : 1) : a.path.localeCompare(b.path)
  );
}

/** "2 created · 5 modified" — empty string when nothing changed. */
export function summarizeChanges(changes: FileChange[]): string {
  const created  = changes.filter((c) => c.kind === 'created').length;
  const modified = changes.filter((c) => c.kind === 'modified').length;
  const parts: string[] = [];
  if (created)  parts.push(`${created} created`);
  if (modified) parts.push(`${modified} modified`);
  return parts.join(' · ');
}

// ── Touched-map: everything the agent looked at, not just what it changed ────

export interface TouchedFile {
  path: string;
  kind: 'read' | 'modified' | 'created';
}

const READ_TOOLS = new Set(['Read', 'NotebookRead']);

/**
 * The session's spatial footprint: every file read plus every file changed.
 * A file that was read AND edited reports its stronger claim (created >
 * modified > read). "The agent read 14 files to make this 2-line change" is
 * a coupling insight no chronological view surfaces.
 */
export function computeTouched(nodes: TimelineNode[]): TouchedFile[] {
  const rank = { read: 0, modified: 1, created: 2 } as const;
  const byPath = new Map<string, TouchedFile['kind']>();

  function record(toolName: string, input?: Record<string, unknown>): void {
    if (!input) return;
    const p = (input.file_path ?? input.path ?? input.notebook_path) as string | undefined;
    if (!p) return;
    const kind: TouchedFile['kind'] | null =
      WRITE_TOOLS.has(toolName) ? 'created'
      : EDIT_TOOLS.has(toolName) ? 'modified'
      : READ_TOOLS.has(toolName) ? 'read'
      : null;
    if (!kind) return;
    const prev = byPath.get(p);
    if (!prev || rank[kind] > rank[prev]) byPath.set(p, kind);
  }

  for (const node of nodes) {
    if (node.isBatch && node.batchItems) {
      for (const item of node.batchItems) record(node.toolName, item.toolInput);
    } else {
      record(node.toolName, node.toolInput);
    }
  }

  return Array.from(byPath.entries())
    .map(([path, kind]) => ({ path, kind }))
    .sort((a, b) => a.path.localeCompare(b.path));
}

/** "14 read · 3 modified · 1 created" */
export function summarizeTouched(touched: TouchedFile[]): string {
  const counts = { read: 0, modified: 0, created: 0 };
  for (const t of touched) counts[t.kind]++;
  const parts: string[] = [];
  if (counts.read)     parts.push(`${counts.read} read`);
  if (counts.modified) parts.push(`${counts.modified} modified`);
  if (counts.created)  parts.push(`${counts.created} created`);
  return parts.join(' · ');
}

// ── Verification: was each change exercised after its last edit? ─────────────

/**
 * verified   — a check command ran after the file's last edit and succeeded
 * failed     — check commands ran after the last edit but the latest errored
 * unverified — nothing that looks like a check ran after the last edit
 *
 * Deliberately evidence-based: the agent saying "done" counts for nothing;
 * only a command with an exit code does.
 */
export type VerifyStatus = 'verified' | 'failed' | 'unverified';

const CHECK_COMMAND = new RegExp(
  '\\b(test|spec|vitest|jest|pytest|unittest|tsc|typecheck|build|compile|' +
  'lint|eslint|ruff|mypy|check|vet|fmt|verify)\\b', 'i'
);

function isCheckCommand(command: string, basename: string): boolean {
  return CHECK_COMMAND.test(command) || (basename.length > 3 && command.includes(basename));
}

/**
 * Node-list order is chronological, so "after the last edit" is an index
 * comparison; batch items share their batch node's position.
 */
export function verifyChanges(nodes: TimelineNode[]): Map<string, VerifyStatus> {
  const lastEditIdx = new Map<string, number>();
  const bashRuns: { idx: number; command: string; error: boolean; pending: boolean }[] = [];

  nodes.forEach((node, idx) => {
    const items = node.isBatch && node.batchItems ? node.batchItems : [node];
    for (const item of items) {
      const input = item.toolInput;
      if (!input) continue;
      if (WRITE_TOOLS.has(node.toolName) || EDIT_TOOLS.has(node.toolName)) {
        const p = (input.file_path ?? input.path ?? input.notebook_path) as string | undefined;
        if (p) lastEditIdx.set(p, idx);
      } else if (node.toolName === 'Bash' && typeof input.command === 'string') {
        bashRuns.push({
          idx,
          command: input.command,
          error:   item.status === 'error',
          pending: item.status === 'pending',
        });
      }
    }
  });

  const out = new Map<string, VerifyStatus>();
  for (const [path, editIdx] of lastEditIdx) {
    const basename = path.split('/').filter(Boolean).pop() ?? '';
    const checks = bashRuns.filter(
      (r) => r.idx > editIdx && !r.pending && isCheckCommand(r.command, basename)
    );
    if (checks.length === 0) {
      out.set(path, 'unverified');
    } else {
      out.set(path, checks[checks.length - 1].error ? 'failed' : 'verified');
    }
  }
  return out;
}

/** "2 of 5 changed files never exercised after their last edit" — '' when clean. */
export function summarizeVerification(statuses: Map<string, VerifyStatus>): string {
  const total = statuses.size;
  if (total === 0) return '';
  let unverified = 0, failed = 0;
  for (const s of statuses.values()) {
    if (s === 'unverified') unverified++;
    if (s === 'failed') failed++;
  }
  const parts: string[] = [];
  if (unverified > 0)
    parts.push(`${unverified} of ${total} changed file${total === 1 ? '' : 's'} never exercised after the last edit`);
  if (failed > 0)
    parts.push(`${failed} failing the latest check`);
  return parts.join(' · ');
}
