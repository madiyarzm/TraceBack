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
