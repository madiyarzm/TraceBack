import type { TimelineNode } from './components/TimelineCard';

/**
 * Net-change review annotations: the "why" behind each touched file, derived
 * from the node list. The file contents themselves (baseline vs. now) come
 * from the extension host on demand; this module supplies the provenance that
 * turns a raw diff into a reviewable change.
 */

export interface FileReviewAnnotations {
  /** Intent sentences from the edits that touched this file, deduped, in order. */
  intents:     string[];
  editCount:   number;
  readCount:   number;
  /** Label of the nearest failed Bash call BEFORE the first edit — the
   *  evidence that likely motivated the change. */
  triggeredBy?: string;
}

const EDIT_TOOLS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite']);

function pathOf(input?: Record<string, unknown>): string | undefined {
  return (input?.file_path ?? input?.path ?? input?.notebook_path) as string | undefined;
}

export function annotateReviewFile(nodes: TimelineNode[], path: string): FileReviewAnnotations {
  const intents: string[] = [];
  let editCount = 0;
  let readCount = 0;
  let firstEditSeen = false;
  let lastFailedBash: string | undefined;
  let triggeredBy: string | undefined;

  for (const node of nodes) {
    const items = node.isBatch && node.batchItems ? node.batchItems : [node];
    for (const item of items) {
      const p = pathOf(item.toolInput);
      if (node.toolName === 'Bash') {
        if (item.status === 'error') lastFailedBash = item.label ?? node.label;
        continue;
      }
      if (p !== path) continue;
      if (node.toolName === 'Read') { readCount++; continue; }
      if (!EDIT_TOOLS.has(node.toolName)) continue;

      editCount++;
      if (!firstEditSeen) {
        firstEditSeen = true;
        triggeredBy = lastFailedBash; // failure immediately preceding the first edit
      }
      const intent = item.intent ?? node.intent;
      if (intent && !intents.includes(intent)) intents.push(intent);
    }
  }

  return { intents, editCount, readCount, triggeredBy };
}

export type NetKind = 'created' | 'modified' | 'deleted' | 'unchanged';

/** What actually happened to the file between baseline and now. */
export function netKindOf(baseline: string | null, current: string | null): NetKind {
  if (baseline === null && current !== null) return 'created';
  if (baseline !== null && current === null) return 'deleted';
  if (baseline === current)                  return 'unchanged';
  return 'modified';
}
