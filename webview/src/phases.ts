import type { TimelineNode } from './components/TimelineCard';

/**
 * Phase detection: classify tool calls into the macro-modes of an agent
 * session — exploring the codebase, building changes, verifying them.
 * Pure pattern-matching over the node list; no LLM, deterministic, testable.
 *
 * The session's phase sequence is its "shape": healthy work reads
 * explore → build → verify; repeated build↔explore oscillation or
 * build→verify→build cycling is the visual signature of a lost agent.
 */

export type Phase = 'explore' | 'build' | 'verify';

export interface PhaseSegment {
  phase:   Phase;
  count:   number;
  startTs: number;
  endTs:   number;
}

export const PHASE_COLOR: Record<Phase, string> = {
  explore: '#58a6ff',
  build:   '#3fb950',
  verify:  '#a371f7',
};

export const PHASE_LABEL: Record<Phase, string> = {
  explore: 'exploring',
  build:   'building',
  verify:  'verifying',
};

const EXPLORE_TOOLS = new Set([
  'Read', 'Grep', 'Glob', 'WebSearch', 'WebFetch', 'Agent', 'TodoRead', 'LS', 'NotebookRead',
]);
const BUILD_TOOLS = new Set([
  'Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite',
]);
const VERIFY_BASH = /\b(test|vitest|jest|pytest|build|compile|tsc|lint|eslint|check|typecheck|mypy|cargo (test|build|check)|go (test|build|vet))\b/i;

/** null = excluded from the ribbon (thinking rows, plan updates, unknowns). */
export function classifyNode(node: TimelineNode): Phase | null {
  if (node.toolName === '__thinking__' || node.isPlanUpdate) return null;
  if (EXPLORE_TOOLS.has(node.toolName)) return 'explore';
  if (BUILD_TOOLS.has(node.toolName))   return 'build';
  if (node.toolName === 'Bash') {
    const cmd = `${(node.toolInput?.command as string) ?? ''} ${node.label}`;
    return VERIFY_BASH.test(cmd) ? 'verify' : 'build';
  }
  return null;
}

export function computePhases(nodes: TimelineNode[]): PhaseSegment[] {
  const segments: PhaseSegment[] = [];

  for (const node of nodes) {
    const phase = classifyNode(node);
    if (!phase) continue;
    const weight = node.isBatch ? node.count : node.count;
    const last = segments[segments.length - 1];
    if (last && last.phase === phase) {
      last.count += weight;
      last.endTs = node.timestamp;
    } else {
      segments.push({ phase, count: weight, startTs: node.timestamp, endTs: node.timestamp });
    }
  }

  return segments;
}
