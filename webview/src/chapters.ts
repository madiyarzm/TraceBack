import type { TimelineNode } from './components/TimelineCard';
import type { AnomalyRecordUI } from './useSessionFeed';

/**
 * Prompt-as-chapter model: a session is a book, each UserPromptSubmit opens a
 * chapter, and everything the agent did until the next prompt belongs to it.
 * Pure derivation over the node list — no extension-host changes needed.
 */

export interface ChapterPlanItem {
  content:     string;
  status:      'pending' | 'in_progress' | 'completed';
  activeForm?: string;
}

export interface ChapterTaskGroup {
  /** Objective the actions ran under; '' for actions before any plan existed. */
  objective: string;
  status:    'pending' | 'in_progress' | 'completed' | 'none';
  nodes:     TimelineNode[];
  /** Distinct file names touched — the one-line summary for done tasks. */
  files:     string[];
  errorCount: number;
}

export type ChapterBadge = 'done' | 'active' | 'errors' | 'anomalies' | 'queued';

export interface PromptChapter {
  /** Node id of the __prompt__ node ('' for the pre-prompt chapter). */
  id:          string;
  index:       number;   // 1-based → P1, P2, …
  text:        string;   // full prompt text
  timestamp:   number;
  /** Timestamp of the next prompt (or Infinity for the last chapter). */
  endTimestamp: number;
  /** Every node in the slice (incl. plan updates), prompt node excluded. */
  nodes:       TimelineNode[];
  actionCount: number;
  errorCount:  number;
  durationMs:  number;
  /** Plan snapshot at the END of this chapter's slice. */
  plan:        ChapterPlanItem[];
  taskGroups:  ChapterTaskGroup[];
}

const isReal = (n: TimelineNode) => !n.toolName.startsWith('__');

function fileNameOf(input?: Record<string, unknown>): string | null {
  const p = (input?.file_path ?? input?.path ?? input?.notebook_path) as string | undefined;
  if (!p) return null;
  return p.split('/').filter(Boolean).pop() ?? null;
}

/** Plan mutation applied per node — mirrors traceStore's applyPlanTool. */
function applyPlanNode(
  plan: ChapterPlanItem[],
  taskCounterRef: { n: number },
  taskIds: Map<number, string>,
  node: TimelineNode,
): ChapterPlanItem[] {
  const input = node.toolInput ?? {};
  if (node.toolName === 'TodoWrite' && Array.isArray(input.todos)) {
    const items: ChapterPlanItem[] = [];
    for (const raw of input.todos as Record<string, unknown>[]) {
      if (!raw || typeof raw.content !== 'string') continue;
      items.push({
        content: raw.content,
        status: raw.status === 'in_progress' || raw.status === 'completed' ? raw.status : 'pending',
        activeForm: typeof raw.activeForm === 'string' ? raw.activeForm : undefined,
      });
    }
    return items.length ? items : plan;
  }
  if (node.toolName === 'TaskCreate' && typeof input.subject === 'string') {
    taskCounterRef.n++;
    taskIds.set(plan.length, String(taskCounterRef.n));
    return [...plan, {
      content: input.subject,
      status: 'pending',
      activeForm: typeof input.activeForm === 'string' ? input.activeForm : undefined,
    }];
  }
  if (node.toolName === 'TaskUpdate') {
    const rawId = input.taskId;
    const wanted =
      typeof rawId === 'string' ? rawId.replace(/^#/, '')
      : typeof rawId === 'number' ? String(rawId)
      : null;
    if (!wanted) return plan;
    return plan
      .filter((_, i) => !(taskIds.get(i) === wanted && input.status === 'deleted'))
      .map((it, i) => taskIds.get(i) !== wanted ? it : {
        ...it,
        status:
          input.status === 'in_progress' || input.status === 'completed'
            ? input.status
            : it.status,
        activeForm: typeof input.activeForm === 'string' ? input.activeForm : it.activeForm,
        content:    typeof input.subject === 'string' ? input.subject : it.content,
      });
  }
  return plan;
}

export function computeChapters(nodes: TimelineNode[]): PromptChapter[] {
  const chapters: PromptChapter[] = [];
  let current: PromptChapter | null = null;
  let plan: ChapterPlanItem[] = [];
  const taskCounter = { n: 0 };
  const taskIds = new Map<number, string>();

  function open(id: string, text: string, timestamp: number): PromptChapter {
    const chapter: PromptChapter = {
      id, index: chapters.length + 1, text, timestamp,
      endTimestamp: Infinity,
      nodes: [], actionCount: 0, errorCount: 0, durationMs: 0,
      plan, // inherit the running snapshot — a queued prompt still has a plan
      taskGroups: [],
    };
    chapters.push(chapter);
    return chapter;
  }

  for (const node of nodes) {
    if (node.toolName === '__thinking__') continue;
    if (node.toolName === '__prompt__') {
      const text = (node.detail ?? node.label).trim();
      // Slash commands (/login, /clear, /compact, /login …) are client
      // commands, not task prompts. They must not open a chapter — otherwise
      // the work that follows gets misfiled under "/login" instead of the
      // real prompt.
      if (text.startsWith('/')) continue;
      // Re-submits: an identical or corrected prompt arriving before the
      // current chapter did any work supersedes it in place, so a
      // double-fired prompt doesn't leave an empty duplicate chapter.
      if (current && current.actionCount === 0) {
        current.id = node.id;
        current.text = text;
        current.timestamp = node.timestamp;
        continue;
      }
      if (current) current.endTimestamp = node.timestamp;
      current = open(node.id, text, node.timestamp);
      continue;
    }
    if (!current) current = open('', '(before first prompt)', node.timestamp);
    if (node.isPlanUpdate) plan = applyPlanNode(plan, taskCounter, taskIds, node);
    const c = current;
    c.nodes.push(node);
    c.plan = plan;
    if (isReal(node) && !node.isPlanUpdate) {
      c.actionCount += node.count;
      c.errorCount  += node.status === 'error' ? 1 : 0;
      c.durationMs  += node.durationMs ?? 0;
    }
  }

  for (const c of chapters) c.taskGroups = buildTaskGroups(c);
  return chapters;
}

/**
 * Group a chapter's actions by the objective (todo item) they ran under,
 * preserving first-seen order, and stamp each group with the item's status
 * from the chapter-end plan snapshot.
 */
function buildTaskGroups(chapter: PromptChapter): ChapterTaskGroup[] {
  const groups = new Map<string, ChapterTaskGroup>();

  for (const node of chapter.nodes) {
    if (!isReal(node) || node.isPlanUpdate) continue;
    const key = node.objective ?? '';
    let g = groups.get(key);
    if (!g) {
      g = { objective: key, status: 'none', nodes: [], files: [], errorCount: 0 };
      groups.set(key, g);
    }
    g.nodes.push(node);
    if (node.status === 'error') g.errorCount++;
    const items = node.isBatch && node.batchItems ? node.batchItems : [node];
    for (const item of items) {
      const f = fileNameOf(item.toolInput);
      if (f && !g.files.includes(f)) g.files.push(f);
    }
  }

  for (const g of groups.values()) {
    const item = chapter.plan.find(
      (p) => p.content === g.objective || p.activeForm === g.objective
    );
    if (item) g.status = item.status;
  }

  return Array.from(groups.values());
}

/** Plan items in the chapter snapshot with no recorded actions (locked/pending). */
export function pendingPlanItems(chapter: PromptChapter): ChapterPlanItem[] {
  const covered = new Set(chapter.taskGroups.map((g) => g.objective));
  return chapter.plan.filter(
    (p) => !covered.has(p.content) && !covered.has(p.activeForm ?? '')
  );
}

/** Anomaly records that fell inside this chapter's time slice. */
export function anomaliesFor(
  chapter: PromptChapter,
  records: AnomalyRecordUI[] | undefined,
): AnomalyRecordUI[] {
  if (!records) return [];
  return records.filter(
    (r) => r.detectedAt >= chapter.timestamp && r.detectedAt < chapter.endTimestamp
  );
}

export function chapterBadge(
  chapter: PromptChapter,
  opts: { isLast: boolean; isLive: boolean; anomalyCount: number },
): { badge: ChapterBadge; label: string } {
  if (opts.anomalyCount > 0)
    return { badge: 'anomalies', label: `${opts.anomalyCount} anomal${opts.anomalyCount === 1 ? 'y' : 'ies'}` };
  if (chapter.errorCount > 0)
    return { badge: 'errors', label: `${chapter.errorCount} error${chapter.errorCount === 1 ? '' : 's'}` };
  if (opts.isLast && opts.isLive)
    return chapter.actionCount === 0
      ? { badge: 'queued', label: 'queued' }
      : { badge: 'active', label: 'active' };
  if (chapter.actionCount === 0 && opts.isLive)
    return { badge: 'queued', label: 'queued' };
  return { badge: 'done', label: 'done' };
}

export function timeAgo(ts: number, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - ts) / 1000));
  if (s < 45)     return 'now';
  if (s < 3600)   return `${Math.round(s / 60)}m ago`;
  if (s < 86_400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86_400)}d ago`;
}
