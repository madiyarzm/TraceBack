import { describe, expect, it } from 'vitest';
import {
  anomaliesFor, chapterBadge, computeChapters, computePhaseBlocks, pendingPlanItems,
} from '../webview/src/chapters';
import type { TimelineNode } from '../webview/src/components/TimelineCard';
import type { AnomalyRecordUI } from '../webview/src/useSessionFeed';

const T0 = 1_700_000_000_000;
let nextId = 0;

function node(partial: Partial<TimelineNode> & { toolName: string }): TimelineNode {
  return {
    id: `n-${++nextId}`,
    status: 'success',
    label: partial.toolName,
    count: 1,
    timestamp: T0,
    ...partial,
  };
}

const prompt = (text: string, t: number) =>
  node({ toolName: '__prompt__', label: text.slice(0, 60), detail: text, timestamp: t });

describe('computeChapters', () => {
  it('slices nodes into chapters between prompts', () => {
    const nodes = [
      prompt('first prompt', T0),
      node({ toolName: 'Read', timestamp: T0 + 1000 }),
      node({ toolName: 'Edit', timestamp: T0 + 2000 }),
      prompt('second prompt', T0 + 10_000),
      node({ toolName: 'Bash', timestamp: T0 + 11_000, status: 'error' }),
    ];
    const chapters = computeChapters(nodes);
    expect(chapters).toHaveLength(2);
    expect(chapters[0].text).toBe('first prompt');
    expect(chapters[0].actionCount).toBe(2);
    expect(chapters[0].endTimestamp).toBe(T0 + 10_000);
    expect(chapters[1].actionCount).toBe(1);
    expect(chapters[1].errorCount).toBe(1);
  });

  it('coalesces a re-submitted prompt that did no work into one chapter', () => {
    // The exact shape seen in a real session: the prompt fired twice before
    // any tool ran, then all the work followed.
    const nodes = [
      prompt('refactor to async/await', T0),
      prompt('refactor to async/await', T0 + 500),
      node({ toolName: 'Read', timestamp: T0 + 1000 }),
      node({ toolName: 'Edit', timestamp: T0 + 2000 }),
    ];
    const chapters = computeChapters(nodes);
    expect(chapters).toHaveLength(1);
    expect(chapters[0].text).toBe('refactor to async/await');
    expect(chapters[0].actionCount).toBe(2);
  });

  it('ignores slash-command prompts so work stays under the real prompt', () => {
    const nodes = [
      prompt('refactor to async/await', T0),
      prompt('/login', T0 + 500),
      node({ toolName: 'Read', timestamp: T0 + 1000 }),
      node({ toolName: 'Bash', timestamp: T0 + 2000 }),
    ];
    const chapters = computeChapters(nodes);
    expect(chapters).toHaveLength(1);
    expect(chapters[0].text).toBe('refactor to async/await');
    expect(chapters[0].actionCount).toBe(2);
  });

  it('still opens a new chapter for a real prompt after work is done', () => {
    const nodes = [
      prompt('task one', T0),
      node({ toolName: 'Read', timestamp: T0 + 1000 }),
      prompt('task two', T0 + 5000),
      node({ toolName: 'Edit', timestamp: T0 + 6000 }),
    ];
    const chapters = computeChapters(nodes);
    expect(chapters).toHaveLength(2);
    expect(chapters[0].text).toBe('task one');
    expect(chapters[1].text).toBe('task two');
  });

  it('puts pre-prompt work into a synthetic opening chapter', () => {
    const nodes = [
      node({ toolName: 'Read', timestamp: T0 }),
      prompt('real prompt', T0 + 5000),
      node({ toolName: 'Edit', timestamp: T0 + 6000 }),
    ];
    const chapters = computeChapters(nodes);
    expect(chapters).toHaveLength(2);
    expect(chapters[0].id).toBe('');
    expect(chapters[0].actionCount).toBe(1);
  });

  it('skips thinking rows and excludes plan updates from action counts', () => {
    const nodes = [
      prompt('p', T0),
      node({ toolName: '__thinking__', timestamp: T0 + 1 }),
      node({ toolName: 'TodoWrite', isPlanUpdate: true, timestamp: T0 + 2, toolInput: {
        todos: [{ content: 'task A', status: 'in_progress' }],
      } }),
      node({ toolName: 'Read', timestamp: T0 + 3 }),
    ];
    const c = computeChapters(nodes)[0];
    expect(c.actionCount).toBe(1);
    expect(c.plan).toHaveLength(1);
    expect(c.plan[0].status).toBe('in_progress');
  });

  it('groups actions by objective and stamps plan status', () => {
    const nodes = [
      prompt('p', T0),
      node({ toolName: 'TodoWrite', isPlanUpdate: true, timestamp: T0 + 1, toolInput: {
        todos: [
          { content: 'task A', status: 'in_progress', activeForm: 'Doing task A' },
          { content: 'task B', status: 'pending' },
        ],
      } }),
      node({ toolName: 'Edit', objective: 'Doing task A', timestamp: T0 + 2,
             toolInput: { file_path: '/x/logger.ts' } }),
      node({ toolName: 'Bash', objective: 'Doing task A', timestamp: T0 + 3 }),
    ];
    const c = computeChapters(nodes)[0];
    expect(c.taskGroups).toHaveLength(1);
    expect(c.taskGroups[0].objective).toBe('Doing task A');
    expect(c.taskGroups[0].status).toBe('in_progress');
    expect(c.taskGroups[0].files).toEqual(['logger.ts']);
    expect(pendingPlanItems(c).map((p) => p.content)).toEqual(['task B']);
  });
});

describe('anomaliesFor', () => {
  it('assigns records to chapters by detection time', () => {
    const nodes = [
      prompt('p1', T0),
      node({ toolName: 'Read', timestamp: T0 + 1000 }),
      prompt('p2', T0 + 10_000),
      node({ toolName: 'Bash', timestamp: T0 + 11_000 }),
    ];
    const chapters = computeChapters(nodes);
    const records: AnomalyRecordUI[] = [
      { type: 'stall', reason: 'old', flaggedEventIds: [], detectedAt: T0 + 5000 },
      { type: 'error_thrash', reason: 'new', flaggedEventIds: [], detectedAt: T0 + 12_000 },
    ];
    expect(anomaliesFor(chapters[0], records).map((r) => r.reason)).toEqual(['old']);
    expect(anomaliesFor(chapters[1], records).map((r) => r.reason)).toEqual(['new']);
  });
});

describe('chapterBadge', () => {
  const base = computeChapters([
    prompt('p', T0),
    node({ toolName: 'Read', timestamp: T0 + 1 }),
  ])[0];

  it('anomalies outrank errors, errors outrank active', () => {
    expect(chapterBadge(base, { isLast: true, isLive: true, anomalyCount: 2 }).badge).toBe('anomalies');
    const withError = { ...base, errorCount: 1 };
    expect(chapterBadge(withError, { isLast: true, isLive: true, anomalyCount: 0 }).badge).toBe('errors');
    expect(chapterBadge(base, { isLast: true, isLive: true, anomalyCount: 0 }).badge).toBe('active');
    expect(chapterBadge(base, { isLast: false, isLive: true, anomalyCount: 0 }).badge).toBe('done');
    expect(chapterBadge(base, { isLast: true, isLive: false, anomalyCount: 0 }).badge).toBe('done');
  });

  it('marks an empty live last chapter as queued', () => {
    const empty = { ...base, actionCount: 0 };
    expect(chapterBadge(empty, { isLast: true, isLive: true, anomalyCount: 0 }).badge).toBe('queued');
  });
});

describe('computePhaseBlocks', () => {
  it('groups consecutive same-type runs into typed blocks', () => {
    const blocks = computePhaseBlocks([
      node({ toolName: 'Read', durationMs: 100 }),
      node({ toolName: 'Read', durationMs: 200 }),
      node({ toolName: 'Bash', durationMs: 50 }),
      node({ toolName: 'Bash', durationMs: 50 }),
      node({ toolName: 'Edit' }),
      node({ toolName: 'Write' }),
    ]);
    expect(blocks.map((b) => b.label)).toEqual(['Reading', 'Running', 'Editing']);
    expect(blocks[0].actionCount).toBe(2);
    expect(blocks[0].durationMs).toBe(300);
  });

  it('pools single actions and unclassified tools into Actions blocks', () => {
    const blocks = computePhaseBlocks([
      node({ toolName: 'Read' }),           // run of 1 → Actions
      node({ toolName: 'WebSearch' }),      // unclassified → Actions (merged)
      node({ toolName: 'Bash' }),
      node({ toolName: 'Bash' }),
      node({ toolName: 'Grep' }),           // unclassified, single → Actions
    ]);
    expect(blocks.map((b) => b.label)).toEqual(['Actions', 'Running', 'Actions']);
    expect(blocks[0].actionCount).toBe(2);
  });

  it('a batch node counts as a typed run on its own', () => {
    const blocks = computePhaseBlocks([
      node({ toolName: 'Read', isBatch: true, count: 4 }),
      node({ toolName: 'Edit' }),
      node({ toolName: 'Edit' }),
    ]);
    expect(blocks.map((b) => b.label)).toEqual(['Reading', 'Editing']);
    expect(blocks[0].actionCount).toBe(4);
  });

  it('tracks errors per block without changing the grouping', () => {
    const blocks = computePhaseBlocks([
      node({ toolName: 'Bash' }),
      node({ toolName: 'Bash', status: 'error' }),
    ]);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].errorCount).toBe(1);
  });

  it('never returns a flat list — everything lands in some block', () => {
    const blocks = computePhaseBlocks([node({ toolName: 'WebFetch' })]);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].label).toBe('Actions');
  });
});

describe('phase block summaries', () => {
  it('names the files a Reading/Editing block touched', () => {
    const blocks = computePhaseBlocks([
      node({ toolName: 'Read', toolInput: { file_path: '/src/codename.ts' } }),
      node({ toolName: 'Read', toolInput: { file_path: '/src/PanelApp.tsx' } }),
      node({ toolName: 'Read', toolInput: { file_path: '/src/chapters.ts' } }),
    ]);
    expect(blocks[0].summary).toBe('codename.ts, PanelApp.tsx +1');
  });

  it('names the concrete objects of an Actions block, not bare tool names', () => {
    const blocks = computePhaseBlocks([
      node({ toolName: 'ToolSearch', toolInput: { query: 'select:TaskCreate' } }),
      node({ toolName: 'Grep', toolInput: { pattern: 'errorCount' } }),
    ]);
    expect(blocks[0].label).toBe('Actions');
    expect(blocks[0].summary).toBe('ToolSearch select:TaskCreate, Grep errorCount');
  });

  it('names the command stems a Running block executed, deduped', () => {
    const blocks = computePhaseBlocks([
      node({ toolName: 'Bash', toolInput: { command: 'npm test' } }),
      node({ toolName: 'Bash', toolInput: { command: 'npm test -- --watch=false' } }),
      node({ toolName: 'Bash', toolInput: { command: 'npx tsc --noEmit' } }),
    ]);
    expect(blocks[0].summary).toBe('npm test, npx tsc');
  });
});

describe('chapter plan scoping', () => {
  it('does not leak stale pending tasks into later chapters', () => {
    const nodes = [
      prompt('build the footer', T0),
      node({ toolName: 'TaskCreate', isPlanUpdate: true, timestamp: T0 + 1,
             toolInput: { subject: 'Create Footer component' },
             detail: 'Task #1 created successfully' }),
      node({ toolName: 'TaskCreate', isPlanUpdate: true, timestamp: T0 + 2,
             toolInput: { subject: 'Integrate Footer into app layout' },
             detail: 'Task #2 created successfully' }),
      node({ toolName: 'TaskUpdate', isPlanUpdate: true, timestamp: T0 + 3,
             toolInput: { taskId: '1', status: 'completed' } }),
      node({ toolName: 'Edit', timestamp: T0 + 4 }),
      prompt('now commit it', T0 + 10_000),
      node({ toolName: 'Bash', timestamp: T0 + 11_000 }),
    ];
    const [c1, c2] = computeChapters(nodes);
    // Chapter 1 owns both tasks — including the still-pending one.
    expect(c1.plan.map((p) => p.content)).toEqual(
      ['Create Footer component', 'Integrate Footer into app layout']);
    expect(pendingPlanItems(c1).map((p) => p.content))
      .toEqual(['Integrate Footer into app layout']);
    // Chapter 2 touched no tasks: no inherited plan, no waiting rows,
    // no "1 of 2 tasks" progress ghost.
    expect(c2.plan).toEqual([]);
    expect(pendingPlanItems(c2)).toEqual([]);
  });

  it('shows an old task in a later chapter once that chapter updates it', () => {
    const nodes = [
      prompt('plan it', T0),
      node({ toolName: 'TaskCreate', isPlanUpdate: true, timestamp: T0 + 1,
             toolInput: { subject: 'Fix duplicate footer' },
             detail: 'Task #7 created successfully' }),
      prompt('go finish that task', T0 + 10_000),
      node({ toolName: 'TaskUpdate', isPlanUpdate: true, timestamp: T0 + 11_000,
             toolInput: { taskId: '7', status: 'completed' } }),
      node({ toolName: 'Edit', timestamp: T0 + 12_000 }),
    ];
    const [c1, c2] = computeChapters(nodes);
    expect(c1.plan.map((p) => p.status)).toEqual(['pending']);
    expect(c2.plan).toHaveLength(1);
    expect(c2.plan[0].status).toBe('completed');
  });

  it('keeps a prior task visible when a later chapter runs actions under it', () => {
    const nodes = [
      prompt('plan it', T0),
      node({ toolName: 'TodoWrite', isPlanUpdate: true, timestamp: T0 + 1, toolInput: {
        todos: [{ content: 'task A', status: 'in_progress', activeForm: 'Doing task A' }],
      } }),
      node({ toolName: 'Read', objective: 'Doing task A', timestamp: T0 + 2 }),
      prompt('continue', T0 + 10_000),
      node({ toolName: 'Edit', objective: 'Doing task A', timestamp: T0 + 11_000 }),
    ];
    const [, c2] = computeChapters(nodes);
    expect(c2.plan.map((p) => p.content)).toEqual(['task A']);
    expect(c2.taskGroups[0].status).toBe('in_progress');
  });

  it('a queued prompt with no work yet shows no inherited plan', () => {
    const nodes = [
      prompt('build it', T0),
      node({ toolName: 'TaskCreate', isPlanUpdate: true, timestamp: T0 + 1,
             toolInput: { subject: 'task A' }, detail: 'Task #1 created successfully' }),
      node({ toolName: 'Read', timestamp: T0 + 2 }),
      prompt('queued follow-up', T0 + 10_000),
    ];
    const chapters = computeChapters(nodes);
    expect(chapters).toHaveLength(2);
    expect(chapters[1].plan).toEqual([]);
  });
});

describe('expected stumbles (Read on a directory)', () => {
  const dirRead = () => node({
    toolName: 'Read', status: 'error', timestamp: T0 + 2,
    toolInput: { file_path: '/repo/frontend-vite/src' },
    detail: 'EISDIR: illegal operation on a directory, read',
  });

  it('does not count toward chapter error counts', () => {
    const chapters = computeChapters([
      prompt('p', T0),
      dirRead(),
      node({ toolName: 'Bash', status: 'error', timestamp: T0 + 3 }),
    ]);
    expect(chapters[0].errorCount).toBe(1); // only the real bash failure
  });

  it('does not count toward phase block error counts', () => {
    const blocks = computePhaseBlocks([dirRead(), dirRead()]);
    expect(blocks[0].errorCount).toBe(0);
  });
});

describe('TaskCreate/TaskUpdate with conversation-wide ids', () => {
  it('resolves the real id from the TaskCreate response in node.detail', () => {
    const nodes = [
      prompt('p', T0),
      node({ toolName: 'TaskCreate', isPlanUpdate: true, timestamp: T0 + 1,
             toolInput: { subject: 'Reclassify stall' },
             detail: 'Task #18 created successfully: Reclassify stall' }),
      node({ toolName: 'TaskUpdate', isPlanUpdate: true, timestamp: T0 + 2,
             toolInput: { taskId: '18', status: 'in_progress' } }),
      node({ toolName: 'TaskUpdate', isPlanUpdate: true, timestamp: T0 + 3,
             toolInput: { taskId: '18', status: 'completed' } }),
    ];
    const c = computeChapters(nodes)[0];
    expect(c.plan).toHaveLength(1);
    expect(c.plan[0].status).toBe('completed');
    expect(pendingPlanItems(c)).toHaveLength(0);
  });
});
