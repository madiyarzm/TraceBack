import { describe, expect, it } from 'vitest';
import {
  anomaliesFor, chapterBadge, computeChapters, pendingPlanItems,
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
