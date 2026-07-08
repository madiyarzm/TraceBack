import { describe, expect, it } from 'vitest';
import { annotateReviewFile, netKindOf } from '../webview/src/review';
import type { TimelineNode } from '../webview/src/components/TimelineCard';

let nextId = 0;
function node(
  toolName: string,
  toolInput?: Record<string, unknown>,
  extra: Partial<TimelineNode> = {}
): TimelineNode {
  return {
    id: `n${++nextId}`,
    toolName,
    status: 'success',
    label: extra.label ?? toolName,
    count: 1,
    timestamp: Date.now(),
    toolInput,
    ...extra,
  };
}

describe('netKindOf', () => {
  it('classifies baseline/current combinations', () => {
    expect(netKindOf(null, 'x')).toBe('created');
    expect(netKindOf('x', null)).toBe('deleted');
    expect(netKindOf('x', 'x')).toBe('unchanged');
    expect(netKindOf('x', 'y')).toBe('modified');
  });
});

describe('annotateReviewFile', () => {
  const FILE = '/p/server.ts';

  it('collects intents from edits to the file, deduped and in order', () => {
    const ann = annotateReviewFile([
      node('Edit', { file_path: FILE }, { intent: 'The retry logic needs a backoff to avoid hammering the endpoint.' }),
      node('Edit', { file_path: FILE }, { intent: 'The retry logic needs a backoff to avoid hammering the endpoint.' }),
      node('Edit', { file_path: FILE }, { intent: 'The timeout should come from config rather than a literal.' }),
      node('Edit', { file_path: '/other.ts' }, { intent: 'Unrelated edit reasoning that must not leak in.' }),
    ], FILE);
    expect(ann.editCount).toBe(3);
    expect(ann.intents).toEqual([
      'The retry logic needs a backoff to avoid hammering the endpoint.',
      'The timeout should come from config rather than a literal.',
    ]);
  });

  it('attributes the nearest failed bash before the FIRST edit as the trigger', () => {
    const ann = annotateReviewFile([
      node('Bash', { command: 'npm test' }, { status: 'error', label: 'npm test' }),
      node('Edit', { file_path: FILE }),
      node('Bash', { command: 'npm test' }, { status: 'error', label: 'npm test again' }),
      node('Edit', { file_path: FILE }),
    ], FILE);
    expect(ann.triggeredBy).toBe('npm test');
  });

  it('has no trigger when no bash failed before the first edit', () => {
    const ann = annotateReviewFile([
      node('Bash', { command: 'ls' }, { label: 'ls' }),
      node('Edit', { file_path: FILE }),
    ], FILE);
    expect(ann.triggeredBy).toBeUndefined();
  });

  it('counts reads of the file and resolves batch items', () => {
    const batch = node('Read', undefined, {
      isBatch: true,
      count: 3,
      batchItems: [
        { label: 'a', status: 'success', toolInput: { file_path: FILE } },
        { label: 'b', status: 'success', toolInput: { file_path: '/other.ts' } },
        { label: 'c', status: 'success', toolInput: { file_path: FILE } },
      ],
    });
    const ann = annotateReviewFile([batch, node('Edit', { file_path: FILE })], FILE);
    expect(ann.readCount).toBe(2);
    expect(ann.editCount).toBe(1);
  });
});
