import { describe, expect, it } from 'vitest';
import { classifyNode, computePhases } from '../webview/src/phases';
import type { TimelineNode } from '../webview/src/components/TimelineCard';

let nextId = 0;
function node(
  toolName: string,
  extra: Partial<TimelineNode> = {}
): TimelineNode {
  return {
    id: `n${++nextId}`,
    toolName,
    status: 'success',
    label: 'x',
    count: 1,
    timestamp: 1000 + nextId,
    ...extra,
  };
}

describe('classifyNode', () => {
  it('classifies read-side tools as explore', () => {
    for (const t of ['Read', 'Grep', 'Glob', 'WebSearch', 'WebFetch', 'Agent']) {
      expect(classifyNode(node(t))).toBe('explore');
    }
  });

  it('classifies write-side tools as build', () => {
    for (const t of ['Edit', 'Write', 'MultiEdit', 'NotebookEdit']) {
      expect(classifyNode(node(t))).toBe('build');
    }
  });

  it('classifies test/build Bash commands as verify', () => {
    expect(classifyNode(node('Bash', { toolInput: { command: 'npm test' } }))).toBe('verify');
    expect(classifyNode(node('Bash', { toolInput: { command: 'npx vitest run' } }))).toBe('verify');
    expect(classifyNode(node('Bash', { toolInput: { command: 'tsc -p ./' } }))).toBe('verify');
  });

  it('classifies other Bash commands as build', () => {
    expect(classifyNode(node('Bash', { toolInput: { command: 'git add .' } }))).toBe('build');
    expect(classifyNode(node('Bash', { toolInput: { command: 'mkdir src' } }))).toBe('build');
  });

  it('excludes thinking and plan-update nodes', () => {
    expect(classifyNode(node('__thinking__'))).toBeNull();
    expect(classifyNode(node('TodoWrite', { isPlanUpdate: true }))).toBeNull();
  });
});

describe('computePhases', () => {
  it('merges consecutive same-phase nodes into segments', () => {
    const segments = computePhases([
      node('Read'), node('Grep'), node('Read'),       // explore ×3
      node('Edit'), node('Write'),                    // build ×2
      node('Bash', { toolInput: { command: 'npm test' } }), // verify ×1
    ]);
    expect(segments.map((s) => [s.phase, s.count])).toEqual([
      ['explore', 3], ['build', 2], ['verify', 1],
    ]);
  });

  it('keeps oscillation as separate segments (the pathology signal)', () => {
    const segments = computePhases([
      node('Edit'), node('Read'), node('Edit'), node('Read'),
    ]);
    expect(segments.map((s) => s.phase)).toEqual(['build', 'explore', 'build', 'explore']);
  });

  it('weights collapsed nodes by their count', () => {
    const segments = computePhases([node('Read', { count: 5 })]);
    expect(segments[0].count).toBe(5);
  });

  it('skips excluded nodes without breaking adjacency', () => {
    const segments = computePhases([
      node('Read'),
      node('__thinking__'),
      node('Read'),
    ]);
    expect(segments).toHaveLength(1);
    expect(segments[0].count).toBe(2);
  });
});
