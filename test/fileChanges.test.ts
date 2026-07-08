import { describe, expect, it } from 'vitest';
import { computeFileChanges, summarizeChanges, summarizeVerification, computeTouched, summarizeTouched, verifyChanges } from '../webview/src/fileChanges';
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
    label: 'x',
    count: 1,
    timestamp: Date.now(),
    toolInput,
    ...extra,
  };
}

describe('computeFileChanges', () => {
  it('classifies Write as created and Edit as modified', () => {
    const changes = computeFileChanges([
      node('Write', { file_path: '/p/new.ts', content: 'a\nb\nc' }),
      node('Edit',  { file_path: '/p/old.ts', old_string: 'x\ny', new_string: 'z' }),
      node('Read',  { file_path: '/p/ignored.ts' }),
      node('Bash',  { command: 'ls' }),
    ]);
    expect(changes).toHaveLength(2);
    const created  = changes.find((c) => c.path === '/p/new.ts')!;
    const modified = changes.find((c) => c.path === '/p/old.ts')!;
    expect(created.kind).toBe('created');
    expect(created.linesAdded).toBe(3);
    expect(created.linesRemoved).toBe(0);
    expect(modified.kind).toBe('modified');
    expect(modified.linesAdded).toBe(1);
    expect(modified.linesRemoved).toBe(2);
  });

  it('merges repeated edits to the same file', () => {
    const changes = computeFileChanges([
      node('Edit', { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      node('Edit', { file_path: '/p/a.ts', old_string: 'q', new_string: 'r\ns' }),
    ]);
    expect(changes).toHaveLength(1);
    expect(changes[0].edits).toBe(2);
    expect(changes[0].linesAdded).toBe(3);
  });

  it('a Write after Edits upgrades the file to created', () => {
    const changes = computeFileChanges([
      node('Edit',  { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      node('Write', { file_path: '/p/a.ts', content: 'fresh' }),
    ]);
    expect(changes[0].kind).toBe('created');
  });

  it('sums MultiEdit sub-edits', () => {
    const changes = computeFileChanges([
      node('MultiEdit', {
        file_path: '/p/multi.ts',
        edits: [
          { old_string: 'a\nb', new_string: 'c' },
          { old_string: 'd', new_string: 'e\nf\ng' },
        ],
      }),
    ]);
    expect(changes[0].linesAdded).toBe(4);
    expect(changes[0].linesRemoved).toBe(3);
  });

  it('walks batch items using their per-item toolInput', () => {
    const changes = computeFileChanges([
      node('Edit', undefined, {
        isBatch: true,
        count: 3,
        batchItems: [
          { label: 'a', status: 'success', toolInput: { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' } },
          { label: 'b', status: 'success', toolInput: { file_path: '/p/b.ts', old_string: 'x', new_string: 'y' } },
          { label: 'a2', status: 'success', toolInput: { file_path: '/p/a.ts', old_string: 'q', new_string: 'r' } },
        ],
      }),
    ]);
    expect(changes).toHaveLength(2);
    expect(changes.find((c) => c.path === '/p/a.ts')!.edits).toBe(2);
  });

  it('sorts created before modified', () => {
    const changes = computeFileChanges([
      node('Edit',  { file_path: '/p/z-mod.ts', old_string: 'x', new_string: 'y' }),
      node('Write', { file_path: '/p/a-new.ts', content: 'n' }),
    ]);
    expect(changes[0].kind).toBe('created');
  });
});

describe('summarizeChanges', () => {
  it('renders counts and omits zero categories', () => {
    const changes = computeFileChanges([
      node('Write', { file_path: '/p/a.ts', content: 'n' }),
      node('Edit',  { file_path: '/p/b.ts', old_string: 'x', new_string: 'y' }),
      node('Edit',  { file_path: '/p/c.ts', old_string: 'x', new_string: 'y' }),
    ]);
    expect(summarizeChanges(changes)).toBe('1 created · 2 modified');
    expect(summarizeChanges([])).toBe('');
  });
});

describe('verifyChanges', () => {
  it('marks a file verified when a check command succeeds after its last edit', () => {
    const statuses = verifyChanges([
      node('Edit', { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      node('Bash', { command: 'npx vitest run' }),
    ]);
    expect(statuses.get('/p/a.ts')).toBe('verified');
  });

  it('marks a file failed when the latest check after the edit errored', () => {
    const statuses = verifyChanges([
      node('Edit', { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      node('Bash', { command: 'npm test' }, { status: 'error' }),
    ]);
    expect(statuses.get('/p/a.ts')).toBe('failed');
  });

  it('marks a file unverified when the agent edits AFTER the last check', () => {
    const statuses = verifyChanges([
      node('Edit', { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      node('Bash', { command: 'npm test' }),
      node('Edit', { file_path: '/p/a.ts', old_string: 'y', new_string: 'z' }),
    ]);
    expect(statuses.get('/p/a.ts')).toBe('unverified');
  });

  it('ignores bash commands that are not checks', () => {
    const statuses = verifyChanges([
      node('Edit', { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      node('Bash', { command: 'git status' }),
      node('Bash', { command: 'ls -la' }),
    ]);
    expect(statuses.get('/p/a.ts')).toBe('unverified');
  });

  it('counts a command naming the file as a check', () => {
    const statuses = verifyChanges([
      node('Write', { file_path: '/p/deploy.sh', content: 'echo hi' }),
      node('Bash', { command: 'bash /p/deploy.sh' }),
    ]);
    expect(statuses.get('/p/deploy.sh')).toBe('verified');
  });

  it('resolves edits and checks inside batch nodes', () => {
    const batch = node('Bash', undefined, {
      isBatch: true,
      count: 3,
      batchItems: [
        { label: 'a', status: 'success', toolInput: { command: 'ls' } },
        { label: 'b', status: 'success', toolInput: { command: 'npx tsc --noEmit' } },
        { label: 'c', status: 'success', toolInput: { command: 'git diff' } },
      ],
    });
    const statuses = verifyChanges([
      node('Edit', { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      batch,
    ]);
    expect(statuses.get('/p/a.ts')).toBe('verified');
  });
});

describe('summarizeVerification', () => {
  it('is empty when everything is verified', () => {
    const m = new Map([['/a', 'verified'], ['/b', 'verified']] as const);
    expect(summarizeVerification(new Map(m))).toBe('');
  });

  it('reports unverified and failing counts', () => {
    const m = new Map<string, 'verified' | 'failed' | 'unverified'>([
      ['/a', 'unverified'], ['/b', 'failed'], ['/c', 'verified'],
    ]);
    const line = summarizeVerification(m);
    expect(line).toContain('1 of 3 changed files never exercised');
    expect(line).toContain('1 failing');
  });
});

describe('computeTouched', () => {
  it('collects reads and changes, stronger claim winning per file', () => {
    const touched = computeTouched([
      node('Read',  { file_path: '/p/a.ts' }),
      node('Read',  { file_path: '/p/b.ts' }),
      node('Edit',  { file_path: '/p/a.ts', old_string: 'x', new_string: 'y' }),
      node('Write', { file_path: '/p/c.ts', content: 'new' }),
      node('Bash',  { command: 'ls' }),
    ]);
    const byPath = new Map(touched.map((t) => [t.path, t.kind]));
    expect(byPath.get('/p/a.ts')).toBe('modified'); // read then edited → modified wins
    expect(byPath.get('/p/b.ts')).toBe('read');
    expect(byPath.get('/p/c.ts')).toBe('created');
    expect(touched).toHaveLength(3);
  });

  it('resolves batch items', () => {
    const batch = node('Read', undefined, {
      isBatch: true,
      count: 2,
      batchItems: [
        { label: 'a', status: 'success', toolInput: { file_path: '/p/x.ts' } },
        { label: 'b', status: 'success', toolInput: { file_path: '/p/y.ts' } },
      ],
    });
    expect(computeTouched([batch])).toHaveLength(2);
  });

  it('summarizes counts', () => {
    expect(summarizeTouched([
      { path: '/a', kind: 'read' },
      { path: '/b', kind: 'read' },
      { path: '/c', kind: 'modified' },
    ])).toBe('2 read · 1 modified');
  });
});
