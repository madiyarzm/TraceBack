import { beforeAll, describe, expect, it, vi } from 'vitest';

vi.mock('vscode', () => ({
  EventEmitter: class {
    private listeners: Array<(v: unknown) => void> = [];
    event = (cb: (v: unknown) => void) => {
      this.listeners.push(cb);
      return { dispose: () => {} };
    };
    fire = (value: unknown) => {
      this.listeners.forEach((l) => l(value));
    };
    dispose = () => {
      this.listeners = [];
    };
  },
}));

beforeAll(() => {
  vi.spyOn(console, 'log').mockImplementation(() => {});
});

const { traceStore } = await import('../src/traceStore');
import type { TraceEvent } from '../src/traceStore';

let nextId = 0;
const id = () => `evt-${++nextId}`;

function evt(
  sessionId: string,
  kind: TraceEvent['kind'],
  toolName?: string,
  input: Record<string, unknown> = {},
  isError = false
): TraceEvent {
  return {
    id:        id(),
    sessionId,
    kind,
    toolName,
    toolInput: input,
    isError,
    timestamp: Date.now(),
  };
}

describe('traceStore', () => {
  it('routes events to the session named by sessionId', () => {
    traceStore.clearActive();
    traceStore.addEvent(evt('s-a', 'pre_tool_use', 'Read', { file_path: '/a.ts' }));
    traceStore.addEvent(evt('s-b', 'pre_tool_use', 'Edit', { file_path: '/b.ts' }));

    const a = traceStore.getSession('s-a');
    const b = traceStore.getSession('s-b');

    expect(a?.nodes.some((n) => n.toolName === 'Read')).toBe(true);
    expect(b?.nodes.some((n) => n.toolName === 'Edit')).toBe(true);
    expect(a?.nodes.some((n) => n.toolName === 'Edit')).toBe(false);
  });

  it('flips pending -> success on PostToolUse', () => {
    traceStore.clearActive();
    traceStore.addEvent(evt('s', 'pre_tool_use', 'Read', { file_path: '/x.ts' }));
    let nodes = traceStore.getSession('s')!.nodes.filter((n) => n.toolName !== '__thinking__');
    expect(nodes[0].status).toBe('pending');

    traceStore.addEvent(evt('s', 'post_tool_use', 'Read', {}, false));
    nodes = traceStore.getSession('s')!.nodes.filter((n) => n.toolName !== '__thinking__');
    expect(nodes[0].status).toBe('success');
  });

  it('marks tool node as error when PostToolUse has isError=true', () => {
    traceStore.clearActive();
    traceStore.addEvent(evt('s', 'pre_tool_use', 'Bash', { command: 'oops' }));
    traceStore.addEvent(evt('s', 'post_tool_use', 'Bash', {}, true));
    const node = traceStore
      .getSession('s')!
      .nodes.find((n) => n.toolName === 'Bash');
    expect(node?.status).toBe('error');
  });

  it('collapses 3+ consecutive same-tool calls into a batch node', () => {
    traceStore.clearActive();
    for (let i = 0; i < 4; i++) {
      traceStore.addEvent(evt('s', 'pre_tool_use', 'Read', { file_path: `/f${i}.ts` }));
      traceStore.addEvent(evt('s', 'post_tool_use', 'Read', {}, false));
    }
    const nodes = traceStore.getSession('s')!.nodes;
    const batch = nodes.find((n) => n.isBatch);
    expect(batch).toBeDefined();
    expect(batch?.count).toBe(4);
    expect(batch?.batchItems).toHaveLength(4);
  });

  it('clearActive wipes every session', () => {
    traceStore.clearActive();
    traceStore.addEvent(evt('s1', 'pre_tool_use', 'Read'));
    traceStore.addEvent(evt('s2', 'pre_tool_use', 'Edit'));
    traceStore.clearActive();
    expect(traceStore.getSession('s1')).toBeUndefined();
    expect(traceStore.getSession('s2')).toBeUndefined();
  });

  it('marks the session stopped on Stop event', () => {
    traceStore.clearActive();
    traceStore.addEvent(evt('s', 'pre_tool_use', 'Read'));
    traceStore.addEvent(evt('s', 'post_tool_use', 'Read'));
    traceStore.addEvent(evt('s', 'stop'));
    expect(traceStore.getSession('s')?.stopped).toBe(true);
  });

  it('clears anomaly state once the session is stopped', () => {
    traceStore.clearActive();
    for (let i = 0; i < 4; i++) {
      traceStore.addEvent(evt('s', 'pre_tool_use', 'Read', { file_path: '/same.ts' }));
    }
    expect(traceStore.getSession('s')?.anomaly?.isAnomalous).toBe(true);
    traceStore.addEvent(evt('s', 'stop'));
    expect(traceStore.getSession('s')?.anomaly).toBeUndefined();
  });
});
