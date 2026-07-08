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
    for (let i = 0; i < 5; i++) {
      traceStore.addEvent(evt('s', 'pre_tool_use', 'Read', { file_path: '/same.ts' }));
    }
    expect(traceStore.getSession('s')?.anomaly?.isAnomalous).toBe(true);
    traceStore.addEvent(evt('s', 'stop'));
    expect(traceStore.getSession('s')?.anomaly).toBeUndefined();
  });

  it('renders user prompts as __prompt__ nodes with the full text in detail', () => {
    traceStore.clearActive();
    const prompt = 'Refactor the auth module and add tests for the edge cases';
    traceStore.addEvent({
      id: 'p1', sessionId: 's', kind: 'user_prompt',
      toolResponse: prompt, timestamp: Date.now(),
    });
    traceStore.addEvent(evt('s', 'pre_tool_use', 'Read', { file_path: '/auth.ts' }));

    const nodes = traceStore.getSession('s')!.nodes;
    expect(nodes[0].toolName).toBe('__prompt__');
    expect(nodes[0].detail).toBe(prompt);
    // Prompts are conversation, not actions — excluded from batching scans
    expect(nodes.some((n) => n.isBatch)).toBe(false);
  });

  describe('plan tracking (TodoWrite)', () => {
    const TODOS = {
      todos: [
        { content: 'Add JSON export',   status: 'completed',   activeForm: 'Adding JSON export' },
        { content: 'Add trace stats',   status: 'in_progress', activeForm: 'Adding trace stats' },
        { content: 'Build CLI tool',    status: 'pending',     activeForm: 'Building CLI tool' },
      ],
    };

    it('captures the plan from a TodoWrite call', () => {
      traceStore.clearActive();
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TodoWrite', TODOS));
      const plan = traceStore.getSession('s')?.plan;
      expect(plan?.items).toHaveLength(3);
      expect(plan?.items[1].status).toBe('in_progress');
    });

    it('stamps subsequent nodes with the in-progress objective', () => {
      traceStore.clearActive();
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TodoWrite', TODOS));
      traceStore.addEvent(evt('s', 'pre_tool_use', 'Edit', { file_path: '/stats.ts' }));
      const edit = traceStore.getSession('s')!.nodes.find((n) => n.toolName === 'Edit');
      expect(edit?.objective).toBe('Adding trace stats');
    });

    it('marks TodoWrite nodes as plan updates and never batches them', () => {
      traceStore.clearActive();
      for (let i = 0; i < 3; i++) {
        traceStore.addEvent(evt('s', 'pre_tool_use', 'TodoWrite', TODOS));
        traceStore.addEvent(evt('s', 'post_tool_use', 'TodoWrite'));
      }
      const nodes = traceStore.getSession('s')!.nodes.filter((n) => n.toolName === 'TodoWrite');
      expect(nodes.every((n) => n.isPlanUpdate)).toBe(true);
      expect(nodes.some((n) => n.isBatch)).toBe(false);
    });

    it('ignores malformed TodoWrite input', () => {
      traceStore.clearActive();
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TodoWrite', { todos: 'nope' }));
      expect(traceStore.getSession('s')?.plan).toBeUndefined();
    });

    it('builds the plan from TaskCreate/TaskUpdate (new Claude Code task tools)', () => {
      traceStore.clearActive();
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TaskCreate', { subject: 'Add export', activeForm: 'Adding export' }));
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TaskCreate', { subject: 'Write tests', activeForm: 'Writing tests' }));
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TaskUpdate', { taskId: '1', status: 'in_progress' }));

      const plan = traceStore.getSession('s')?.plan;
      expect(plan?.items).toHaveLength(2);
      // '?1' = provisional id (no PostToolUse response carried the real one);
      // TaskUpdate '1' still matches it via the fallback.
      expect(plan?.items[0]).toMatchObject({ id: '?1', content: 'Add export', status: 'in_progress' });
      expect(plan?.items[1].status).toBe('pending');

      // Subsequent tool calls run under the in-progress objective
      traceStore.addEvent(evt('s', 'pre_tool_use', 'Edit', { file_path: '/x.ts' }));
      const edit = traceStore.getSession('s')!.nodes.find((n) => n.toolName === 'Edit');
      expect(edit?.objective).toBe('Adding export');

      // Completing + deleting
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TaskUpdate', { taskId: '1', status: 'completed' }));
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TaskUpdate', { taskId: '2', status: 'deleted' }));
      const after = traceStore.getSession('s')?.plan;
      expect(after?.items).toHaveLength(1);
      expect(after?.items[0].status).toBe('completed');
    });

    it('accepts "#1"-style task ids', () => {
      traceStore.clearActive();
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TaskCreate', { subject: 'Only task' }));
      traceStore.addEvent(evt('s', 'pre_tool_use', 'TaskUpdate', { taskId: '#1', status: 'completed' }));
      expect(traceStore.getSession('s')?.plan?.items[0].status).toBe('completed');
    });
  });
});

describe('batch grouping vs. task attribution', () => {
  it('keeps the objective on batch nodes and splits batches at task boundaries', () => {
    traceStore.clearActive();
    const s = 'batch-objective';

    // Task A in progress → 3 reads (batchable) → task B in progress → 3 more reads.
    traceStore.addEvent(evt(s, 'pre_tool_use', 'TodoWrite', {
      todos: [
        { content: 'Task A', status: 'in_progress' },
        { content: 'Task B', status: 'pending' },
      ],
    }));
    for (const f of ['/a1.ts', '/a2.ts', '/a3.ts']) {
      traceStore.addEvent(evt(s, 'pre_tool_use', 'Read', { file_path: f }));
      traceStore.addEvent(evt(s, 'post_tool_use', 'Read'));
    }
    traceStore.addEvent(evt(s, 'pre_tool_use', 'TodoWrite', {
      todos: [
        { content: 'Task A', status: 'completed' },
        { content: 'Task B', status: 'in_progress' },
      ],
    }));
    for (const f of ['/b1.ts', '/b2.ts', '/b3.ts']) {
      traceStore.addEvent(evt(s, 'pre_tool_use', 'Read', { file_path: f }));
      traceStore.addEvent(evt(s, 'post_tool_use', 'Read'));
    }

    const batches = traceStore.getSession(s)!.nodes.filter((n) => n.isBatch);
    expect(batches).toHaveLength(2);           // NOT one 6-read batch
    expect(batches[0].objective).toBe('Task A');
    expect(batches[1].objective).toBe('Task B');
  });
});

describe('stall is a waiting notice, never an anomaly record', () => {
  it('shows the live stall state but records nothing in anomalyHistory', () => {
    traceStore.clearActive();
    const s = 'stall-no-record';
    const e = evt(s, 'pre_tool_use', 'Edit', { file_path: '/a.ts' });
    e.timestamp = Date.now() - 300_000; // pending for 5 minutes
    traceStore.addEvent(e);
    traceStore.checkStalls();

    const session = traceStore.getSession(s)!;
    expect(session.anomaly?.type).toBe('stall');       // live notice is shown
    expect(session.anomalyHistory).toHaveLength(0);    // but never recorded
  });
});

describe('TaskCreate ids come from the tool response, not a local counter', () => {
  it('matches TaskUpdate by the conversation-wide id from the response', () => {
    traceStore.clearActive();
    const s = 'real-task-ids';

    traceStore.addEvent(evt(s, 'pre_tool_use', 'TaskCreate', { subject: 'Fix the bug' }));
    const post = evt(s, 'post_tool_use', 'TaskCreate');
    post.toolResponse = 'Task #18 created successfully: Fix the bug';
    traceStore.addEvent(post);

    traceStore.addEvent(evt(s, 'pre_tool_use', 'TaskUpdate', { taskId: '18', status: 'in_progress' }));
    traceStore.addEvent(evt(s, 'post_tool_use', 'TaskUpdate'));
    traceStore.addEvent(evt(s, 'pre_tool_use', 'Edit', { file_path: '/a.ts' }));

    const session = traceStore.getSession(s)!;
    expect(session.plan?.items[0].id).toBe('18');
    expect(session.plan?.items[0].status).toBe('in_progress');
    const editNode = session.nodes.find((n) => n.toolName === 'Edit')!;
    expect(editNode.objective).toBe('Fix the bug');
  });

  it('still matches sequential ids when the response carried none', () => {
    traceStore.clearActive();
    const s = 'fallback-ids';
    traceStore.addEvent(evt(s, 'pre_tool_use', 'TaskCreate', { subject: 'Task one' }));
    traceStore.addEvent(evt(s, 'post_tool_use', 'TaskCreate')); // no parsable id
    traceStore.addEvent(evt(s, 'pre_tool_use', 'TaskUpdate', { taskId: '1', status: 'completed' }));
    expect(traceStore.getSession(s)!.plan?.items[0].status).toBe('completed');
  });
});
