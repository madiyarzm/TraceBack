import { describe, expect, it } from 'vitest';
import { detectAnomaly, NO_ANOMALY } from '../src/anomalyDetector';
import type { TraceEvent } from '../src/traceStore';

const NOW = 1_700_000_000_000;
const CWD = '/home/user/project';
let nextId = 0;
const id = () => `evt-${++nextId}`;

function pre(toolName: string, input: Record<string, unknown> = {}, t = NOW): TraceEvent {
  return {
    id:        id(),
    sessionId: 's1',
    kind:      'pre_tool_use',
    toolName,
    toolInput: input,
    cwd:       CWD,
    timestamp: t,
  };
}

function post(toolName: string, isError = false, t = NOW, input?: Record<string, unknown>): TraceEvent {
  return {
    id:        id(),
    sessionId: 's1',
    kind:      'post_tool_use',
    toolName,
    toolInput: input,
    isError,
    timestamp: t,
  };
}

describe('detectAnomaly', () => {
  it('returns NO_ANOMALY for empty event list', () => {
    expect(detectAnomaly([], NOW)).toEqual(NO_ANOMALY);
  });

  it('returns NO_ANOMALY for healthy mixed traffic', () => {
    const events = [
      pre('Read', { file_path: `${CWD}/a.ts` }),
      post('Read', false),
      pre('Edit', { file_path: `${CWD}/a.ts` }),
      post('Edit', false),
      pre('Bash', { command: 'ls' }),
      post('Bash', false),
    ];
    expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
  });

  describe('near_duplicate_loop', () => {
    /** n failed runs of `cmd`, interleaved with successful reads so
     *  error_thrash (3 consecutive failed posts) never masks the loop rule. */
    function failedRuns(cmd: string, n: number): TraceEvent[] {
      const events: TraceEvent[] = [];
      for (let i = 0; i < n; i++) {
        events.push(pre('Bash', { command: cmd }));
        events.push(post('Bash', true, NOW, { command: cmd }));
        events.push(pre('Read', { file_path: `/pad${i}.ts` }));
        events.push(post('Read', false));
      }
      return events;
    }

    it('flags the exact same bash command failing 3 times', () => {
      const a = detectAnomaly(failedRuns('npm test', 3), NOW);
      expect(a.isAnomalous).toBe(true);
      expect(a.type).toBe('near_duplicate_loop');
      expect(a.severity).toBe('high');
      expect(a.flaggedEventIds).toHaveLength(3);
    });

    it('does NOT flag 2 failed runs (under threshold)', () => {
      expect(detectAnomaly(failedRuns('npm test', 2), NOW).isAnomalous).toBe(false);
    });

    it('does NOT flag when a run of the same command succeeded in between', () => {
      const events = [
        ...failedRuns('npm test', 2),
        pre('Bash', { command: 'npm test' }),
        post('Bash', false, NOW, { command: 'npm test' }),
        ...failedRuns('npm test', 2),
      ];
      expect(detectAnomaly(events, NOW).type).not.toBe('near_duplicate_loop');
    });

    it('does NOT flag the same base command with different args', () => {
      const events = [
        pre('Bash', { command: 'python test.py' }),
        post('Bash', true, NOW, { command: 'python test.py' }),
        pre('Bash', { command: 'python test.py --verbose' }),
        post('Bash', true, NOW, { command: 'python test.py --verbose' }),
        pre('Bash', { command: 'python test.py -x' }),
        post('Bash', true, NOW, { command: 'python test.py -x' }),
      ];
      expect(detectAnomaly(events, NOW).type).not.toBe('near_duplicate_loop');
    });

    it('flags the same file read 5 times with no edit in between', () => {
      const events = Array.from({ length: 5 }, () =>
        pre('Read', { file_path: '/loop.ts' })
      );
      const a = detectAnomaly(events, NOW);
      expect(a.type).toBe('near_duplicate_loop');
      expect(a.flaggedEventIds).toHaveLength(5);
    });

    it('does NOT flag 4 reads of the same file (under threshold)', () => {
      const events = Array.from({ length: 4 }, () =>
        pre('Read', { file_path: '/loop.ts' })
      );
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });

    it('does NOT flag read→edit→re-read verification cycles', () => {
      const FILE = `${CWD}/iter.ts`;
      const events: TraceEvent[] = [];
      for (let i = 0; i < 5; i++) {
        events.push(pre('Read', { file_path: FILE }));
        events.push(pre('Edit', { file_path: FILE }));
      }
      events.push(pre('Read', { file_path: FILE }));
      expect(detectAnomaly(events, NOW).type).not.toBe('near_duplicate_loop');
    });

    it('does NOT flag reads of different files', () => {
      const events = Array.from({ length: 5 }, (_, i) =>
        pre('Read', { file_path: `/f${i}.ts` })
      );
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });

    it('does NOT flag repeated edits to the same file (iterative editing)', () => {
      const events = Array.from({ length: 6 }, (_, i) =>
        pre('Edit', { file_path: `${CWD}/a.ts`, old_string: `v${i}`, new_string: `v${i + 1}` })
      );
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  describe('error_thrash', () => {
    it('flags 3 consecutive failed PostToolUse', () => {
      const events = [
        pre('Bash', { command: 'bad cmd' }),
        post('Bash', true),
        pre('Edit', { file_path: '/a.ts' }),
        post('Edit', true),
        pre('Bash', { command: 'other thing' }),
        post('Bash', true),
      ];
      const a = detectAnomaly(events, NOW);
      expect(a.isAnomalous).toBe(true);
      expect(a.type).toBe('error_thrash');
      expect(a.severity).toBe('high');
    });

    it('self-clears after a successful PostToolUse', () => {
      const events = [
        pre('Bash', { command: 'a' }), post('Bash', true),
        pre('Edit', { file_path: '/b' }), post('Edit', true),
        pre('Bash', { command: 'c' }), post('Bash', true),
        pre('Bash', { command: 'd' }), post('Bash', false),
      ];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  describe('stall', () => {
    it('flags a PreToolUse pending longer than 120s (severity medium)', () => {
      const events = [pre('Read', { file_path: '/slow.ts' }, NOW - 121_000)];
      const a = detectAnomaly(events, NOW);
      expect(a.isAnomalous).toBe(true);
      expect(a.type).toBe('stall');
      expect(a.severity).toBe('medium');
      expect(a.description).toMatch(/pending for \d+s/);
      expect(a.description).toMatch(/waiting for your approval/i);
    });

    it('does NOT flag a PreToolUse under the threshold', () => {
      const events = [pre('Read', { file_path: '/fast.ts' }, NOW - 90_000)];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });

    it('does NOT flag a stall once PostToolUse arrives', () => {
      const events = [
        pre('Read', { file_path: '/slow.ts' }, NOW - 121_000),
        post('Read', false, NOW - 100),
      ];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });

    it('does NOT flag a stall while Claude waits on the user (Notification)', () => {
      const events: TraceEvent[] = [
        pre('Bash', { command: 'git commit' }, NOW - 300_000),
        {
          id: id(), sessionId: 's1', kind: 'notification',
          toolResponse: 'Claude needs your permission to use Bash',
          timestamp: NOW - 299_000,
        },
      ];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  describe('context_spiral', () => {
    it('flags 8 consecutive reads with no edit/write/bash', () => {
      const events: TraceEvent[] = [];
      for (let i = 0; i < 8; i++) {
        events.push(pre('Read', { file_path: `/f${i}.ts` }));
        events.push(post('Read', false));
      }
      const a = detectAnomaly(events, NOW);
      expect(a.type).toBe('context_spiral');
      expect(a.severity).toBe('medium');
      expect(a.flaggedEventIds).toHaveLength(8);
    });

    it('resets the run when an Edit appears in between', () => {
      const events: TraceEvent[] = [];
      for (let i = 0; i < 5; i++) events.push(pre('Read', { file_path: `/f${i}.ts` }));
      events.push(pre('Edit', { file_path: `${CWD}/f0.ts` }));
      for (let i = 5; i < 10; i++) events.push(pre('Read', { file_path: `/f${i}.ts` }));
      expect(detectAnomaly(events, NOW).type).not.toBe('context_spiral');
    });

    it('does NOT flag 7 reads (under threshold)', () => {
      const events = Array.from({ length: 7 }, (_, i) =>
        pre('Read', { file_path: `/f${i}.ts` })
      );
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  describe('scope_creep', () => {
    it('flags an Edit outside the session cwd', () => {
      const events = [
        pre('Read', { file_path: `${CWD}/a.ts` }),
        post('Read', false),
        pre('Edit', { file_path: '/etc/config/app.ts' }),
      ];
      const a = detectAnomaly(events, NOW);
      expect(a.type).toBe('scope_creep');
      expect(a.severity).toBe('medium');
    });

    it('does NOT flag edits inside the cwd', () => {
      const events = [pre('Edit', { file_path: `${CWD}/src/a.ts` })];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });

    it('flags a sibling directory even when it shares the cwd string prefix', () => {
      const events = [pre('Edit', { file_path: `${CWD}-backup/a.ts` })];
      expect(detectAnomaly(events, NOW).type).toBe('scope_creep');
    });

    it('does NOT fire without a known cwd', () => {
      const e = pre('Edit', { file_path: '/etc/passwd' });
      delete (e as { cwd?: string }).cwd;
      expect(detectAnomaly([e], NOW).isAnomalous).toBe(false);
    });
  });

  describe('read→edit iteration is never an anomaly (thrash detector removed)', () => {
    it('does NOT flag repeated read→edit rounds on the same file', () => {
      const FILE = `${CWD}/logger.ts`;
      const events: TraceEvent[] = [];
      for (let i = 0; i < 4; i++) {
        events.push(pre('Read', { file_path: FILE }));
        events.push(post('Read', false));
        events.push(pre('Edit', { file_path: FILE, old_string: `v${i}`, new_string: `v${i + 1}` }));
        events.push(post('Edit', false));
      }
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  describe('anomaly shape', () => {
    it('every anomaly carries type, severity, title, description', () => {
      const events = Array.from({ length: 5 }, () =>
        pre('Read', { file_path: '/loop.ts' })
      );
      const a = detectAnomaly(events, NOW);
      expect(a.type).toBeTruthy();
      expect(a.severity).toBeTruthy();
      expect(a.title).toBeTruthy();
      expect(a.description).toBeTruthy();
      expect(a.reason).toBe(a.description); // legacy alias stays in sync
    });
  });

  it('prioritizes stall over loops (most acute first)', () => {
    const events = [
      pre('Read', { file_path: '/x' }),
      pre('Read', { file_path: '/x' }),
      pre('Read', { file_path: '/x' }, NOW - 150_000),
    ];
    expect(detectAnomaly(events, NOW).type).toBe('stall');
  });
});

describe('scope_creep whitelist', () => {
  it("does NOT flag writes into Claude's own config/memory directories", () => {
    const events = [
      pre('Read', { file_path: `${CWD}/a.ts` }),
      pre('Edit', { file_path: '/Users/me/.claude/projects/x/memory/notes.md' }),
    ];
    expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
  });
});
