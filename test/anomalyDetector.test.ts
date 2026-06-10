import { describe, expect, it } from 'vitest';
import { detectAnomaly, NO_ANOMALY } from '../src/anomalyDetector';
import type { TraceEvent } from '../src/traceStore';

const NOW = 1_700_000_000_000;
let nextId = 0;
const id = () => `evt-${++nextId}`;

function pre(toolName: string, input: Record<string, unknown> = {}, t = NOW): TraceEvent {
  return {
    id:        id(),
    sessionId: 's1',
    kind:      'pre_tool_use',
    toolName,
    toolInput: input,
    timestamp: t,
  };
}

function post(toolName: string, isError = false, t = NOW): TraceEvent {
  return {
    id:        id(),
    sessionId: 's1',
    kind:      'post_tool_use',
    toolName,
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
      pre('Read', { file_path: '/a.ts' }),
      post('Read', false),
      pre('Edit', { file_path: '/a.ts' }),
      post('Edit', false),
      pre('Bash', { command: 'ls' }),
      post('Bash', false),
    ];
    expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
  });

  describe('repeater', () => {
    it('flags 4 identical PreToolUse calls in a row', () => {
      const events = Array.from({ length: 4 }, () =>
        pre('Read', { file_path: '/loop.ts' })
      );
      const a = detectAnomaly(events, NOW);
      expect(a.isAnomalous).toBe(true);
      expect(a.type).toBe('repeater');
      expect(a.flaggedEventIds).toHaveLength(4);
    });

    it('does NOT flag 3 identical calls (under threshold)', () => {
      const events = Array.from({ length: 3 }, () =>
        pre('Read', { file_path: '/loop.ts' })
      );
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });

    it('does NOT flag same tool with different inputs', () => {
      const events = [
        pre('Read', { file_path: '/a.ts' }),
        pre('Read', { file_path: '/b.ts' }),
        pre('Read', { file_path: '/c.ts' }),
        pre('Read', { file_path: '/d.ts' }),
      ];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  describe('error_thrash', () => {
    it('flags 3 consecutive failed PostToolUse (with varied inputs to avoid repeater)', () => {
      const events = [
        pre('Bash', { command: 'bad' }),
        post('Bash', true),
        pre('Bash', { command: 'worse' }),
        post('Bash', true),
        pre('Bash', { command: 'terrible' }),
        post('Bash', true),
      ];
      const a = detectAnomaly(events, NOW);
      expect(a.isAnomalous).toBe(true);
      expect(a.type).toBe('error_thrash');
    });

    it('self-clears after a successful PostToolUse', () => {
      const events = [
        pre('Bash', { command: 'a' }), post('Bash', true),
        pre('Bash', { command: 'b' }), post('Bash', true),
        pre('Bash', { command: 'c' }), post('Bash', true),
        pre('Bash', { command: 'd' }), post('Bash', false),
      ];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  describe('stall', () => {
    it('flags a PreToolUse pending longer than 60s', () => {
      const events = [pre('Read', { file_path: '/slow.ts' }, NOW - 61_000)];
      const a = detectAnomaly(events, NOW);
      expect(a.isAnomalous).toBe(true);
      expect(a.type).toBe('stall');
      expect(a.reason).toMatch(/Stalled/i);
    });

    it('does NOT flag a recent PreToolUse', () => {
      const events = [pre('Read', { file_path: '/fast.ts' }, NOW - 5_000)];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });

    it('does NOT flag a stall once PostToolUse arrives', () => {
      const events = [
        pre('Read', { file_path: '/slow.ts' }, NOW - 61_000),
        post('Read', false, NOW - 100),
      ];
      expect(detectAnomaly(events, NOW).isAnomalous).toBe(false);
    });
  });

  it('prioritizes stall over repeater (most acute first)', () => {
    const events = [
      pre('Read', { file_path: '/x' }),
      pre('Read', { file_path: '/x' }),
      pre('Read', { file_path: '/x' }),
      pre('Read', { file_path: '/x' }, NOW - 90_000),
    ];
    expect(detectAnomaly(events, NOW).type).toBe('stall');
  });
});
