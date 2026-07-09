import { describe, expect, it } from 'vitest';
import { isExpectedStumble, summarizeOutput } from '../webview/src/payloadParser';
import { computeMetrics } from '../webview/src/metrics';
import type { TimelineNode } from '../webview/src/components/TimelineCard';

describe('summarizeOutput — directory-read hint', () => {
  it('explains a failed Read whose path looks like a directory (empty output)', () => {
    const out = summarizeOutput(undefined, true, {
      toolName: 'Read',
      toolInput: { file_path: '/Users/me/agent-observability-demo' },
    });
    expect(out).toMatch(/path is a directory/i);
  });

  it('explains an EISDIR error regardless of path shape', () => {
    const out = summarizeOutput('EISDIR: illegal operation on a directory, read', true, {
      toolName: 'Read',
      toolInput: { file_path: '/some/dir.name' },
    });
    expect(out).toMatch(/path is a directory/i);
  });

  it('does NOT hint for a failed Read of a real file', () => {
    const out = summarizeOutput('Error: file not found', true, {
      toolName: 'Read',
      toolInput: { file_path: '/src/app.ts' },
    });
    expect(out).toMatch(/file not found/i);
  });

  it('keeps the generic fallback for non-Read failures with no output', () => {
    expect(summarizeOutput(undefined, true, { toolName: 'Bash' })).toBe('failed, no output');
    expect(summarizeOutput(undefined, true)).toBe('failed, no output');
  });
});

describe('isExpectedStumble', () => {
  const dirRead = {
    toolName: 'Read', status: 'error',
    toolInput: { file_path: '/repo/frontend-vite/src' },
    detail: 'EISDIR: illegal operation on a directory, read',
  };

  it('matches a failed Read on a directory', () => {
    expect(isExpectedStumble(dirRead)).toBe(true);
    expect(isExpectedStumble({
      toolName: 'Read', status: 'error',
      detail: 'path is a directory — Read only works on files',
    })).toBe(true);
    // extensionless path + empty output — the shape hooks actually deliver
    expect(isExpectedStumble({
      toolName: 'Read', status: 'error',
      toolInput: { file_path: '/repo/src' }, detail: '',
    })).toBe(true);
  });

  it('never matches real failures or successes', () => {
    expect(isExpectedStumble({ ...dirRead, status: 'success' })).toBe(false);
    expect(isExpectedStumble({ ...dirRead, toolName: 'Bash' })).toBe(false);
    expect(isExpectedStumble({
      toolName: 'Read', status: 'error',
      toolInput: { file_path: '/src/app.ts' }, detail: 'Error: file not found',
    })).toBe(false);
  });

  it('is excluded from session error metrics', () => {
    const base = { id: 'n1', label: 'Read', count: 1, timestamp: 0 };
    const nodes = [
      { ...base, ...dirRead } as TimelineNode,
      { ...base, id: 'n2', toolName: 'Bash', label: 'Bash', status: 'error' } as TimelineNode,
    ];
    expect(computeMetrics(nodes).errorCount).toBe(1);
  });
});
