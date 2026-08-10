import { describe, expect, it } from 'vitest';
import { isExpectedStumble, summarizeOutput, parseAskQuestions, parseToolPayload } from '../webview/src/payloadParser';
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

describe('parseAskQuestions', () => {
  const input = {
    questions: [{
      header: 'Default surface',
      question: 'Which default surface behavior should I build?',
      options: [
        { label: 'Lean sidebar + auto-open panel', description: 'x' },
        { label: 'Lean sidebar only', description: 'y' },
      ],
    }],
  };

  it('pairs each question with its options and the chosen one from the result', () => {
    const out = parseAskQuestions(input,
      'Your questions have been answered: "Which default surface behavior should I build?"="Lean sidebar + auto-open panel"');
    expect(out).toHaveLength(1);
    expect(out[0].header).toBe('Default surface');
    expect(out[0].options).toEqual(['Lean sidebar + auto-open panel', 'Lean sidebar only']);
    expect(out[0].chosen).toBe('Lean sidebar + auto-open panel');
  });

  it('leaves chosen undefined when the result has no answer pair', () => {
    expect(parseAskQuestions(input, undefined)[0].chosen).toBeUndefined();
  });

  it('captures a custom "Other" answer not present in the options', () => {
    const out = parseAskQuestions(input,
      '"Which default surface behavior should I build?"="Something I typed myself"');
    expect(out[0].chosen).toBe('Something I typed myself');
    expect(out[0].options.includes(out[0].chosen!)).toBe(false);
  });

  it('is reached via parseToolPayload for the AskUserQuestion tool', () => {
    const p = parseToolPayload('AskUserQuestion', input, '"Which default surface behavior should I build?"="Lean sidebar only"');
    expect(p.kind).toBe('askuser');
    expect(p.questions?.[0].chosen).toBe('Lean sidebar only');
  });
});
