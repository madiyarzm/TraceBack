import { describe, expect, it } from 'vitest';
import { summarizeOutput } from '../webview/src/payloadParser';

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
