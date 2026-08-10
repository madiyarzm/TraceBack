import { describe, expect, it } from 'vitest';
import { formatShellCommand, splitCommand } from '../webview/src/shellFormat';

describe('splitCommand', () => {
  it('leaves a simple command as one stage', () => {
    expect(splitCommand('ls -la')).toEqual([{ text: 'ls -la', operator: '' }]);
  });

  it('splits on && and keeps the operator on the preceding stage', () => {
    expect(splitCommand('cd x && grep foo f')).toEqual([
      { text: 'cd x', operator: '&&' },
      { text: 'grep foo f', operator: '' },
    ]);
  });

  it('splits on pipes and semicolons', () => {
    expect(splitCommand('cat f | grep x | head -3')).toEqual([
      { text: 'cat f', operator: '|' },
      { text: 'grep x', operator: '|' },
      { text: 'head -3', operator: '' },
    ]);
    expect(splitCommand('a; b; c')).toEqual([
      { text: 'a', operator: ';' },
      { text: 'b', operator: ';' },
      { text: 'c', operator: '' },
    ]);
  });

  it('distinguishes || from |', () => {
    expect(splitCommand('a || b')).toEqual([
      { text: 'a', operator: '||' },
      { text: 'b', operator: '' },
    ]);
  });

  it('does NOT split on operators inside double or single quotes', () => {
    expect(splitCommand('echo "a && b" && ls')).toEqual([
      { text: 'echo "a && b"', operator: '&&' },
      { text: 'ls', operator: '' },
    ]);
    expect(splitCommand("grep 'x | y' file")).toEqual([
      { text: "grep 'x | y' file", operator: '' },
    ]);
  });

  it('keeps an escaped-pipe grep alternation together (the real-world wall case)', () => {
    const cmd = 'grep -n "openMap\\|reveal\\|focus" src/extension.ts && echo done';
    expect(splitCommand(cmd)).toEqual([
      { text: 'grep -n "openMap\\|reveal\\|focus" src/extension.ts', operator: '&&' },
      { text: 'echo done', operator: '' },
    ]);
  });

  it('drops empty stages from a trailing operator', () => {
    expect(splitCommand('ls &&')).toEqual([{ text: 'ls', operator: '&&' }]);
  });
});

describe('formatShellCommand', () => {
  it('returns one line per stage with trailing operators', () => {
    expect(formatShellCommand('cd x && echo "===" && grep foo f | head -3')).toEqual([
      'cd x &&',
      'echo "===" &&',
      'grep foo f |',
      'head -3',
    ]);
  });

  it('returns a single line for a plain command', () => {
    expect(formatShellCommand('npm test')).toEqual(['npm test']);
  });
});
