import { describe, expect, it } from 'vitest';
import { extractDecisions, firstSentence, isTranscriptPath, meaningfulIntent, mineDecisions, readContextTokens } from '../src/tokenReader';

describe('firstSentence', () => {
  it('takes the first sentence and collapses whitespace', () => {
    expect(firstSentence('First thing.\nSecond thing.')).toBe('First thing.');
    expect(firstSentence('  spaced   out  ')).toBe('spaced out');
  });

  it('truncates long sentences with an ellipsis', () => {
    const long = 'a'.repeat(200) + '.';
    expect(firstSentence(long)!.length).toBeLessThanOrEqual(120);
    expect(firstSentence(long)!.endsWith('…')).toBe(true);
  });

  it('returns null for empty text', () => {
    expect(firstSentence('')).toBeNull();
    expect(firstSentence('   ')).toBeNull();
  });
});

describe('meaningfulIntent', () => {
  it('keeps real reasoning sentences', () => {
    const text = 'The failing test points at the session parser, so I need to inspect how batches are built.';
    expect(meaningfulIntent(text)).toBe(text);
  });

  it('rejects post-action reactions regardless of length', () => {
    expect(meaningfulIntent('Perfect! The tests pass now and everything is working as expected.')).toBeNull();
    expect(meaningfulIntent('Excellent!')).toBeNull();
    expect(meaningfulIntent('Good progress on the refactor so far, this is coming together nicely.')).toBeNull();
    expect(meaningfulIntent('Great — that fixed it.')).toBeNull();
  });

  it('rejects connective filler openers', () => {
    expect(meaningfulIntent("Now I'll update the second file with the same pattern as before.")).toBeNull();
    expect(meaningfulIntent('Now let me check whether the webview picks up the new message shape.')).toBeNull();
    expect(meaningfulIntent("I'll now rerun the whole suite to confirm nothing else regressed.")).toBeNull();
  });

  it('rejects anything under 40 characters', () => {
    expect(meaningfulIntent('Checking the config.')).toBeNull();
  });

  it('judges only the first sentence', () => {
    expect(meaningfulIntent('Perfect! Next I will rework the export pipeline to stream chunks.')).toBeNull();
  });
});

describe('mineDecisions', () => {
  const T = 1_700_000_000_000;
  const mine = (text: string) => mineDecisions(text, T);

  it('classifies assumption sentences', () => {
    const items = mine("I'll assume the config should stay in JSON format for this project.");
    expect(items).toHaveLength(1);
    expect(items[0].kind).toBe('assumption');
    expect(items[0].timestamp).toBe(T);
  });

  it('classifies decision sentences', () => {
    const items = mine('I went with a regex-based parser instead of pulling in a dependency.');
    expect(items).toHaveLength(1);
    expect(items[0].kind).toBe('decision');
  });

  it('extracts multiple judgment sentences from one block', () => {
    const items = mine(
      'The tests pass now. ' +
      "I'm assuming the legacy endpoint is unused by any client. " +
      'For simplicity I kept the retry count hardcoded at three attempts.'
    );
    expect(items.map((i) => i.kind)).toEqual(['assumption', 'decision']);
  });

  it('ignores plain narration and short fragments', () => {
    expect(mine('The tests pass and the build is green across all packages.')).toHaveLength(0);
    expect(mine('Went with plan B.')).toHaveLength(0); // under min length
  });

  it('skips markdown structure lines', () => {
    expect(mine('- instead of the old approach we now use the new one everywhere')).toHaveLength(0);
  });

  it('truncates very long sentences at a word boundary', () => {
    const long = "I'll assume " + 'word '.repeat(120).trim() + '.';
    const items = mine(long);
    expect(items[0].text.length).toBeLessThanOrEqual(401);
    expect(items[0].text.endsWith('…')).toBe(true);
    expect(items[0].text).not.toMatch(/wor…$/); // no mid-word chop
  });

  it('strips inline markdown from card text', () => {
    const items = mine('I went with parsing the id from the response **#18** rather than the `taskCounter` fallback.');
    expect(items[0].text).toBe('I went with parsing the id from the response #18 rather than the taskCounter fallback.');
  });
});

describe('isTranscriptPath — untrusted path guard', () => {
  it('accepts only .jsonl transcript paths', () => {
    expect(isTranscriptPath('/Users/me/.claude/projects/x/abc.jsonl')).toBe(true);
    expect(isTranscriptPath('/etc/passwd')).toBe(false);
    expect(isTranscriptPath('/Users/me/.ssh/id_rsa')).toBe(false);
    expect(isTranscriptPath(undefined)).toBe(false);
  });

  it('readContextTokens refuses a non-transcript path without touching the fs', async () => {
    expect(await readContextTokens('/etc/passwd')).toBeNull();
  });

  it('extractDecisions refuses a non-transcript path', async () => {
    expect(await extractDecisions('/etc/shadow')).toEqual([]);
  });
});
