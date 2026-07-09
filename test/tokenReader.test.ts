import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  extractDecisions, extractIntentForTool, firstSentence, isTranscriptPath,
  meaningfulIntent, mineDecisions, noteTranscript, readContextTokens,
} from '../src/tokenReader';

describe('firstSentence', () => {
  it('takes the first sentence and collapses whitespace', () => {
    expect(firstSentence('First thing.\nSecond thing.')).toBe('First thing.');
    expect(firstSentence('  spaced   out  ')).toBe('spaced out');
  });

  it('truncates long sentences with an ellipsis', () => {
    const long = 'a'.repeat(300) + '.';
    expect(firstSentence(long)!.length).toBeLessThanOrEqual(220);
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

  it('keeps "Now let me…" action announcements — that IS the intent', () => {
    expect(meaningfulIntent("Now I'll update the second file with the same pattern as before."))
      .toBe("Now I'll update the second file with the same pattern as before.");
    expect(meaningfulIntent('Now let me check whether the webview picks up the new message shape.'))
      .toBe('Now let me check whether the webview picks up the new message shape.');
  });

  it('strips reaction openers instead of dropping the whole block', () => {
    expect(meaningfulIntent('Perfect! Next I will rework the export pipeline to stream chunks.'))
      .toBe('Next I will rework the export pipeline to stream chunks.');
    expect(meaningfulIntent('Right — the LandingPage already has its own light-themed footer stacking below.'))
      .toBe('the LandingPage already has its own light-themed footer stacking below.');
  });

  it('rejects pure reactions and commentary', () => {
    expect(meaningfulIntent('Excellent!')).toBeNull();
    expect(meaningfulIntent('Great — that fixed it.')).toBeNull();
    expect(meaningfulIntent('Good progress on the refactor so far, this is coming together nicely.')).toBeNull();
  });

  it('rejects stubs under the minimum length', () => {
    expect(meaningfulIntent('Checking the config.')).toBeNull();
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

describe('extractIntentForTool — attribution', () => {
  const assistantLine = (blocks: unknown[], sidechain = false) =>
    JSON.stringify({ type: 'assistant', isSidechain: sidechain || undefined,
                     message: { content: blocks } });
  const text = (t: string) => ({ type: 'text', text: t });
  const tool = (name: string) => ({ type: 'tool_use', name, input: {} });

  function writeTranscript(sessionId: string, lines: string[]): void {
    const p = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'tb-intent-')), 'x.jsonl');
    fs.writeFileSync(p, lines.join('\n') + '\n');
    noteTranscript(sessionId, p);
  }

  it('attributes the preceding sentence to the FIRST call of a burst only', async () => {
    writeTranscript('s-burst', [
      assistantLine([
        text('Now let me read both config files to compare their formats.'),
        tool('Read'), tool('Read'),
      ]),
    ]);
    expect(await extractIntentForTool('s-burst', 1, 'Read'))
      .toBe('Now let me read both config files to compare their formats.');
    // the sibling gets null, not a duplicate subtitle
    expect(await extractIntentForTool('s-burst', 0, 'Read')).toBeNull();
  });

  it('aligns by tool name, unaffected by other tools around the target', async () => {
    writeTranscript('s-name', [
      assistantLine([text('Now let me check the failing test file for the broken assertion.'), tool('Read')]),
      assistantLine([tool('Bash'), tool('Bash')]),
    ]);
    // Two Bash calls follow, but counting Reads only still lands on the Read.
    expect(await extractIntentForTool('s-name', 0, 'Read'))
      .toBe('Now let me check the failing test file for the broken assertion.');
  });

  it('ignores sidechain (subagent) lines entirely', async () => {
    writeTranscript('s-side', [
      assistantLine([text('Now let me inspect the extension manifest for the icon path.'), tool('Read')]),
      assistantLine([text('Subagent chatter that would steal the attribution slot.'), tool('Read')], true),
    ]);
    expect(await extractIntentForTool('s-side', 0, 'Read'))
      .toBe('Now let me inspect the extension manifest for the icon path.');
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
