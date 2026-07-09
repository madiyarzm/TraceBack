import { describe, expect, it } from 'vitest';
import { classifyPrompt, todoInstructionFor } from '../src/promptHeuristics';

describe('classifyPrompt', () => {
  it('injects nothing for continuations and short prompts', () => {
    expect(classifyPrompt('go ahead')).toBe('none');
    expect(classifyPrompt('yes')).toBe('none');
    expect(classifyPrompt('fix this')).toBe('none');
    expect(classifyPrompt('try again with the other approach please')).toBe('none');
    expect(classifyPrompt('/compact')).toBe('none');
  });

  it('injects nothing for a longer prompt that still opens as a continuation', () => {
    expect(classifyPrompt(
      'continue where you left off and make sure the tests still pass afterwards'
    )).toBe('none');
  });

  it('soft nudge for medium single-ask prompts', () => {
    expect(classifyPrompt(
      'rename the SessionOdometer component to SessionHeader and update all the imports'
    )).toBe('soft');
  });

  it('firm ask for bulleted or numbered multi-part prompts', () => {
    expect(classifyPrompt(
      'please do the following:\n- add a dark mode toggle\n- persist it to settings\n- update the docs'
    )).toBe('firm');
    expect(classifyPrompt(
      'two things today.\n1. fix the flaky login test\n2. add retries to the api client'
    )).toBe('firm');
  });

  it('firm ask for prompts with many sentences', () => {
    expect(classifyPrompt(
      'The export flow is broken. It writes an empty file. I think the stream closes early. ' +
      'Please investigate and fix it. Then add a regression test.'
    )).toBe('firm');
  });

  it('firm ask for very long prompts regardless of structure', () => {
    expect(classifyPrompt('refactor the entire data layer ' + 'x'.repeat(400))).toBe('firm');
  });
});

describe('todoInstructionFor', () => {
  it('returns null when nothing should be injected', () => {
    expect(todoInstructionFor('go ahead')).toBeNull();
  });

  it('firm instruction asks for in_progress marking', () => {
    const firm = todoInstructionFor(
      'here is what i need done today:\n' +
      '- first migrate the config loader to the new format\n' +
      '- then update every call site that reads from it\n' +
      '- finally add a regression test for the old format'
    );
    expect(firm).toMatch(/in_progress/);
    expect(firm).toMatch(/completed/);
  });

  it('soft and firm both ask for a one-sentence narration before tool calls', () => {
    const soft = todoInstructionFor(
      'please refactor the session parser so it can handle multiple formats'
    );
    const firm = todoInstructionFor(
      'first do this thing. then do that thing. after that verify it. finally ship it.'
    );
    expect(soft).toMatch(/before each tool call/i);
    expect(firm).toMatch(/before each tool call/i);
  });
});
