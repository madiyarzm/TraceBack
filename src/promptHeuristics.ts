/**
 * Prompt-substance heuristics for the UserPromptSubmit injection.
 *
 * The observer should distort the observed as little as possible: a todo
 * nudge on "go ahead" causes over-planning noise, while a six-part feature
 * request deserves a firm ask — including in_progress marking, which is what
 * actually makes plan attribution work in the chapter view.
 */

export type TodoNudge = 'none' | 'soft' | 'firm';

const CONTINUATION = new RegExp(
  '^(yes|yeah|yep|ok(ay)?|sure|go ahead|continue|proceed|resume|keep going|' +
  'do it|try again|retry|fix (it|this|that)|sounds good|lgtm|approved?|next|' +
  'and now|what about)\\b', 'i'
);

export function classifyPrompt(prompt: string): TodoNudge {
  const p = prompt.trim();

  // Client commands and trivial acknowledgements: inject nothing.
  if (p.startsWith('/')) return 'none';
  if (p.length < 60) return 'none';
  if (p.length < 140 && CONTINUATION.test(p)) return 'none';

  // Substance signals: explicit lists, several sentences, or sheer length.
  const listItems = (p.match(/^\s*(?:[-*•]|\d+[.)])\s+/gm) ?? []).length;
  const sentences = (p.match(/[.!?](?:\s|$)/g) ?? []).length;
  if (p.length >= 400 || listItems >= 2 || sentences >= 4) return 'firm';

  return 'soft';
}

export function todoInstructionFor(prompt: string): string | null {
  switch (classifyPrompt(prompt)) {
    case 'none':
      return null;
    case 'soft':
      return 'If this task has multiple steps, consider creating a todo list before starting.';
    case 'firm':
      return (
        'This looks like a multi-step task. Break it into a todo list before ' +
        'starting, and keep it live: mark each task in_progress when you begin ' +
        'it and completed when you finish it.'
      );
  }
}
