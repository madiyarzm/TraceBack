/**
 * Pretty-print a shell command for the expanded Bash card.
 *
 * Agents love to chain a whole investigation into one line —
 * `cd x && echo "===" && grep -n "a\|b\|c" f && echo "" | head`. Rendered raw
 * with break-all, that's an unreadable wall of characters (the very thing that
 * makes CLI permission prompts inscrutable). We split the command at TOP-LEVEL
 * shell operators (&&, ||, |, ;) so each stage reads on its own line, while
 * leaving anything inside quotes untouched. Display-only: the copy button still
 * yields the original one-liner, so this never has to be valid shell.
 */

export interface CommandLine {
  /** The command text for this stage (operator NOT included). */
  text:     string;
  /** The connector that FOLLOWS this line (`&&`, `||`, `|`, `;`), or '' for the last. */
  operator: string;
}

const OPERATORS = ['&&', '||', '|', ';'] as const;

/**
 * Split a command into stages at unquoted &&, ||, |, ;. Single quotes, double
 * quotes, and backslash escapes are respected, so an operator inside a string
 * (e.g. `grep "a && b"`) never splits. `|&` and `||` are handled by matching
 * the two-char operators before the one-char `|`.
 */
export function splitCommand(command: string): CommandLine[] {
  const lines: CommandLine[] = [];
  let buf = '';
  let quote: '"' | "'" | null = null;

  for (let i = 0; i < command.length; i++) {
    const c = command[i];

    // Inside a quoted string: copy verbatim until the matching close quote.
    if (quote) {
      buf += c;
      if (c === '\\' && quote === '"' && i + 1 < command.length) {
        buf += command[++i];       // escaped char inside double quotes
      } else if (c === quote) {
        quote = null;
      }
      continue;
    }

    if (c === '"' || c === "'") { quote = c; buf += c; continue; }
    if (c === '\\' && i + 1 < command.length) { buf += c + command[++i]; continue; }

    // A `|` that is part of `||` is handled by the two-char check; a lone `|`
    // is a pipe. `&` alone (background) is left attached — rare and not a stage.
    const two = command.slice(i, i + 2);
    const op = OPERATORS.find((o) => (o.length === 2 ? two === o : c === o && two !== '||'));
    if (op) {
      lines.push({ text: buf.trim(), operator: op });
      buf = '';
      i += op.length - 1;
      continue;
    }

    buf += c;
  }

  if (buf.trim() || lines.length === 0) lines.push({ text: buf.trim(), operator: '' });
  // Drop empty stages that can arise from a trailing operator.
  return lines.filter((l) => l.text.length > 0);
}

/**
 * Render lines for display: the operator trails its stage (how a human writes a
 * multi-line pipeline), so the eye can scan the left edge for the actual
 * commands. Single-stage commands come back as one line, unchanged.
 */
export function formatShellCommand(command: string): string[] {
  const lines = splitCommand(command.trim());
  return lines.map((l, i) =>
    i < lines.length - 1 && l.operator ? `${l.text} ${l.operator}` : l.text,
  );
}
