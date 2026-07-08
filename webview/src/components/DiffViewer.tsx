import { memo } from 'react';

/**
 * Git-style inline diff for file-modification tools.
 *
 * Uses a hand-rolled LCS (longest-common-subsequence) line diff instead of a
 * library: Edit old/new strings are small (typically < 100 lines), so the
 * O(n*m) DP is instant, and it keeps the webview bundle lean. A cell cap
 * guards against pathological inputs by falling back to before/after blocks.
 */

type DiffLine = { type: 'same' | 'add' | 'del'; text: string };

const MAX_DP_CELLS = 200_000;

function diffLines(oldText: string, newText: string): DiffLine[] | null {
  const a = oldText.split('\n');
  const b = newText.split('\n');

  if (a.length * b.length > MAX_DP_CELLS) return null; // caller falls back

  // LCS table
  const dp: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array<number>(b.length + 1).fill(0)
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  // Walk the table to emit del/add/same in order
  const out: DiffLine[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      out.push({ type: 'same', text: a[i] });
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ type: 'del', text: a[i] });
      i++;
    } else {
      out.push({ type: 'add', text: b[j] });
      j++;
    }
  }
  while (i < a.length) out.push({ type: 'del', text: a[i++] });
  while (j < b.length) out.push({ type: 'add', text: b[j++] });
  return out;
}

const LINE_STYLE: Record<DiffLine['type'], React.CSSProperties> = {
  same: { color: '#8b949e' },
  add:  { color: '#aff5b4', background: 'rgba(46,160,67,0.18)' },
  del:  { color: '#ffdcd7', background: 'rgba(248,81,73,0.18)' },
};

const PREFIX: Record<DiffLine['type'], string> = { same: ' ', add: '+', del: '-' };

const CONTAINER: React.CSSProperties = {
  background:   '#0d1117',
  border:       '1px solid var(--tb-border)',
  borderRadius: 4,
  fontFamily:   'var(--tb-mono-font, ui-monospace, monospace)',
  fontSize: 11.5,
  lineHeight:   1.5,
  overflow:     'auto',
};

interface Props {
  oldText:   string;
  newText:   string;
  filePath?: string;
  /** Card context uses the compact default; the review panel goes taller. */
  maxHeight?: number;
}

function DiffViewer({ oldText, newText, filePath, maxHeight = 260 }: Props) {
  const lines = diffLines(oldText, newText);

  return (
    <div style={{ ...CONTAINER, maxHeight }}>
      {filePath && (
        <div style={{
          padding: '4px 10px',
          borderBottom: '1px solid var(--tb-border)',
          color: '#8b949e',
          fontSize: 11,
          position: 'sticky', top: 0,
          background: '#0d1117',
        }}>
          {filePath}
        </div>
      )}

      {lines ? (
        <div style={{ padding: '4px 0' }}>
          {lines.map((line, idx) => (
            <div key={idx} style={{
              ...LINE_STYLE[line.type],
              padding: '0 10px',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}>
              <span style={{ userSelect: 'none', opacity: 0.7 }}>{PREFIX[line.type]} </span>
              {line.text || ' '}
            </div>
          ))}
        </div>
      ) : (
        // Fallback for inputs too large to diff: plain before/after blocks
        <div style={{ padding: '4px 0' }}>
          {oldText && (
            <div style={{ ...LINE_STYLE.del, padding: '2px 10px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {`- removed (${oldText.split('\n').length} lines)\n${oldText.slice(0, 2000)}${oldText.length > 2000 ? '\n…' : ''}`}
            </div>
          )}
          {newText && (
            <div style={{ ...LINE_STYLE.add, padding: '2px 10px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {`+ added (${newText.split('\n').length} lines)\n${newText.slice(0, 2000)}${newText.length > 2000 ? '\n…' : ''}`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default memo(DiffViewer);
