import { useMemo, useState } from 'react';
import { TimelineNode } from './TimelineCard';
import {
  computeFileChanges, computeTouched, summarizeChanges, summarizeTouched,
  TouchedFile, verifyChanges, VerifyStatus,
} from '../fileChanges';
import { ChevronIcon, FolderIcon } from './Icons';
import vscode from '../vscodeApi';

interface Props {
  nodes: TimelineNode[];
  /** Archived sessions are read-only — file paths may no longer exist. */
  clickable?: boolean;
  /** Start expanded (panel) or collapsed (sidebar). */
  defaultOpen?: boolean;
}

/**
 * "What did the agent do to my codebase?" — the most concrete output of any
 * session, derived from Edit/Write nodes. Rows open the file in the editor.
 */
export default function FileChangesPanel({ nodes, clickable = true, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const [pickedMode, setMode] = useState<'changes' | 'touched'>('changes');
  const changes  = useMemo(() => computeFileChanges(nodes), [nodes]);
  const verified = useMemo(() => verifyChanges(nodes), [nodes]);
  const touched  = useMemo(() => computeTouched(nodes), [nodes]);
  // A read-only session has no changes yet — show the touched map instead.
  const mode = changes.length === 0 ? 'touched' : pickedMode;

  if (changes.length === 0 && touched.length === 0) return null;

  return (
    <div style={{
      margin: '4px 12px 12px',
      border: '1px solid var(--tb-border)',
      borderRadius: 5,
      background: 'var(--tb-surface)',
      fontFamily: 'var(--tb-ui-font)',
      overflow: 'hidden',
    }}>
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 10px',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <span style={{ display: 'flex', color: 'var(--tb-text-muted)' }}>
          <FolderIcon size={12} />
        </span>
        <span style={{
          fontSize: 11, fontWeight: 600,
          letterSpacing: '0.06em', textTransform: 'uppercase',
          color: 'var(--tb-text-muted)',
        }}>
          {mode === 'changes' ? 'Files changed' : 'Files touched'}
        </span>
        <span style={{ fontSize: 11.5, color: 'var(--tb-text)', flex: 1 }}>
          {mode === 'changes' ? summarizeChanges(changes) : summarizeTouched(touched)}
        </span>
        <span style={{
          color: 'var(--tb-text-dim)', display: 'flex',
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.15s',
        }}>
          <ChevronIcon size={11} />
        </span>
      </div>

      {open && (
        <div style={{ borderTop: '1px solid var(--tb-border)', animation: 'cardBodyIn 0.15s ease-out' }}>
          {/* ── Changes | Touched toggle ── */}
          <div style={{ display: 'flex', gap: 4, padding: '7px 10px 5px' }}>
            {(['changes', 'touched'] as const).map((m) => (
              <button
                key={m}
                onClick={(e) => { e.stopPropagation(); setMode(m); }}
                style={{
                  fontSize: 10.5, fontWeight: mode === m ? 700 : 500,
                  fontFamily: 'var(--tb-ui-font)',
                  letterSpacing: '0.04em', textTransform: 'uppercase',
                  padding: '2px 9px', cursor: 'pointer',
                  background: mode === m ? 'var(--tb-surface-2)' : 'transparent',
                  color: mode === m ? 'var(--tb-text)' : 'var(--tb-text-dim)',
                  border: `1px solid ${mode === m ? 'var(--tb-border-2)' : 'transparent'}`,
                  borderRadius: 4,
                }}
              >
                {m}
              </button>
            ))}
          </div>

          {mode === 'touched' ? (
            <TouchedTree touched={touched} clickable={clickable} />
          ) : (
          <>
          {changes.map((c) => (
            <div
              key={c.path}
              onClick={() => clickable && vscode.postMessage({ type: 'open_file', filePath: c.path })}
              title={clickable ? `Open ${c.path}` : c.path}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '4px 10px',
                cursor: clickable ? 'pointer' : 'default',
                borderBottom: '1px solid rgba(33,38,45,0.5)',
              }}
              onMouseEnter={(e) => { if (clickable) e.currentTarget.style.background = 'var(--tb-surface-2)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
            >
              <span style={{
                flexShrink: 0,
                fontSize: 9.5, fontWeight: 700,
                width: 52,
                letterSpacing: '0.05em', textTransform: 'uppercase',
                color: c.kind === 'created' ? '#3fb950' : '#d29922',
              }}>
                {c.kind}
              </span>
              <span style={{
                fontSize: 12,
                fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                color: 'var(--tb-text)',
                flex: 1, minWidth: 0,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                direction: 'rtl', textAlign: 'left',
              }}>
                {c.path}
              </span>
              {c.edits > 1 && (
                <span style={{ fontSize: 10, color: 'var(--tb-text-dim)', flexShrink: 0 }}>
                  ×{c.edits}
                </span>
              )}
              <VerifyBadge status={verified.get(c.path) ?? 'unverified'} />
              <span style={{
                flexShrink: 0,
                fontSize: 11,
                fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
              }}>
                <span style={{ color: '#3fb950' }}>+{c.linesAdded}</span>
                {' '}
                <span style={{ color: '#f85149' }}>−{c.linesRemoved}</span>
              </span>
            </div>
          ))}
          </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Touched tree ──────────────────────────────────────────────────────────────

const TOUCH_COLOR: Record<TouchedFile['kind'], string> = {
  read:     '#58a6ff',
  modified: '#d29922',
  created:  '#3fb950',
};

interface TreeRow {
  depth:  number;
  name:   string;
  path?:  string;                   // set for files (clickable)
  kind?:  TouchedFile['kind'];
}

/** Flatten touched paths into indented rows under their common root, folding
 *  single-child directory chains ("src/components/…") GitHub-style. */
function buildTreeRows(touched: TouchedFile[]): TreeRow[] {
  if (touched.length === 0) return [];

  interface Dir { dirs: Map<string, Dir>; files: { name: string; t: TouchedFile }[] }
  const mkDir = (): Dir => ({ dirs: new Map(), files: [] });

  // Common root: shared leading segments across all paths.
  const split = touched.map((t) => t.path.split('/').filter(Boolean));
  let common = split[0].slice(0, -1);
  for (const segs of split) {
    let i = 0;
    while (i < common.length && i < segs.length - 1 && common[i] === segs[i]) i++;
    common = common.slice(0, i);
  }

  const root = mkDir();
  touched.forEach((t, idx) => {
    const segs = split[idx].slice(common.length);
    let dir = root;
    for (const seg of segs.slice(0, -1)) {
      if (!dir.dirs.has(seg)) dir.dirs.set(seg, mkDir());
      dir = dir.dirs.get(seg)!;
    }
    dir.files.push({ name: segs[segs.length - 1] ?? t.path, t });
  });

  const rows: TreeRow[] = [];
  function walk(dir: Dir, depth: number, prefix: string): void {
    for (const [name, sub] of Array.from(dir.dirs.entries()).sort(([a], [b]) => a.localeCompare(b))) {
      // Fold chains: dir with exactly one subdir and no files merges downward.
      let label = prefix + name;
      let cur = sub;
      while (cur.files.length === 0 && cur.dirs.size === 1) {
        const [childName, child] = cur.dirs.entries().next().value as [string, Dir];
        label += '/' + childName;
        cur = child;
      }
      rows.push({ depth, name: label + '/' });
      walk(cur, depth + 1, '');
    }
    for (const f of dir.files.sort((a, b) => a.name.localeCompare(b.name))) {
      rows.push({ depth, name: f.name, path: f.t.path, kind: f.t.kind });
    }
  }
  walk(root, 0, '');
  return rows;
}

function TouchedTree({ touched, clickable }: { touched: TouchedFile[]; clickable: boolean }) {
  const rows = useMemo(() => buildTreeRows(touched), [touched]);

  return (
    <div style={{ padding: '2px 0 6px' }}>
      {rows.map((row, i) => (
        <div
          key={i}
          onClick={() => clickable && row.path &&
            vscode.postMessage({ type: 'open_file', filePath: row.path })}
          title={row.path}
          style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: `2px 10px 2px ${10 + row.depth * 14}px`,
            cursor: clickable && row.path ? 'pointer' : 'default',
          }}
          onMouseEnter={(e) => { if (clickable && row.path) e.currentTarget.style.background = 'var(--tb-surface-2)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
        >
          {row.kind ? (
            <span style={{
              width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
              background: TOUCH_COLOR[row.kind],
              opacity: row.kind === 'read' ? 0.55 : 1,
            }} />
          ) : (
            <span style={{ width: 7, flexShrink: 0 }} />
          )}
          <span style={{
            fontSize: 12,
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
            color: row.kind ? 'var(--tb-text)' : 'var(--tb-text-dim)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {row.name}
          </span>
          {row.kind && row.kind !== 'read' && (
            <span style={{
              fontSize: 9.5, fontWeight: 700, letterSpacing: '0.05em',
              textTransform: 'uppercase',
              color: TOUCH_COLOR[row.kind], flexShrink: 0, marginLeft: 'auto',
            }}>
              {row.kind}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

const VERIFY_STYLE: Record<VerifyStatus, { label: string; color: string; title: string }> = {
  verified: {
    label: '✓ verified', color: '#3fb950',
    title: 'A check command ran after the last edit to this file and succeeded',
  },
  failed: {
    label: '✗ failing', color: '#f85149',
    title: 'The latest check after this file\'s last edit failed',
  },
  unverified: {
    label: 'unverified', color: 'var(--tb-text-dim)',
    title: 'No test/build/lint command ran after the last edit to this file',
  },
};

function VerifyBadge({ status }: { status: VerifyStatus }) {
  const s = VERIFY_STYLE[status];
  return (
    <span
      title={s.title}
      style={{
        flexShrink: 0,
        fontSize: 9.5, fontWeight: 600, letterSpacing: '0.04em',
        color: s.color,
        border: `1px solid ${status === 'unverified' ? 'var(--tb-border-2)' : `${s.color}55`}`,
        borderRadius: 4, padding: '1px 5px',
        whiteSpace: 'nowrap',
      }}
    >
      {s.label}
    </span>
  );
}
