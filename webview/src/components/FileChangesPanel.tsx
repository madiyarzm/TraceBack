import { useMemo, useState } from 'react';
import { TimelineNode } from './TimelineCard';
import { computeFileChanges, summarizeChanges } from '../fileChanges';
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
  const changes = useMemo(() => computeFileChanges(nodes), [nodes]);

  if (changes.length === 0) return null;

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
          fontSize: 10, fontWeight: 600,
          letterSpacing: '0.06em', textTransform: 'uppercase',
          color: 'var(--tb-text-muted)',
        }}>
          Files changed
        </span>
        <span style={{ fontSize: 10.5, color: 'var(--tb-text)', flex: 1 }}>
          {summarizeChanges(changes)}
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
                fontSize: 8.5, fontWeight: 700,
                width: 52,
                letterSpacing: '0.05em', textTransform: 'uppercase',
                color: c.kind === 'created' ? '#3fb950' : '#d29922',
              }}>
                {c.kind}
              </span>
              <span style={{
                fontSize: 11,
                fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                color: 'var(--tb-text)',
                flex: 1, minWidth: 0,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                direction: 'rtl', textAlign: 'left',
              }}>
                {c.path}
              </span>
              {c.edits > 1 && (
                <span style={{ fontSize: 9, color: 'var(--tb-text-dim)', flexShrink: 0 }}>
                  ×{c.edits}
                </span>
              )}
              <span style={{
                flexShrink: 0,
                fontSize: 10,
                fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
              }}>
                <span style={{ color: '#3fb950' }}>+{c.linesAdded}</span>
                {' '}
                <span style={{ color: '#f85149' }}>−{c.linesRemoved}</span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
