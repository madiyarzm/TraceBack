import { useMemo } from 'react';
import DiffViewer from './DiffViewer';
import { TimelineNode } from './TimelineCard';
import { annotateReviewFile, netKindOf, NetKind } from '../review';
import { verifyChanges, VerifyStatus } from '../fileChanges';
import { ReviewFile } from '../useSessionFeed';
import { CloseIcon } from './Icons';
import vscode from '../vscodeApi';

interface Props {
  /** null = requested, still loading. */
  files:     ReviewFile[] | null;
  nodes:     TimelineNode[];
  clickable: boolean;
  onClose:   () => void;
}

/**
 * The net-change review: one entry per touched file, showing the TRUE
 * baseline→now diff (not the sequence of edits), the agent's stated reasoning
 * for those edits, the failure that triggered them, and whether anything
 * verified the result. This is the "review code you didn't write" surface —
 * the session's net effect, made checkable.
 */
export default function ReviewPanel({ files, nodes, clickable, onClose }: Props) {
  const verified = useMemo(() => verifyChanges(nodes), [nodes]);

  return (
    <div style={{ padding: '14px 16px 28px', fontFamily: 'var(--tb-ui-font)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--tb-text)' }}>
            Net-change review
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--tb-text-muted)', marginTop: 3 }}>
            Each file's true before → after for this session, with the agent's reasoning.
            Intermediate edits are collapsed.
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <button
          onClick={onClose}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: 'var(--tb-surface)',
            border: '1px solid var(--tb-border-2)',
            borderRadius: 7,
            color: 'var(--tb-text)',
            fontSize: 13, fontWeight: 600, fontFamily: 'var(--tb-ui-font)',
            padding: '7px 13px', cursor: 'pointer',
          }}
        >
          <CloseIcon size={12} />
          Back to timeline
        </button>
      </div>

      {files === null && (
        <div style={{ fontSize: 13, color: 'var(--tb-text-dim)', padding: '18px 2px' }}>
          Reading files…
        </div>
      )}

      {files !== null && files.length === 0 && (
        <div style={{ fontSize: 13, color: 'var(--tb-text-dim)', padding: '18px 2px' }}>
          No file changes captured in this session — baselines are recorded from the
          moment TraceBack is running, as the agent makes its first edit to each file.
        </div>
      )}

      {files !== null && files.map((f) => (
        <FileReview
          key={f.path}
          file={f}
          nodes={nodes}
          verify={verified.get(f.path) ?? 'unverified'}
          clickable={clickable}
        />
      ))}
    </div>
  );
}

const KIND_COLOR: Record<NetKind, string> = {
  created:   '#3fb950',
  modified:  '#d29922',
  deleted:   '#f85149',
  unchanged: 'var(--tb-text-dim)',
};

const VERIFY_LABEL: Record<VerifyStatus, { text: string; color: string }> = {
  verified:   { text: '✓ verified',  color: '#3fb950' },
  failed:     { text: '✗ failing',   color: '#f85149' },
  unverified: { text: 'unverified',  color: 'var(--tb-text-dim)' },
};

function FileReview({ file, nodes, verify, clickable }: {
  file: ReviewFile; nodes: TimelineNode[]; verify: VerifyStatus; clickable: boolean;
}) {
  const kind = netKindOf(file.baseline, file.current);
  const ann  = useMemo(() => annotateReviewFile(nodes, file.path), [nodes, file.path]);
  const v    = VERIFY_LABEL[verify];
  const shortPath = file.path.split('/').filter(Boolean).slice(-3).join('/');

  return (
    <div style={{
      border: '1px solid var(--tb-border)',
      borderRadius: 10,
      background: 'var(--tb-surface)',
      overflow: 'hidden',
      marginBottom: 12,
    }}>
      {/* ── File header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '11px 14px',
        flexWrap: 'wrap',
      }}>
        <span style={{
          fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
          textTransform: 'uppercase',
          color: KIND_COLOR[kind],
          border: `1px solid ${kind === 'unchanged' ? 'var(--tb-border-2)' : `${KIND_COLOR[kind]}55`}`,
          borderRadius: 4, padding: '2px 7px', flexShrink: 0,
        }}>
          {kind}
        </span>
        <span
          onClick={() => clickable && vscode.postMessage({ type: 'open_file', filePath: file.path })}
          title={file.path}
          style={{
            fontSize: 13.5, fontWeight: 600,
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
            color: 'var(--tb-text)',
            cursor: clickable ? 'pointer' : 'default',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            minWidth: 0, flex: 1,
          }}
        >
          {shortPath}
        </span>
        <span style={{ fontSize: 11.5, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
          {ann.editCount} edit{ann.editCount === 1 ? '' : 's'}
          {ann.readCount > 0 ? ` · read ${ann.readCount}×` : ''}
        </span>
        <span style={{ fontSize: 11.5, fontWeight: 600, color: v.color, flexShrink: 0 }}>
          {v.text}
        </span>
      </div>

      {/* ── Why ── */}
      {(ann.triggeredBy || ann.intents.length > 0) && (
        <div style={{
          padding: '0 14px 11px',
          display: 'flex', flexDirection: 'column', gap: 5,
        }}>
          {ann.triggeredBy && (
            <div style={{ fontSize: 12.5, color: '#e8c877', lineHeight: 1.5 }}>
              After failing: <span style={{
                fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
              }}>{ann.triggeredBy}</span>
            </div>
          )}
          {ann.intents.map((intent, i) => (
            <div key={i} style={{
              fontSize: 12.5, color: 'var(--tb-text-muted)', lineHeight: 1.5,
              paddingLeft: 10,
              borderLeft: '2px solid var(--tb-border-2)',
            }}>
              {intent}
            </div>
          ))}
        </div>
      )}

      {/* ── Net diff ── */}
      {kind === 'unchanged' ? (
        <div style={{
          borderTop: '1px solid var(--tb-border)',
          padding: '10px 14px',
          fontSize: 12.5, color: 'var(--tb-text-dim)',
        }}>
          No net change — the session's edits to this file canceled out.
        </div>
      ) : (
        <div style={{ borderTop: '1px solid var(--tb-border)', padding: 10 }}>
          <DiffViewer
            oldText={file.baseline ?? ''}
            newText={file.current ?? ''}
            maxHeight={420}
          />
        </div>
      )}
    </div>
  );
}
