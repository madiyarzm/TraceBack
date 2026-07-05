import { useState } from 'react';
import TimelineCard, { TimelineNode, formatDuration } from './TimelineCard';
import { ChapterTaskGroup, pendingPlanItems, PromptChapter } from '../chapters';
import { computeMetrics, formatTokens } from '../metrics';
import { ChevronIcon } from './Icons';

interface Props {
  chapter:     PromptChapter;
  isLive:      boolean;
  /** Real session tokens shown only for the live (last) chapter. */
  realTokens?: number;
  isLast:      boolean;
  expandedId:  string | null;
  onToggle:    (id: string) => void;
}

/**
 * The main-panel chapter view: full prompt text, task progress for this
 * prompt, then the agent's work grouped into collapsible task blocks. Done
 * tasks collapse to a one-line file summary; the active task shows its
 * action cards; pending plan items render locked.
 */
export default function PromptChapterView({
  chapter, isLive, realTokens, isLast, expandedId, onToggle,
}: Props) {
  const done  = chapter.plan.filter((p) => p.status === 'completed').length;
  const total = chapter.plan.length;
  const m     = computeMetrics(chapter.nodes);
  const tokens = isLast && realTokens !== undefined ? realTokens : m.estTokens;
  const pending = pendingPlanItems(chapter);

  return (
    <div style={{ fontFamily: 'var(--tb-ui-font)' }}>
      {/* ── Chapter header ── */}
      <div style={{
        padding: '14px 16px 12px',
        borderBottom: '1px solid var(--tb-border)',
        background: 'var(--tb-surface)',
      }}>
        <div style={{
          fontSize: 14, fontWeight: 600, lineHeight: 1.5,
          color: 'var(--tb-text)',
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        }}>
          “{chapter.text}”
        </div>

        {total > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 10 }}>
            <div style={{
              flex: 1, height: 4, borderRadius: 2,
              background: 'var(--tb-surface-2)', overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                width: `${(done / total) * 100}%`,
                background: done === total ? 'var(--tb-green)' : 'var(--tb-blue)',
                borderRadius: 2,
                transition: 'width 0.4s ease',
              }} />
            </div>
            <span style={{
              fontSize: 10.5, flexShrink: 0,
              fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
              color: done === total ? 'var(--tb-green)' : 'var(--tb-text-muted)',
            }}>
              {done} of {total} tasks
            </span>
          </div>
        )}

        <div style={{ display: 'flex', gap: 18, marginTop: 10 }}>
          <HeaderStat value={String(chapter.actionCount)} label="actions" />
          <HeaderStat
            value={String(chapter.errorCount)} label="errors"
            color={chapter.errorCount > 0 ? '#f85149' : undefined}
          />
          <HeaderStat value={formatDuration(chapter.durationMs) ?? '—'} label="duration" />
          <HeaderStat
            value={formatTokens(tokens)}
            label={isLast && realTokens !== undefined ? 'tokens' : '≈ tokens'}
          />
        </div>
      </div>

      {/* ── Task blocks ── */}
      <div style={{ padding: '12px 14px 24px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {chapter.nodes.length === 0 && (
          <div style={{ fontSize: 11, color: 'var(--tb-text-dim)', padding: '8px 2px' }}>
            {isLive ? 'Waiting — no actions for this prompt yet.' : 'No actions recorded for this prompt.'}
          </div>
        )}

        {chapter.taskGroups.map((g, i) => (
          <TaskBlock
            key={i}
            group={g}
            isLive={isLive}
            expandedId={expandedId}
            onToggle={onToggle}
          />
        ))}

        {pending.map((p, i) => (
          <div key={`pending-${i}`} style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '8px 12px',
            borderRadius: 5,
            border: '1px dashed var(--tb-border)',
            color: 'var(--tb-text-dim)',
            fontSize: 11,
            userSelect: 'none',
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
              border: '1px solid var(--tb-border-2)',
            }} />
            <span style={{
              flex: 1, minWidth: 0,
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>{p.content}</span>
            <span style={{ fontSize: 9, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              pending
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TaskBlock({ group, isLive, expandedId, onToggle }: {
  group:      ChapterTaskGroup;
  isLive:     boolean;
  expandedId: string | null;
  onToggle:   (id: string) => void;
}) {
  const isDone   = group.status === 'completed' && group.errorCount === 0;
  const isActive = group.status === 'in_progress';
  const [open, setOpen] = useState(!isDone);

  // Ungrouped actions (no objective recorded) — render cards directly.
  if (!group.objective) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {group.nodes.map((n) => (
          <ActionCardRow key={n.id} node={n} expandedId={expandedId} onToggle={onToggle} />
        ))}
      </div>
    );
  }

  const dotColor =
    group.errorCount > 0 ? '#f85149'
    : isDone ? 'var(--tb-green)'
    : isActive ? 'var(--tb-blue)'
    : 'var(--tb-text-dim)';

  return (
    <div style={{
      border: `1px solid ${isActive ? 'var(--tb-border-2)' : 'var(--tb-border)'}`,
      borderRadius: 6,
      background: 'var(--tb-surface)',
      overflow: 'hidden',
    }}>
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 9,
          padding: '9px 12px',
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <span style={{
          width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
          background: dotColor,
          ...(isActive && isLive ? { animation: 'pendingPulse 1.6s ease-in-out infinite' } : {}),
        }} />
        <span style={{
          fontSize: 12, fontWeight: isActive ? 600 : 500,
          color: isDone ? 'var(--tb-text-muted)' : 'var(--tb-text)',
          flex: 1, minWidth: 0,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {group.objective}
        </span>
        {isActive && (
          <span style={{ fontSize: 10, color: 'var(--tb-blue)', flexShrink: 0 }}>
            active · {group.nodes.length} action{group.nodes.length === 1 ? '' : 's'}
          </span>
        )}
        {!isActive && (
          <span style={{ fontSize: 10, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
            {group.nodes.length} action{group.nodes.length === 1 ? '' : 's'}
          </span>
        )}
        <span style={{
          color: 'var(--tb-text-dim)', display: 'flex', flexShrink: 0,
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.15s',
        }}>
          <ChevronIcon size={11} />
        </span>
      </div>

      {/* Done tasks collapse to a one-line file summary */}
      {!open && group.files.length > 0 && (
        <div style={{
          padding: '0 12px 8px 29px',
          fontSize: 10, color: 'var(--tb-text-dim)',
          fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {group.files.join(' · ')}
        </div>
      )}

      {open && (
        <div style={{
          borderTop: '1px solid var(--tb-border)',
          padding: '10px 12px',
          display: 'flex', flexDirection: 'column', gap: 8,
          animation: 'cardBodyIn 0.15s ease-out',
        }}>
          {group.nodes.map((n) => (
            <ActionCardRow key={n.id} node={n} expandedId={expandedId} onToggle={onToggle} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Action card — the shared TimelineCard, whose leading dot doubles as the
 *  status indicator inside a task block. */
function ActionCardRow({ node, expandedId, onToggle }: {
  node: TimelineNode; expandedId: string | null; onToggle: (id: string) => void;
}) {
  return (
    <TimelineCard
      node={node}
      expanded={expandedId === node.id}
      onToggle={onToggle}
    />
  );
}

function HeaderStat({ value, label, color }: { value: string; label: string; color?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{
        fontSize: 13, fontWeight: 600,
        fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
        color: color ?? 'var(--tb-text)',
        lineHeight: 1.1,
      }}>{value}</span>
      <span style={{
        fontSize: 8.5, fontWeight: 600,
        color: 'var(--tb-text-dim)',
        letterSpacing: '0.07em', textTransform: 'uppercase',
      }}>{label}</span>
    </div>
  );
}
