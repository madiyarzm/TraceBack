import { useMemo, useState } from 'react';
import { TimelineNode, formatDuration } from './TimelineCard';
import { computePhases, PHASE_COLOR, PHASE_LABEL } from '../phases';
import { ChevronIcon, TargetIcon } from './Icons';

export interface PlanItemUI {
  content:     string;
  status:      'pending' | 'in_progress' | 'completed';
  activeForm?: string;
}

export interface SessionPlanUI {
  items:     PlanItemUI[];
  updatedAt: number;
}

interface Props {
  plan?:  SessionPlanUI;
  nodes:  TimelineNode[];
  isLive: boolean;
  /** Sidebar gets a tighter layout. */
  slim?:  boolean;
}

interface ObjectiveRollup {
  actions:    number;
  durationMs: number;
}

/**
 * Mission control: the agent's own plan (from TodoWrite calls) rendered as a
 * live objective + progress + the session's phase "shape". This is the
 * structural answer to "where is this heading" — hierarchy over chronology,
 * which the transcript can't show.
 */
export default function ObjectiveHeader({ plan, nodes, isLive, slim = false }: Props) {
  const [open, setOpen] = useState(false);

  const segments = useMemo(() => computePhases(nodes), [nodes]);

  // Effort attribution: actions + wall time grouped by the objective that was
  // in progress when each call ran.
  const rollups = useMemo(() => {
    const map = new Map<string, ObjectiveRollup>();
    for (const n of nodes) {
      if (n.toolName.startsWith('__') || n.isPlanUpdate || !n.objective) continue;
      const r = map.get(n.objective) ?? { actions: 0, durationMs: 0 };
      r.actions    += n.count;
      r.durationMs += n.durationMs ?? 0;
      map.set(n.objective, r);
    }
    return map;
  }, [nodes]);

  if (!plan || plan.items.length === 0) return null;

  const done    = plan.items.filter((i) => i.status === 'completed').length;
  const total   = plan.items.length;
  const current = plan.items.find((i) => i.status === 'in_progress');
  const allDone = done === total;

  const headline = current
    ? (current.activeForm ?? current.content)
    : allDone ? 'All tasks complete' : 'Between tasks';

  const totalWeight = segments.reduce((s, seg) => s + seg.count, 0);

  return (
    <div style={{
      borderBottom: '1px solid var(--tb-border)',
      background: 'var(--tb-surface)',
      fontFamily: 'var(--tb-ui-font)',
    }}>
      {/* ── Row 1: current objective + progress ── */}
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: slim ? '6px 12px 4px' : '8px 16px 5px',
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <span style={{
          display: 'flex', flexShrink: 0,
          color: allDone ? 'var(--tb-green)' : 'var(--tb-blue)',
          ...(isLive && current ? { animation: 'pendingPulse 2s ease-in-out infinite' } : {}),
        }}>
          <TargetIcon size={slim ? 12 : 13} />
        </span>

        <span style={{
          fontSize: slim ? 11 : 12, fontWeight: 600,
          color: 'var(--tb-text)',
          flex: 1, minWidth: 0,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {headline}
        </span>

        <span style={{
          fontSize: slim ? 9.5 : 10,
          fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
          fontVariantNumeric: 'tabular-nums',
          color: allDone ? 'var(--tb-green)' : 'var(--tb-text-muted)',
          flexShrink: 0,
        }}>
          {done}/{total}
        </span>

        <span style={{
          color: 'var(--tb-text-dim)', display: 'flex', flexShrink: 0,
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.15s',
        }}>
          <ChevronIcon size={10} />
        </span>
      </div>

      {/* ── Progress bar ── */}
      <div style={{ padding: slim ? '0 12px 6px' : '0 16px 7px' }}>
        <div style={{
          height: 3, borderRadius: 2,
          background: 'var(--tb-surface-2)',
          overflow: 'hidden',
        }}>
          <div style={{
            height: '100%',
            width: `${total ? (done / total) * 100 : 0}%`,
            background: allDone ? 'var(--tb-green)' : 'var(--tb-blue)',
            borderRadius: 2,
            transition: 'width 0.4s ease',
          }} />
        </div>
      </div>

      {/* ── Phase ribbon: the session's shape ── */}
      {totalWeight > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: slim ? '0 12px 7px' : '0 16px 8px',
        }}>
          <div style={{
            display: 'flex', flex: 1, height: 5,
            borderRadius: 3, overflow: 'hidden', gap: 1,
          }}>
            {segments.map((seg, i) => (
              <div
                key={i}
                title={`${PHASE_LABEL[seg.phase]} · ${seg.count} action${seg.count === 1 ? '' : 's'}`}
                style={{
                  width: `${(seg.count / totalWeight) * 100}%`,
                  minWidth: 3,
                  background: PHASE_COLOR[seg.phase],
                  opacity: 0.85,
                }}
              />
            ))}
          </div>
          {!slim && (
            <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
              {(['explore', 'build', 'verify'] as const).map((p) => (
                <span key={p} style={{
                  display: 'flex', alignItems: 'center', gap: 3,
                  fontSize: 9.5, letterSpacing: '0.05em', textTransform: 'uppercase',
                  color: 'var(--tb-text-dim)',
                }}>
                  <span style={{
                    width: 5, height: 5, borderRadius: '50%',
                    background: PHASE_COLOR[p], opacity: 0.85,
                  }} />
                  {PHASE_LABEL[p]}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Expanded checklist with per-objective effort ── */}
      {open && (
        <div style={{
          borderTop: '1px solid var(--tb-border)',
          padding: slim ? '5px 12px 8px' : '6px 16px 10px',
          animation: 'cardBodyIn 0.15s ease-out',
        }}>
          {plan.items.map((item, i) => {
            const key    = item.activeForm ?? item.content;
            const rollup = rollups.get(key) ?? rollups.get(item.content);
            const active = item.status === 'in_progress';
            return (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '3px 0',
              }}>
                <span style={{
                  width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                  background:
                    item.status === 'completed' ? 'var(--tb-green)'
                    : active ? 'var(--tb-blue)'
                    : 'transparent',
                  border: item.status === 'pending' ? '1px solid var(--tb-border-2)' : 'none',
                  ...(active && isLive ? { animation: 'pendingPulse 1.6s ease-in-out infinite' } : {}),
                }} />
                <span style={{
                  fontSize: 12, flex: 1, minWidth: 0,
                  color: item.status === 'completed' ? 'var(--tb-text-muted)'
                       : active ? 'var(--tb-text)' : 'var(--tb-text-muted)',
                  fontWeight: active ? 600 : 400,
                  textDecoration: item.status === 'completed' ? 'line-through' : 'none',
                  textDecorationColor: 'var(--tb-text-dim)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {item.content}
                </span>
                {rollup && (
                  <span style={{
                    fontSize: 10, flexShrink: 0,
                    fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                    fontVariantNumeric: 'tabular-nums',
                    color: 'var(--tb-text-dim)',
                  }}>
                    {rollup.actions} act{formatDuration(rollup.durationMs) ? ` · ${formatDuration(rollup.durationMs)}` : ''}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
