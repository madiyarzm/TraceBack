import { useMemo } from 'react';
import { TimelineNode, formatDuration, STATUS_COLOR } from './TimelineCard';
import { SessionPlanUI } from './ObjectiveHeader';
import { AnomalyRecordUI } from '../useSessionFeed';
import { computePhases, PHASE_COLOR, PHASE_LABEL } from '../phases';
import { AlertIcon, CheckIcon } from './Icons';

interface Props {
  nodes:   TimelineNode[];
  plan?:   SessionPlanUI;
  history?: AnomalyRecordUI[];
  isLive:  boolean;
  /** Wide screens: sections flow into columns instead of one stack. */
  grid?:   boolean;
}

/**
 * The zoomed-out view: the whole session in one screen. Objective chapters
 * with effort rollups when the agent has a plan, phase segments otherwise;
 * errors and anomalies always listed individually — they are the signal at
 * every zoom level.
 */
export default function SessionMap({ nodes, plan, history = [], isLive, grid = false }: Props) {
  const real = useMemo(
    () => nodes.filter((n) => !n.toolName.startsWith('__') && !n.isPlanUpdate),
    [nodes],
  );
  const segments = useMemo(() => computePhases(nodes), [nodes]);

  // Effort + error rollups per objective
  const rollups = useMemo(() => {
    const map = new Map<string, { actions: number; durationMs: number; errors: number }>();
    for (const n of real) {
      if (!n.objective) continue;
      const r = map.get(n.objective) ?? { actions: 0, durationMs: 0, errors: 0 };
      r.actions    += n.count;
      r.durationMs += n.durationMs ?? 0;
      if (n.status === 'error') r.errors++;
      map.set(n.objective, r);
    }
    return map;
  }, [real]);

  const errorNodes = real.filter((n) => n.status === 'error');

  return (
    <div style={grid ? {
      padding: '14px 16px 24px',
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
      gap: 18,
      alignItems: 'start',
      fontFamily: 'var(--tb-ui-font)',
    } : {
      padding: '12px 14px 24px',
      display: 'flex', flexDirection: 'column', gap: 16,
      fontFamily: 'var(--tb-ui-font)',
    }}>
      {/* ── Objective chapters (or phase segments when there's no plan) ── */}
      {plan && plan.items.length > 0 ? (
        <Section label="Plan">
          {plan.items.map((item, i) => {
            const key    = item.activeForm ?? item.content;
            const rollup = rollups.get(key) ?? rollups.get(item.content);
            const active = item.status === 'in_progress';
            return (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 9,
                padding: '5px 10px',
                borderRadius: 5,
                background: active ? 'var(--tb-surface)' : 'transparent',
                border: `1px solid ${active ? 'var(--tb-border)' : 'transparent'}`,
              }}>
                <span style={{
                  width: 14, height: 14, borderRadius: '50%', flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background:
                    item.status === 'completed' ? 'rgba(63,185,80,0.15)'
                    : active ? 'rgba(88,166,255,0.15)' : 'transparent',
                  border: item.status === 'pending' ? '1px solid var(--tb-border-2)' : 'none',
                  color: item.status === 'completed' ? 'var(--tb-green)' : 'var(--tb-blue)',
                  ...(active && isLive ? { animation: 'pendingPulse 1.6s ease-in-out infinite' } : {}),
                }}>
                  {item.status === 'completed' && <CheckIcon size={9} />}
                  {active && <span style={{
                    width: 5, height: 5, borderRadius: '50%', background: 'var(--tb-blue)',
                  }} />}
                </span>
                <span style={{
                  fontSize: 12, flex: 1, minWidth: 0,
                  fontWeight: active ? 600 : 400,
                  color: item.status === 'pending' ? 'var(--tb-text-muted)' : 'var(--tb-text)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {item.content}
                </span>
                {rollup && rollup.errors > 0 && (
                  <span style={{ fontSize: 9.5, color: '#f85149', flexShrink: 0, fontWeight: 600 }}>
                    {rollup.errors} err
                  </span>
                )}
                {rollup && (
                  <span style={{
                    fontSize: 10, flexShrink: 0,
                    fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                    color: 'var(--tb-text-dim)',
                  }}>
                    {rollup.actions} act{formatDuration(rollup.durationMs) ? ` · ${formatDuration(rollup.durationMs)}` : ''}
                  </span>
                )}
              </div>
            );
          })}
        </Section>
      ) : segments.length > 0 && (
        <Section label="Session shape">
          {segments.map((seg, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 9,
              padding: '4px 10px',
            }}>
              <span style={{
                width: 7, height: 7, borderRadius: 2, flexShrink: 0,
                background: PHASE_COLOR[seg.phase],
              }} />
              <span style={{ fontSize: 12, color: 'var(--tb-text)', flex: 1, textTransform: 'capitalize' }}>
                {PHASE_LABEL[seg.phase]}
              </span>
              <span style={{
                fontSize: 10,
                fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                color: 'var(--tb-text-dim)',
              }}>
                {seg.count} action{seg.count === 1 ? '' : 's'}
              </span>
            </div>
          ))}
        </Section>
      )}

      {/* ── Errors: always individually visible at map scale ── */}
      {errorNodes.length > 0 && (
        <Section label={`Errors · ${errorNodes.length}`}>
          {errorNodes.map((n) => (
            <div key={n.id} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '4px 10px',
              borderRadius: 5,
              background: 'rgba(248,81,73,0.05)',
              border: '1px solid rgba(248,81,73,0.25)',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                background: STATUS_COLOR.error,
              }} />
              <span style={{
                fontSize: 9.5, fontWeight: 600, flexShrink: 0,
                letterSpacing: '0.05em', textTransform: 'uppercase',
                color: 'var(--tb-text-muted)',
              }}>
                {n.toolName}
              </span>
              <span style={{
                fontSize: 11.5, color: '#ffa198', flex: 1, minWidth: 0,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {n.label}
              </span>
              {formatDuration(n.durationMs) && (
                <span style={{
                  fontSize: 10, color: 'var(--tb-text-dim)', flexShrink: 0,
                  fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                }}>
                  {formatDuration(n.durationMs)}
                </span>
              )}
            </div>
          ))}
        </Section>
      )}

      {/* ── Anomaly history ── */}
      {history.length > 0 && (
        <Section label={`Anomalies · ${history.length}`}>
          {history.map((rec, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '4px 10px',
              color: '#d29922',
            }}>
              <AlertIcon size={11} />
              <span style={{
                fontSize: 11.5, flex: 1, minWidth: 0,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {rec.type === 'repeater' ? 'Déjà vu — ' : ''}{rec.reason}
              </span>
              <span style={{ fontSize: 9.5, color: 'var(--tb-text-dim)', flexShrink: 0 }}>
                {new Date(rec.detectedAt).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </Section>
      )}

      {real.length === 0 && (
        <span style={{ fontSize: 11, color: 'var(--tb-text-dim)', padding: '0 10px' }}>
          Nothing recorded yet.
        </span>
      )}
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{
        fontSize: 9, fontWeight: 700,
        letterSpacing: '0.1em', textTransform: 'uppercase',
        color: 'var(--tb-text-dim)',
        padding: '0 10px 5px',
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {children}
      </div>
    </div>
  );
}
