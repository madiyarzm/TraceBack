export interface SessionSummary {
  id:         string;
  label:      string;
  startedAt:  number;
  nodeCount:  number;
  stopped:    boolean;
  anomalous?: boolean;
}

interface Props {
  sessions: SessionSummary[];
  /** Session currently shown. */
  displayId: string | null;
  /** Explicit user pin; null = following the most recently updated session. */
  pinnedId: string | null;
  onSelect: (id: string | null) => void;
}

function badge(s: SessionSummary): string {
  if (s.anomalous) return '🔴';
  if (!s.stopped)  return '🟢';
  return '⚪';
}

/**
 * Fleet selector: one entry per agent session, labeled by working directory,
 * with a live status badge. Anomalies in sessions that are NOT currently
 * displayed also surface as a red alert pill next to the dropdown, so a
 * background failure is visible without opening the menu.
 */
export default function SessionPicker({ sessions, displayId, pinnedId, onSelect }: Props) {
  if (sessions.length === 0) return null;

  const backgroundAnomalous = sessions.filter(
    (s) => s.anomalous && s.id !== displayId
  );

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '0 10px',
      height: 30,
      borderBottom: '1px solid var(--tb-border)',
      background: 'var(--tb-surface)',
      fontFamily: 'var(--tb-ui-font)',
      flexShrink: 0,
      position: 'sticky',
      top: 0,
      zIndex: 30,
    }}>
      <span style={{
        fontSize: 9,
        letterSpacing: '0.15em',
        textTransform: 'uppercase',
        color: 'var(--tb-text-muted)',
        flexShrink: 0,
      }}>
        AGENTS · {sessions.length}
      </span>

      <div style={{ width: 1, height: 12, background: 'var(--tb-border)', flexShrink: 0 }} />

      <select
        value={pinnedId ?? ''}
        onChange={(e) => onSelect(e.target.value || null)}
        style={{
          flex: 1,
          fontSize: 10,
          fontFamily: 'var(--tb-ui-font)',
          letterSpacing: '0.04em',
          background: 'transparent',
          color: 'var(--tb-text)',
          border: 'none',
          outline: 'none',
          cursor: 'pointer',
        }}
      >
        <option value="" style={{ background: 'var(--tb-surface)' }}>
          ⟲ auto — follow latest activity
        </option>
        {sessions.map((s) => (
          <option
            key={s.id}
            value={s.id}
            style={{ background: 'var(--tb-surface)', color: 'var(--tb-text)' }}
          >
            {badge(s)} {s.label} · {s.nodeCount} actions{s.stopped ? '' : ' · LIVE'}
          </option>
        ))}
      </select>

      {backgroundAnomalous.length > 0 && (
        <span
          onClick={() => onSelect(backgroundAnomalous[0].id)}
          title={`Jump to ${backgroundAnomalous[0].label}`}
          style={{
            flexShrink: 0,
            fontSize: 9,
            fontWeight: 700,
            color: '#ffa198',
            background: 'rgba(248,81,73,0.18)',
            border: '1px solid rgba(248,81,73,0.5)',
            borderRadius: 3,
            padding: '1px 6px',
            cursor: 'pointer',
            animation: 'pendingPulse 1.2s ease-in-out infinite',
          }}>
          ⚠ {backgroundAnomalous.length}
        </span>
      )}
    </div>
  );
}
