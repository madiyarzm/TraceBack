import { useState } from 'react';
import { AlertIcon } from './Icons';
import { codename } from '../codename';

export interface SessionSummary {
  id:         string;
  label:      string;
  startedAt:  number;
  nodeCount:  number;
  stopped:    boolean;
  anomalous?: boolean;
}

interface Props {
  sessions:  SessionSummary[];
  displayId: string | null;
  pinnedId:  string | null;
  onSelect:  (id: string | null) => void;
}

interface PillProps {
  label:     string;
  active:    boolean;
  live:      boolean;
  anomalous: boolean;
  count:     number | null;
  onClick:   () => void;
  isAuto?:   boolean;
}

function SessionPill({ label, active, live, anomalous, count, onClick, isAuto }: PillProps) {
  const [hovered, setHovered] = useState(false);

  const activeBorder  = anomalous ? 'rgba(248,81,73,0.6)' : live ? 'rgba(63,185,80,0.45)' : 'var(--tb-border-2)';
  const activeBg      = anomalous ? 'rgba(248,81,73,0.1)' : live ? 'rgba(63,185,80,0.07)' : 'var(--tb-surface-2)';

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={label}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        padding: '3px 9px',
        borderRadius: 4,
        border: `1px solid ${active ? activeBorder : hovered ? 'var(--tb-border-2)' : 'var(--tb-border)'}`,
        background: active ? activeBg : hovered ? 'var(--tb-surface-2)' : 'transparent',
        color: active ? 'var(--tb-text)' : 'var(--tb-text-muted)',
        cursor: 'pointer',
        flexShrink: 0,
        fontFamily: 'var(--tb-ui-font)',
        fontSize: 10,
        fontWeight: active ? 600 : 400,
        transition: 'background 0.1s, border-color 0.1s, color 0.1s',
        whiteSpace: 'nowrap',
      }}
    >
      {isAuto ? (
        <span style={{ fontSize: 9, opacity: 0.7 }}>⟲</span>
      ) : (
        <div
          className={live && !anomalous ? 'live-dot' : anomalous ? '' : ''}
          style={{
            width: 5, height: 5, borderRadius: '50%', flexShrink: 0,
            background: anomalous ? '#f85149' : live ? '#3fb950' : 'var(--tb-text-dim)',
            ...(anomalous ? { animation: 'pendingPulse 1.2s ease-in-out infinite' } : {}),
          }}
        />
      )}

      <span style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {label}
      </span>

      {count !== null && (
        <span style={{
          fontSize: 8.5,
          color: active ? 'var(--tb-text-muted)' : 'var(--tb-text-dim)',
          fontWeight: 400,
        }}>
          {count}
        </span>
      )}

      {anomalous && (
        <span style={{ color: '#f85149', display: 'flex' }}><AlertIcon size={9} /></span>
      )}
    </button>
  );
}

export default function SessionPicker({ sessions, displayId, pinnedId, onSelect }: Props) {
  if (sessions.length === 0) return null;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 5,
      padding: '5px 10px',
      borderBottom: '1px solid var(--tb-border)',
      background: 'var(--tb-surface)',
      fontFamily: 'var(--tb-ui-font)',
      flexShrink: 0,
      overflowX: 'auto',
      // hide scrollbar but keep scrollable
      scrollbarWidth: 'none',
      msOverflowStyle: 'none',
    } as React.CSSProperties}>
      <SessionPill
        label="auto"
        active={pinnedId === null}
        live={false}
        anomalous={false}
        count={null}
        onClick={() => onSelect(null)}
        isAuto
      />

      <div style={{ width: 1, height: 14, background: 'var(--tb-border)', flexShrink: 0, margin: '0 2px' }} />

      {sessions.map((s) => (
        <SessionPill
          key={s.id}
          label={`${codename(s.id)} · ${s.label}`}
          active={s.id === displayId}
          live={!s.stopped}
          anomalous={!!s.anomalous}
          count={s.nodeCount}
          onClick={() => onSelect(s.id)}
        />
      ))}
    </div>
  );
}
