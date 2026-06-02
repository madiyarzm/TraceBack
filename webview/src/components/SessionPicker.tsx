import vscode from '../vscodeApi';

export interface SessionSummary {
  id: string;
  label: string;
  startedAt: number;
  nodeCount: number;
  stopped: boolean;
}

interface Props {
  sessions: SessionSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
}

export default function SessionPicker({ sessions, activeId, onSelect }: Props) {
  if (sessions.length <= 1) return null;

  function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const id = e.target.value;
    onSelect(id);
    vscode.postMessage({ type: 'switch_session', sessionId: id });
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '0 10px',
      height: 28,
      borderBottom: '1px solid var(--tb-border)',
      background: 'var(--tb-surface)',
      fontFamily: 'var(--tb-font)',
      flexShrink: 0,
    }}>
      <span style={{
        fontSize: 9,
        letterSpacing: '0.15em',
        textTransform: 'uppercase',
        color: 'var(--tb-text-muted)',
        flexShrink: 0,
      }}>
        SESSION
      </span>

      <div style={{
        width: 1,
        height: 12,
        background: 'var(--tb-border)',
        flexShrink: 0,
      }} />

      <select
        value={activeId ?? ''}
        onChange={handleChange}
        style={{
          flex: 1,
          fontSize: 10,
          fontFamily: 'var(--tb-font)',
          letterSpacing: '0.04em',
          background: 'transparent',
          color: 'var(--tb-text)',
          border: 'none',
          outline: 'none',
          cursor: 'pointer',
        }}
      >
        {sessions.map((s) => (
          <option
            key={s.id}
            value={s.id}
            style={{ background: 'var(--tb-surface)', color: 'var(--tb-text)' }}
          >
            {s.label}  ·  {s.nodeCount} actions{s.stopped ? '' : '  ·  LIVE'}
          </option>
        ))}
      </select>
    </div>
  );
}
