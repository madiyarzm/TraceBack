import { useState } from 'react';

interface Props {
  isLive: boolean;
  nodeCount: number;
  onExportPng: () => void;
  onExportJson: () => void;
  onCopyReport: () => Promise<void> | void;
  onClear: () => void;
}

export default function Toolbar({ isLive, nodeCount, onExportPng, onExportJson, onCopyReport, onClear }: Props) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    await onCopyReport();
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 10px',
      height: 28,
      borderBottom: '1px solid var(--tb-border)',
      background: 'var(--tb-surface)',
      fontFamily: 'var(--tb-font)',
      flexShrink: 0,
    }}>
      {/* Left: status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {/* Live dot */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div
            className={isLive ? 'live-dot' : ''}
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: isLive ? 'var(--tb-green)' : 'var(--tb-text-dim)',
              flexShrink: 0,
            }}
          />
          <span style={{
            fontSize: 9,
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
            color: isLive ? 'var(--tb-green)' : 'var(--tb-text-muted)',
          }}>
            {isLive ? 'Live' : 'Done'}
          </span>
        </div>

        <span style={{
          width: 1,
          height: 12,
          background: 'var(--tb-border)',
        }} />

        <span style={{
          fontSize: 9,
          letterSpacing: '0.1em',
          color: 'var(--tb-text-muted)',
        }}>
          {nodeCount} <span style={{ color: 'var(--tb-text-dim)' }}>ACTIONS</span>
        </span>
      </div>

      {/* Right: actions */}
      <div style={{ display: 'flex', gap: 2 }}>
        <BarButton onClick={onExportPng}>PNG</BarButton>
        <BarButton onClick={onExportJson}>JSON</BarButton>
        <BarButton onClick={handleCopy}>{copied ? '✓' : 'MD'}</BarButton>
        <BarButton onClick={onClear} danger>CLR</BarButton>
      </div>
    </div>
  );
}

function BarButton({
  children,
  onClick,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        fontSize: 9,
        fontFamily: 'var(--tb-font)',
        fontWeight: 700,
        letterSpacing: '0.12em',
        padding: '2px 7px',
        cursor: 'pointer',
        background: 'transparent',
        color: danger ? 'var(--tb-red)' : 'var(--tb-text-muted)',
        border: `1px solid ${danger ? 'rgba(255,61,90,0.3)' : 'var(--tb-border)'}`,
        transition: 'color 0.1s, border-color 0.1s, background 0.1s',
      }}
      onMouseEnter={(e) => {
        const el = e.currentTarget;
        el.style.color = danger ? '#ff3d5a' : '#00aaff';
        el.style.borderColor = danger ? 'rgba(255,61,90,0.6)' : 'rgba(0,170,255,0.4)';
        el.style.background = danger ? 'rgba(255,61,90,0.06)' : 'rgba(0,170,255,0.06)';
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget;
        el.style.color = danger ? 'var(--tb-red)' : 'var(--tb-text-muted)';
        el.style.borderColor = danger ? 'rgba(255,61,90,0.3)' : 'var(--tb-border)';
        el.style.background = 'transparent';
      }}
    >
      {children}
    </button>
  );
}
