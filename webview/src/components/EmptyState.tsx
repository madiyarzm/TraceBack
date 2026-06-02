export default function EmptyState() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      gap: 32,
      fontFamily: 'var(--tb-font)',
      userSelect: 'none',
    }}>
      {/* Radar animation */}
      <div style={{
        position: 'relative',
        width: 64,
        height: 64,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div className="radar-ring" style={{ width: 64, height: 64, top: 0, left: 0 }} />
        <div className="radar-ring radar-ring-2" style={{ width: 64, height: 64, top: 0, left: 0 }} />
        <div className="radar-ring radar-ring-3" style={{ width: 64, height: 64, top: 0, left: 0 }} />

        {/* Center dot */}
        <div style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: 'var(--tb-blue, #00aaff)',
          boxShadow: '0 0 12px rgba(0, 170, 255, 0.6)',
          zIndex: 1,
        }} />
      </div>

      {/* Text block */}
      <div style={{ textAlign: 'center' }}>
        <p style={{
          fontSize: 10,
          letterSpacing: '0.2em',
          color: 'var(--tb-text-muted, #4a6070)',
          textTransform: 'uppercase',
          marginBottom: 12,
        }}>
          AWAITING SIGNAL
        </p>

        <p style={{
          fontSize: 11,
          color: 'var(--tb-text-dim, #283545)',
          letterSpacing: '0.05em',
          marginBottom: 20,
        }}>
          No agent activity detected
        </p>

        {/* Command hint */}
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          background: 'var(--tb-surface, #0c1017)',
          border: '1px solid var(--tb-border, #1c2a3a)',
          borderLeft: '3px solid var(--tb-blue, #00aaff)',
          padding: '6px 12px',
        }}>
          <span style={{ color: 'var(--tb-blue, #00aaff)', fontSize: 10 }}>$</span>
          <span style={{ fontSize: 11, color: 'var(--tb-text-muted, #4a6070)' }}>claude</span>
          <span className="cursor-blink" style={{
            fontSize: 12,
            color: 'var(--tb-blue, #00aaff)',
            lineHeight: 1,
          }}>▋</span>
        </div>
      </div>

      {/* Footer hint */}
      <p style={{
        fontSize: 9,
        letterSpacing: '0.1em',
        color: 'var(--tb-text-dim, #283545)',
        textTransform: 'uppercase',
        position: 'absolute',
        bottom: 16,
      }}>
        Listening on :7777
      </p>
    </div>
  );
}
