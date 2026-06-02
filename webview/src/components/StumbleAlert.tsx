export interface StumbleAlertData {
  type:       'loop' | 'timeout';
  toolName:   string;
  nodeId:     string;
  label:      string;
  detectedAt: number;
}

interface Props {
  alert:     StumbleAlertData | null;
  onDismiss: () => void;
}

function getRescueHint(alert: StumbleAlertData): string {
  if (alert.type === 'loop') {
    return (
      `Claude is repeating the same ${alert.toolName} call ("${alert.label}") and may be stuck in a loop.\n\n` +
      `Try: clarifying your instructions, breaking the task into smaller steps, or checking if the operation is failing silently.`
    );
  }
  return (
    `A ${alert.toolName} operation ("${alert.label}") has been running for over 45 seconds.\n\n` +
    `Try: checking for permission issues, network problems, or canceling and retrying with a more specific command.`
  );
}

export default function StumbleAlert({ alert, onDismiss }: Props) {
  if (!alert) return null;

  const isLoop = alert.type === 'loop';
  const color  = '#f85149';
  const hint   = getRescueHint(alert);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(hint);
    } catch {
      // clipboard may not be available inside the webview context
    }
  }

  return (
    <div
      style={{
        position:     'absolute',
        top:          12,
        left:         '50%',
        transform:    'translateX(-50%)',
        zIndex:       100,
        background:   'var(--tb-surface)',
        border:       `1px solid ${color}40`,
        borderLeft:   `3px solid ${color}`,
        borderRadius: 5,
        padding:      '8px 12px',
        maxWidth:     380,
        width:        'calc(100% - 40px)',
        boxShadow:    '0 4px 24px rgba(0,0,0,0.6)',
        fontFamily:   'var(--tb-ui-font)',
        animation:    'nodeIn 0.2s cubic-bezier(0.16,1,0.3,1) both',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
        <span style={{ fontSize: 13, flexShrink: 0, marginTop: 1, color }}>
          {isLoop ? '⟳' : '⧖'}
        </span>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color, marginBottom: 2 }}>
            {isLoop ? 'Loop detected' : 'Operation timed out'}
          </div>
          <div style={{
            fontSize:      10,
            color:         'var(--tb-text-muted)',
            overflow:      'hidden',
            textOverflow:  'ellipsis',
            whiteSpace:    'nowrap',
          }}>
            {alert.toolName} · {alert.label}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, flexShrink: 0, alignItems: 'center', marginLeft: 4 }}>
          <button
            onClick={handleCopy}
            title={hint}
            style={{
              fontSize:    10,
              fontWeight:  600,
              color:       'var(--tb-blue)',
              background:  'var(--tb-blue-dim)',
              border:      '1px solid rgba(88,166,255,0.2)',
              borderRadius: 3,
              padding:     '2px 7px',
              cursor:      'pointer',
              fontFamily:  'var(--tb-ui-font)',
              whiteSpace:  'nowrap',
            }}
          >
            copy hint ↗
          </button>
          <button
            onClick={onDismiss}
            style={{
              fontSize:   13,
              color:      'var(--tb-text-muted)',
              background: 'none',
              border:     'none',
              cursor:     'pointer',
              padding:    0,
              lineHeight: 1,
              fontFamily: 'var(--tb-ui-font)',
            }}
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}
