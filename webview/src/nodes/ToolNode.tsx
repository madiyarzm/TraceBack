import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';

export interface ToolNodeData {
  toolName: string;
  label: string;
  status: 'pending' | 'success' | 'error' | 'thinking';
  count: number;
  detail?: string;
  toolInput?: Record<string, unknown>;
  timestamp: number;
  isLooping?: boolean;
  [key: string]: unknown;
}

const STATUS_COLOR: Record<ToolNodeData['status'], string> = {
  pending:  '#d29922',
  thinking: '#58a6ff',
  success:  '#3fb950',
  error:    '#f85149',
};

const TOOL_ICON: Record<string, string> = {
  Read:      '↘',
  Edit:      '✎',
  Write:     '✦',
  Bash:      '$',
  WebSearch: '⌕',
  WebFetch:  '↗',
  TodoRead:  '☐',
  TodoWrite: '☑',
  Agent:     '◈',
};

const HANDLE_STYLE = { background: 'transparent', border: 'none', width: 0, height: 0 };

function ToolNode({ data, selected }: NodeProps) {
  const d = data as ToolNodeData;
  const color = STATUS_COLOR[d.status];
  const icon  = TOOL_ICON[d.toolName] ?? '·';
  const isPending = d.status === 'pending';

  return (
    <div
      style={{
        width: 228,
        background: 'var(--tb-surface)',
        border: `1px solid ${selected ? color + '60' : 'var(--tb-border)'}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 4,
        fontFamily: 'var(--tb-ui-font)',
        cursor: 'pointer',
        position: 'relative',
        boxShadow: selected
          ? `0 0 0 1px ${color}25, 0 4px 16px rgba(0,0,0,0.5)`
          : '0 1px 3px rgba(0,0,0,0.4)',
        transition: 'border-color 0.15s, box-shadow 0.15s',
        ...(d.isLooping ? { animation: 'stumbleHalo 1.2s ease-in-out infinite' } : {}),
      }}
    >
      <Handle type="target" position={Position.Top} style={HANDLE_STYLE} />

      <div style={{ padding: '6px 10px 7px' }}>
        {/* Row 1: tool badge + count */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: 3,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{
              fontSize: 11,
              color,
              lineHeight: 1,
              ...(isPending ? { animation: 'pendingPulse 1.6s ease-in-out infinite' } : {}),
            }}>
              {icon}
            </span>
            <span style={{
              fontSize: 10,
              fontWeight: 600,
              color: 'var(--tb-text-muted)',
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}>
              {d.toolName}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            {d.count > 1 && (
              <span style={{
                fontSize: 9, fontWeight: 600,
                color: 'var(--tb-text-muted)',
                background: 'var(--tb-surface-2)',
                border: '1px solid var(--tb-border)',
                borderRadius: 3,
                padding: '0 4px',
                lineHeight: '14px',
              }}>
                ×{d.count}
              </span>
            )}
            {d.status === 'error' && (
              <span style={{ fontSize: 10, color: '#f85149' }}>⚠</span>
            )}
          </div>
        </div>

        {/* Row 2: main label */}
        <div style={{
          fontSize: 12,
          fontWeight: 400,
          color: d.status === 'error' ? '#ffa198' : 'var(--tb-text)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          lineHeight: 1.4,
        }}>
          {d.label}
        </div>
      </div>

      {/* Pending progress bar at bottom */}
      {isPending && (
        <div style={{
          height: 2,
          background: `${color}40`,
          borderRadius: '0 0 3px 3px',
          overflow: 'hidden',
        }}>
          <div style={{
            height: '100%',
            width: '40%',
            background: color,
            borderRadius: 1,
            animation: 'shimmer 1.8s ease-in-out infinite',
          }} />
        </div>
      )}

      <Handle type="source" position={Position.Bottom} style={HANDLE_STYLE} />
    </div>
  );
}

export default memo(ToolNode);
