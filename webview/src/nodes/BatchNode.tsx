import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';

export interface BatchNodeData {
  toolName: string;
  status: 'pending' | 'success' | 'error' | 'thinking';
  count: number;
  detail?: string;
  timestamp: number;
  batchItems: { label: string; status: 'pending' | 'success' | 'error' | 'thinking' }[];
  isBatch: true;
  [key: string]: unknown;
}

const STATUS_COLOR: Record<BatchNodeData['status'], string> = {
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
const MAX_VISIBLE  = 3;

function BatchNode({ data, selected }: NodeProps) {
  const d       = data as BatchNodeData;
  const color   = STATUS_COLOR[d.status];
  const icon    = TOOL_ICON[d.toolName] ?? '·';
  const visible = d.batchItems.slice(0, MAX_VISIBLE);
  const overflow = d.batchItems.length - MAX_VISIBLE;

  return (
    <div
      style={{
        width: 244,
        background: 'var(--tb-surface)',
        border: `1px solid ${selected ? color + '60' : 'var(--tb-border)'}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 4,
        fontFamily: 'var(--tb-ui-font)',
        cursor: 'pointer',
        boxShadow: selected
          ? `0 0 0 1px ${color}25, 0 4px 16px rgba(0,0,0,0.5)`
          : '0 1px 3px rgba(0,0,0,0.4)',
        transition: 'border-color 0.15s, box-shadow 0.15s',
      }}
    >
      <Handle type="target" position={Position.Top} style={HANDLE_STYLE} />

      {/* ── Header ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 10px 5px',
        borderBottom: '1px solid var(--tb-border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ fontSize: 11, color, lineHeight: 1 }}>{icon}</span>
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

        <span style={{
          fontSize: 10,
          fontWeight: 600,
          color,
          background: `${color}18`,
          border: `1px solid ${color}30`,
          borderRadius: 3,
          padding: '1px 6px',
          lineHeight: '14px',
        }}>
          ×{d.count}
        </span>
      </div>

      {/* ── Step list ── */}
      <div style={{ padding: '5px 10px' }}>
        {visible.map((item, i) => (
          <div
            key={i}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 7,
              padding: '2px 0',
            }}
          >
            <span style={{
              width: 5,
              height: 5,
              borderRadius: '50%',
              background: STATUS_COLOR[item.status],
              flexShrink: 0,
            }} />
            <span style={{
              fontSize: 11,
              color: 'var(--tb-text)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
            }}>
              {item.label}
            </span>
          </div>
        ))}

        {overflow > 0 && (
          <div style={{
            fontSize: 10,
            color: 'var(--tb-text-muted)',
            padding: '2px 0 1px',
          }}>
            +{overflow} more
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <div style={{
        borderTop: '1px solid var(--tb-border)',
        padding: '3px 10px 4px',
        fontSize: 10,
        color: 'var(--tb-text-dim)',
        display: 'flex',
        alignItems: 'center',
        gap: 4,
      }}>
        <span style={{ color: 'var(--tb-text-muted)' }}>↗</span>
        <span>click to inspect all steps</span>
      </div>

      <Handle type="source" position={Position.Bottom} style={HANDLE_STYLE} />
    </div>
  );
}

export default memo(BatchNode);
