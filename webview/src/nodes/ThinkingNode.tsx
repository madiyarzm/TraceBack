import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';

const HANDLE_STYLE = { background: 'transparent', border: 'none', width: 0, height: 0 };

function ThinkingNode(_: NodeProps) {
  return (
    <div
      style={{
        fontFamily: 'var(--tb-ui-font)',
        background: 'rgba(88, 166, 255, 0.05)',
        border: '1px dashed rgba(88, 166, 255, 0.25)',
        borderRadius: 4,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 8,
        padding: '5px 12px',
        cursor: 'default',
        userSelect: 'none',
      }}
    >
      <Handle type="target" position={Position.Top} style={HANDLE_STYLE} />

      <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
        {[1, 2, 3].map((i) => (
          <span
            key={i}
            className={`think-dot-${i}`}
            style={{
              display: 'inline-block',
              width: 4,
              height: 4,
              borderRadius: '50%',
              background: '#58a6ff',
            }}
          />
        ))}
      </div>

      <span style={{
        fontSize: 10,
        fontWeight: 500,
        color: 'rgba(88, 166, 255, 0.6)',
        letterSpacing: '0.02em',
      }}>
        thinking
      </span>

      <Handle type="source" position={Position.Bottom} style={HANDLE_STYLE} />
    </div>
  );
}

export default memo(ThinkingNode);
