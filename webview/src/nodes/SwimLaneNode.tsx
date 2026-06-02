import { memo } from 'react';
import { NodeProps } from '@xyflow/react';

export interface SwimLanePersonality {
  name:     string;
  badge:    string;
  color:    string;
  colorDim: string;
}

export interface SwimLaneData {
  label:       string;
  personality: SwimLanePersonality;
  nodeCount:   number;
  stopped:     boolean;
  aiSummary?:  string;
  [key: string]: unknown;
}

function SwimLaneNode({ data }: NodeProps) {
  const d = data as SwimLaneData;
  const { personality, label, nodeCount, stopped, aiSummary } = d;

  return (
    <div style={{
      width: '100%',
      height: '100%',
      borderRadius: 6,
      border: `1px solid ${personality.color}25`,
      borderLeft: `3px solid ${personality.color}`,
      background: personality.colorDim,
      fontFamily: 'var(--tb-ui-font)',
      overflow: 'hidden',
      boxSizing: 'border-box',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 7,
        padding: '5px 10px',
        background: `${personality.color}0d`,
        borderBottom: `1px solid ${personality.color}18`,
        height: 28,
        boxSizing: 'border-box',
      }}>
        <span style={{ fontSize: 12, lineHeight: 1, flexShrink: 0 }}>{personality.badge}</span>
        <span style={{
          fontSize: 9,
          fontWeight: 700,
          color: personality.color,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          flexShrink: 0,
        }}>
          {personality.name}
        </span>
        <span style={{ color: 'var(--tb-border-2)', fontSize: 10, flexShrink: 0 }}>·</span>
        <span style={{
          fontSize: 10,
          color: 'var(--tb-text-muted)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          flex: 1,
        }}>
          {label}
        </span>
        <span style={{
          fontSize: 9,
          fontWeight: 600,
          color: 'var(--tb-text-muted)',
          background: 'var(--tb-surface)',
          border: '1px solid var(--tb-border)',
          borderRadius: 3,
          padding: '0 5px',
          lineHeight: '16px',
          flexShrink: 0,
        }}>
          {nodeCount}
        </span>
        {!stopped && (
          <div style={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: '#3fb950',
            flexShrink: 0,
            animation: 'livePulse 2s ease-in-out infinite',
          }} />
        )}
      </div>

      {/* AI summary strip */}
      {aiSummary && (
        <div style={{
          padding: '3px 10px',
          fontSize: 9,
          lineHeight: 1.55,
          color: 'var(--tb-text-muted)',
          fontStyle: 'italic',
          borderBottom: `1px solid ${personality.color}12`,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          height: 20,
          boxSizing: 'border-box',
        }}>
          {aiSummary}
        </div>
      )}
    </div>
  );
}

export default memo(SwimLaneNode);
