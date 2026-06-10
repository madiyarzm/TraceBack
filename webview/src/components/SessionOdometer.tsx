import { useMemo, useState } from 'react';
import { TimelineNode, formatDuration } from './TimelineCard';
import { computeMetrics, formatCost, formatTokens } from '../metrics';

export interface AnomalyStateUI {
  isAnomalous:     boolean;
  type?:           'repeater' | 'error_thrash' | 'stall';
  reason?:         string;
  flaggedEventIds: string[];
}

interface Props {
  nodes:       TimelineNode[];
  isLive:      boolean;
  anomaly?:    AnomalyStateUI;
  /** Real token usage from the transcript; falls back to the estimate. */
  realTokens?: number;
  aiSummary?:  string;
  chatAnswer?: string;
  chatLoading: boolean;
  onChat:      (question: string) => void;
}

export default function SessionOdometer({
  nodes, isLive, anomaly, realTokens, aiSummary, chatAnswer, chatLoading, onChat,
}: Props) {
  const [question, setQuestion] = useState('');
  const [chatOpen, setChatOpen] = useState(false);

  const m = useMemo(() => computeMetrics(nodes), [nodes]);

  function submit() {
    const q = question.trim();
    if (!q || chatLoading) return;
    onChat(q);
    setQuestion('');
  }

  const alarmed = anomaly?.isAnomalous ?? false;

  return (
    <div style={{
      position: 'sticky', top: 0, zIndex: 20,
      background: alarmed ? 'rgba(70,16,16,0.97)' : 'var(--tb-bg, #07090d)',
      borderBottom: `1px solid ${alarmed ? 'rgba(248,81,73,0.6)' : 'var(--tb-border)'}`,
      fontFamily: 'var(--tb-ui-font)',
      transition: 'background 0.25s, border-color 0.25s',
    }}>
      {/* ── Anomaly banner ── */}
      {alarmed && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          padding: '6px 12px',
          borderBottom: '1px solid rgba(248,81,73,0.35)',
          color: '#ffa198',
          fontSize: 11, fontWeight: 600,
        }}>
          <span style={{ animation: 'pendingPulse 1.2s ease-in-out infinite' }}>⚠</span>
          <span>{anomaly?.reason}</span>
        </div>
      )}

      {/* ── Stat row ── */}
      <div style={{ display: 'flex', alignItems: 'stretch', padding: '7px 12px', gap: 14, flexWrap: 'wrap' }}>
        <Stat label="total time" value={formatDuration(m.totalDurationMs) ?? '—'} />
        <Stat label="actions"    value={String(m.toolCount)} />
        <Stat
          label="errors"
          value={String(m.errorCount)}
          color={m.errorCount > 0 ? '#f85149' : undefined}
        />
        <Stat
          label="loops"
          value={anomaly?.type === 'repeater' ? '1' : '0'}
          color={anomaly?.type === 'repeater' ? '#f85149' : undefined}
        />
        {/* Real tokens come from the transcript; ≈ marks heuristic fallbacks */}
        <Stat
          label={realTokens !== undefined ? 'tokens' : '≈ tokens'}
          value={formatTokens(realTokens ?? m.estTokens)}
          dim={realTokens === undefined}
        />
        <Stat
          label="≈ cost"
          value={formatCost(((realTokens ?? m.estTokens) / 1_000_000) * 6)}
          dim
        />
        <div style={{ flex: 1 }} />
        <button
          onClick={() => setChatOpen((v) => !v)}
          style={{
            background: 'none',
            border: '1px solid var(--tb-border)',
            borderRadius: 3,
            color: chatOpen ? 'var(--tb-text)' : 'var(--tb-text-muted)',
            fontSize: 9.5, padding: '0 8px', cursor: 'pointer',
            alignSelf: 'center', lineHeight: '18px',
          }}
        >
          ask AI {chatOpen ? '▴' : '▾'}
        </button>
      </div>

      {/* ── AI narrative line ── */}
      {aiSummary && (
        <div style={{
          padding: '0 12px 7px',
          fontSize: 10.5, lineHeight: 1.45,
          color: 'var(--tb-text-muted)',
          fontStyle: 'italic',
        }}>
          {isLive ? '◉ ' : ''}{aiSummary}
        </div>
      )}

      {/* ── Chat (preserves the Narrative Engine Q&A in the new layout) ── */}
      {chatOpen && (
        <div style={{ padding: '0 12px 9px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit()}
              placeholder="Why did the agent fail? What files were touched?"
              style={{
                flex: 1,
                background: 'var(--tb-surface)',
                border: '1px solid var(--tb-border)',
                borderRadius: 3,
                color: 'var(--tb-text)',
                fontSize: 11, padding: '4px 8px',
                outline: 'none',
              }}
            />
            <button
              onClick={submit}
              disabled={chatLoading}
              style={{
                background: 'var(--tb-surface-2)',
                border: '1px solid var(--tb-border)',
                borderRadius: 3,
                color: 'var(--tb-text)',
                fontSize: 10.5, padding: '0 10px',
                cursor: chatLoading ? 'wait' : 'pointer',
              }}
            >
              {chatLoading ? '…' : 'ask'}
            </button>
          </div>
          {chatAnswer && (
            <div style={{
              fontSize: 11, lineHeight: 1.5,
              color: 'var(--tb-text)',
              background: 'var(--tb-surface)',
              border: '1px solid var(--tb-border)',
              borderRadius: 4, padding: '6px 9px',
              maxHeight: 140, overflowY: 'auto',
            }}>
              {chatAnswer}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, color, dim }: {
  label: string; value: string; color?: string; dim?: boolean;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{
        fontSize: 13, fontWeight: 600,
        fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
        color: color ?? (dim ? 'var(--tb-text-muted)' : 'var(--tb-text)'),
        lineHeight: 1.1,
      }}>{value}</span>
      <span style={{
        fontSize: 8.5, fontWeight: 600,
        color: 'var(--tb-text-dim)',
        letterSpacing: '0.07em', textTransform: 'uppercase',
      }}>{label}</span>
    </div>
  );
}
