import { useMemo, useState } from 'react';
import { TimelineNode, formatDuration } from './TimelineCard';
import { computeMetrics, formatCost, formatTokens } from '../metrics';
import { AlertIcon, ClockIcon, PauseIcon, PlayIcon, SparkleIcon } from './Icons';

export interface AnomalyStateUI {
  isAnomalous:     boolean;
  type?:           string;
  severity?:       'high' | 'medium';
  title?:          string;
  description?:    string;
  reason?:         string;
  flaggedEventIds: string[];
}

interface Props {
  nodes:       TimelineNode[];
  isLive:      boolean;
  anomaly?:    AnomalyStateUI;
  /** Cumulative detections this session (anomalyHistory.length) — never resets. */
  anomalyCount?: number;
  /** Breakpoint state: agent frozen at its next tool call. */
  paused?:        boolean;
  onPauseToggle?: () => void;
  onRedirect?:    (message: string) => void;
  /** Real token usage from the transcript; falls back to the estimate. */
  realTokens?: number;
  aiSummary?:  string;
  chatAnswer?: string;
  chatLoading: boolean;
  onChat:      (question: string) => void;
  /** Sidebar mode: one-line inline stats instead of the tile row. */
  slim?: boolean;
  /** Claude is blocked on the user (permission prompt / idle input). */
  awaitingInput?: string;
}

export default function SessionOdometer({
  nodes, isLive, anomaly, anomalyCount = 0, paused = false,
  onPauseToggle, onRedirect, realTokens, aiSummary, chatAnswer, chatLoading, onChat,
  slim = false, awaitingInput,
}: Props) {
  const [question, setQuestion] = useState('');
  const [chatOpen, setChatOpen] = useState(false);
  const [redirect, setRedirect] = useState('');

  function sendRedirect() {
    const msg = redirect.trim();
    if (!msg || !onRedirect) return;
    onRedirect(msg);
    setRedirect('');
  }

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
          animation: 'glitchIn 0.45s steps(3)',
        }}>
          <span style={{ animation: 'pendingPulse 1.2s ease-in-out infinite', display: 'flex' }}>
            <AlertIcon size={13} />
          </span>
          <span>
            {/* A loop is a glitch in the Matrix: you've seen this action before */}
            {(anomaly?.type === 'repeater' || anomaly?.type === 'near_duplicate_loop') && (
              <span style={{ fontStyle: 'italic' }}>Déjà vu — </span>
            )}
            {anomaly?.description ?? anomaly?.reason}
          </span>
        </div>
      )}

      {/* ── Waiting-for-you banner (permission prompt / idle input) ── */}
      {!alarmed && awaitingInput && isLive && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          padding: '6px 12px',
          borderBottom: '1px solid rgba(210,153,34,0.3)',
          background: 'rgba(210,153,34,0.06)',
          color: '#d29922',
          fontSize: 11, fontWeight: 600,
        }}>
          <span style={{ animation: 'pendingPulse 1.6s ease-in-out infinite', display: 'flex' }}>
            <ClockIcon size={13} />
          </span>
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {awaitingInput} — check your terminal
          </span>
        </div>
      )}

      {/* ── Stat row ── */}
      {slim ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 12px' }}>
          <span style={{
            fontSize: 10, color: 'var(--tb-text-muted)',
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
            flex: 1, minWidth: 0,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {formatDuration(m.totalDurationMs) ?? '—'}
            {' · '}{m.toolCount} actions
            {m.errorCount   > 0 && <span style={{ color: '#f85149' }}> · {m.errorCount} err</span>}
            {anomalyCount   > 0 && <span style={{ color: '#f85149' }}> · {anomalyCount} ⚠</span>}
            {' · '}{formatTokens(realTokens ?? m.estTokens)} tok
          </span>
          {isLive && onPauseToggle && (
            <button
              onClick={onPauseToggle}
              title={paused ? 'Resume — release the held tool call' : 'Bullet time — freeze the agent at its next tool call'}
              style={{
                background: paused ? 'rgba(210,153,34,0.15)' : 'none',
                border: `1px solid ${paused ? 'rgba(210,153,34,0.6)' : 'var(--tb-border)'}`,
                borderRadius: 3,
                color: paused ? '#d29922' : 'var(--tb-text-muted)',
                fontSize: 9.5, fontWeight: 600,
                padding: '2px 7px', cursor: 'pointer',
                flexShrink: 0,
                display: 'flex', alignItems: 'center',
              }}
            >
              {paused ? <PlayIcon size={10} /> : <PauseIcon size={10} />}
            </button>
          )}
          <button
            onClick={() => setChatOpen((v) => !v)}
            title="Ask AI about this session"
            style={{
              background: chatOpen ? 'rgba(88,166,255,0.1)' : 'none',
              border: `1px solid ${chatOpen ? 'rgba(88,166,255,0.4)' : 'var(--tb-border)'}`,
              borderRadius: 3,
              color: chatOpen ? 'var(--tb-blue)' : 'var(--tb-text-muted)',
              fontSize: 9.5, padding: '2px 7px', cursor: 'pointer',
              flexShrink: 0,
              display: 'flex', alignItems: 'center', gap: 4,
            }}
          >
            <SparkleIcon size={10} /> AI
          </button>
        </div>
      ) : (
      <div style={{ display: 'flex', alignItems: 'center', padding: '7px 12px', gap: 0, flexWrap: 'wrap' }}>
        <StatGroup>
          <Stat label="time"    value={formatDuration(m.totalDurationMs) ?? '—'} />
          <Stat label="actions" value={String(m.toolCount)} />
        </StatGroup>

        <StatDivider />

        <StatGroup>
          <Stat label="errors"    value={String(m.errorCount)}  color={m.errorCount  > 0 ? '#f85149' : undefined} />
          <Stat label="anomalies" value={String(anomalyCount)}  color={anomalyCount  > 0 ? '#f85149' : undefined} />
        </StatGroup>

        <StatDivider />

        <StatGroup>
          <Stat
            label={realTokens !== undefined ? 'tokens' : '≈ tokens'}
            value={formatTokens(realTokens ?? m.estTokens)}
            dim={realTokens === undefined}
          />
          {/* Only show cost when it's non-trivial */}
          {(realTokens ?? m.estTokens) > 50_000 && (
            <Stat
              label="≈ cost"
              value={formatCost(((realTokens ?? m.estTokens) / 1_000_000) * 6)}
              dim
            />
          )}
        </StatGroup>

        <div style={{ flex: 1 }} />

        {isLive && onPauseToggle && (
          <button
            onClick={onPauseToggle}
            title={paused ? 'Resume — release the held tool call' : 'Bullet time — freeze the agent at its next tool call'}
            style={{
              background: paused ? 'rgba(210,153,34,0.15)' : 'none',
              border: `1px solid ${paused ? 'rgba(210,153,34,0.6)' : 'var(--tb-border)'}`,
              borderRadius: 3,
              color: paused ? '#d29922' : 'var(--tb-text-muted)',
              fontSize: 10, fontWeight: 600,
              padding: '3px 10px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 5,
            }}
          >
            {paused ? <PlayIcon size={10} /> : <PauseIcon size={10} />}
            <span>{paused ? 'Resume' : 'Pause'}</span>
          </button>
        )}

        <button
          onClick={() => setChatOpen((v) => !v)}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            background: chatOpen ? 'rgba(88,166,255,0.1)' : 'none',
            border: `1px solid ${chatOpen ? 'rgba(88,166,255,0.4)' : 'var(--tb-border)'}`,
            borderRadius: 3,
            color: chatOpen ? 'var(--tb-blue)' : 'var(--tb-text-muted)',
            fontSize: 10, fontWeight: chatOpen ? 600 : 400,
            padding: '3px 10px', cursor: 'pointer',
            marginLeft: isLive && onPauseToggle ? 6 : 0,
            transition: 'background 0.1s, border-color 0.1s, color 0.1s',
          }}
        >
          <SparkleIcon size={11} />
          <span>Ask AI</span>
        </button>
      </div>
      )}

      {/* ── Breakpoint banner + human-in-the-loop redirect ── */}
      {paused && (
        <div style={{
          padding: '6px 12px 9px',
          borderBottom: '1px solid rgba(210,153,34,0.35)',
          background: 'rgba(70,52,12,0.45)',
          display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            color: '#d29922', fontSize: 10.5, fontWeight: 600,
          }}>
            <span style={{ animation: 'pendingPulse 1.4s ease-in-out infinite', display: 'flex' }}>
              <PauseIcon size={12} />
            </span>
            <span>BULLET TIME — agent frozen at its next tool call until you resume or redirect</span>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              value={redirect}
              onChange={(e) => setRedirect(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendRedirect()}
              placeholder="Redirect the agent… e.g. 'stop installing that package, use the native lib'"
              style={{
                flex: 1,
                background: 'var(--tb-surface)',
                border: '1px solid rgba(210,153,34,0.4)',
                borderRadius: 3,
                color: 'var(--tb-text)',
                fontSize: 11, padding: '4px 8px',
                outline: 'none',
              }}
            />
            <button
              onClick={sendRedirect}
              disabled={!redirect.trim()}
              style={{
                background: 'rgba(210,153,34,0.15)',
                border: '1px solid rgba(210,153,34,0.5)',
                borderRadius: 3,
                color: '#d29922',
                fontSize: 10.5, fontWeight: 600,
                padding: '0 10px',
                cursor: redirect.trim() ? 'pointer' : 'not-allowed',
              }}
            >
              intercept & send
            </button>
          </div>
        </div>
      )}

      {/* ── AI narrative line ── */}
      {aiSummary && (
        <div style={{
          padding: '0 12px 7px',
          display: 'flex', alignItems: 'baseline', gap: 6,
          fontSize: 10.5, lineHeight: 1.45,
          color: 'var(--tb-text-muted)',
          fontStyle: 'italic',
        }}>
          <SparkleIcon size={10} style={{ position: 'relative', top: 1, opacity: 0.6 }} />
          <span style={{ minWidth: 0, wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
            {stripMd(aiSummary)}
          </span>
        </div>
      )}

      {/* ── Chat ── */}
      {chatOpen && (
        <div style={{
          padding: '0 12px 9px', display: 'flex', flexDirection: 'column', gap: 6,
          animation: 'chatPanelIn 0.15s ease-out',
        }}>
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
              fontSize: 11, lineHeight: 1.55,
              color: 'var(--tb-text)',
              background: 'var(--tb-surface)',
              border: '1px solid var(--tb-border)',
              borderRadius: 4, padding: '6px 9px',
              maxHeight: 180, overflowY: 'auto',
              overflowX: 'hidden',
              wordBreak: 'break-word',
              overflowWrap: 'anywhere',
            }}>
              <MarkdownLite text={chatAnswer} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Belt-and-suspenders: the summary prompt forbids markdown, but strip any
 *  markers that slip through rather than rendering them as literal asterisks. */
function stripMd(text: string): string {
  return text
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1');
}

/**
 * Minimal markdown renderer for chat answers: **bold**, `code`, and "- "
 * bullets only — exactly what the system prompt permits. No library: a full
 * md renderer invites the layout blowouts this exists to prevent.
 */
function MarkdownLite({ text }: { text: string }) {
  const lines = text.split('\n').filter((l) => l.trim().length > 0);
  return (
    <>
      {lines.map((raw, i) => {
        const line     = raw.replace(/^#{1,6}\s*/, ''); // demote stray headings
        const isBullet = /^\s*[-•]\s+/.test(line);
        const content  = isBullet ? line.replace(/^\s*[-•]\s+/, '') : line;
        return (
          <div key={i} style={{
            display: 'flex', gap: 5,
            padding: '1px 0',
            paddingLeft: isBullet ? 4 : 0,
          }}>
            {isBullet && <span style={{ color: 'var(--tb-text-muted)', flexShrink: 0 }}>•</span>}
            <span style={{ minWidth: 0 }}>{renderInline(content)}</span>
          </div>
        );
      })}
    </>
  );
}

function renderInline(line: string): React.ReactNode[] {
  return line
    .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    .filter(Boolean)
    .map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} style={{ fontWeight: 600 }}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={i} style={{
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
            fontSize: 10,
            background: 'var(--tb-surface-2)',
            border: '1px solid var(--tb-border)',
            borderRadius: 3,
            padding: '0 3px',
            wordBreak: 'break-all',
          }}>{part.slice(1, -1)}</code>
        );
      }
      return part;
    });
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

function StatGroup({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 14 }}>
      {children}
    </div>
  );
}

function StatDivider() {
  return (
    <div style={{ width: 1, height: 28, background: 'var(--tb-border)', margin: '0 12px', flexShrink: 0 }} />
  );
}
