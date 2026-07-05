import { useState } from 'react';
import TimelineCard, { TimelineNode, formatDuration } from './TimelineCard';
import { ChapterTaskGroup, pendingPlanItems, PromptChapter } from '../chapters';
import { computeMetrics, formatTokens } from '../metrics';
import { AnomalyStateUI } from './SessionOdometer';
import { AlertIcon, CheckIcon, ChevronIcon, ClockIcon, PauseIcon, PlayIcon } from './Icons';

interface Props {
  chapter:     PromptChapter;
  isLive:      boolean;
  /** Real session tokens shown only for the live (last) chapter. */
  realTokens?: number;
  isLast:      boolean;
  expandedId:  string | null;
  onToggle:    (id: string) => void;
  /** Live-session controls — only meaningful on the last, live chapter. */
  anomaly?:       AnomalyStateUI;
  paused?:        boolean;
  onPauseToggle?: () => void;
  onRedirect?:    (message: string) => void;
}

/**
 * The main-panel chapter view: the prompt with Pause/Redirect controls, task
 * progress and stats for this prompt, an anomaly banner when one is live, then
 * the agent's work grouped into collapsible task blocks. Done tasks collapse to
 * a one-line file summary; the active task shows its action cards; pending plan
 * items render locked.
 */
export default function PromptChapterView({
  chapter, isLive, realTokens, isLast, expandedId, onToggle,
  anomaly, paused = false, onPauseToggle, onRedirect,
}: Props) {
  const done  = chapter.plan.filter((p) => p.status === 'completed').length;
  const total = chapter.plan.length;
  const m     = computeMetrics(chapter.nodes);
  const tokens = isLast && realTokens !== undefined ? realTokens : m.estTokens;
  const pending = pendingPlanItems(chapter);

  const showControls = isLive && (onPauseToggle || onRedirect);
  const [redirectOpen, setRedirectOpen] = useState(false);
  const [redirect, setRedirect] = useState('');

  function sendRedirect() {
    const msg = redirect.trim();
    if (!msg || !onRedirect) return;
    onRedirect(msg);
    setRedirect('');
    setRedirectOpen(false);
  }

  return (
    <div style={{ fontFamily: 'var(--tb-ui-font)' }}>
      {/* ── Chapter header ── */}
      <div style={{
        padding: '16px 20px 14px',
        borderBottom: '1px solid var(--tb-border)',
        background: 'var(--tb-surface)',
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
          <div style={{
            flex: 1, minWidth: 0,
            fontSize: 17, fontWeight: 650, lineHeight: 1.35,
            color: 'var(--tb-text)',
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}>
            “{chapter.text}”
          </div>

          {showControls && (
            <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
              {onPauseToggle && (
                <ControlButton
                  onClick={onPauseToggle}
                  active={paused}
                  activeColor="#d29922"
                  icon={paused ? <PlayIcon size={13} /> : <PauseIcon size={13} />}
                  label={paused ? 'Resume' : 'Pause'}
                />
              )}
              {onRedirect && (
                <ControlButton
                  onClick={() => setRedirectOpen((v) => !v)}
                  active={redirectOpen}
                  activeColor="#58a6ff"
                  icon={<span style={{ fontSize: 14, lineHeight: 1 }}>↵</span>}
                  label="Redirect"
                />
              )}
            </div>
          )}
        </div>

        {/* ── Redirect input ── */}
        {showControls && redirectOpen && (
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <input
              autoFocus
              value={redirect}
              onChange={(e) => setRedirect(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') sendRedirect();
                if (e.key === 'Escape') { setRedirectOpen(false); setRedirect(''); }
              }}
              placeholder="Redirect the agent… e.g. 'stop installing that package, use the native lib'"
              style={{
                flex: 1,
                background: 'var(--tb-bg)',
                border: '1px solid rgba(88,166,255,0.4)',
                borderRadius: 6,
                color: 'var(--tb-text)',
                fontSize: 12, padding: '8px 11px',
                outline: 'none',
              }}
            />
            <button
              onClick={sendRedirect}
              disabled={!redirect.trim()}
              style={{
                background: 'rgba(88,166,255,0.12)',
                border: '1px solid rgba(88,166,255,0.5)',
                borderRadius: 6,
                color: 'var(--tb-blue)',
                fontSize: 12, fontWeight: 600,
                padding: '0 14px',
                cursor: redirect.trim() ? 'pointer' : 'not-allowed',
              }}
            >
              Send
            </button>
          </div>
        )}

        {/* ── Stats row ── */}
        <div style={{
          display: 'flex', alignItems: 'flex-end', gap: 26, marginTop: 16,
          flexWrap: 'wrap',
        }}>
          {total > 0 && (
            <span style={{
              fontSize: 13,
              color: done === total ? 'var(--tb-green)' : 'var(--tb-text-muted)',
              alignSelf: 'flex-end', paddingBottom: 1,
            }}>
              {done} of {total} tasks
            </span>
          )}
          <HeaderStat value={String(chapter.actionCount)} label="actions" />
          <HeaderStat
            value={String(chapter.errorCount)} label="errors"
            color={chapter.errorCount > 0 ? '#f85149' : undefined}
          />
          <HeaderStat value={formatDuration(chapter.durationMs) ?? '—'} label="duration" />
          <HeaderStat
            value={formatTokens(tokens)}
            label={isLast && realTokens !== undefined ? 'tokens' : '≈ tokens'}
          />
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{ padding: '14px 16px 28px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* ── Live anomaly banner ── */}
        {anomaly?.isAnomalous && (
          <AnomalyBanner anomaly={anomaly} paused={paused} onPauseToggle={onPauseToggle} />
        )}

        {chapter.nodes.length === 0 && (
          <div style={{ fontSize: 12, color: 'var(--tb-text-dim)', padding: '8px 2px' }}>
            {isLive ? 'Waiting — no actions for this prompt yet.' : 'No actions recorded for this prompt.'}
          </div>
        )}

        {chapter.taskGroups.map((g, i) => (
          <TaskBlock
            key={i}
            group={g}
            isLive={isLive}
            expandedId={expandedId}
            onToggle={onToggle}
          />
        ))}

        {pending.map((p, i) => (
          <div key={`pending-${i}`} style={{
            display: 'flex', alignItems: 'center', gap: 11,
            padding: '11px 14px',
            borderRadius: 8,
            border: '1px solid var(--tb-border)',
            background: 'var(--tb-surface)',
            color: 'var(--tb-text-dim)',
            fontSize: 12.5,
            userSelect: 'none',
          }}>
            <span style={{
              width: 16, height: 16, borderRadius: '50%', flexShrink: 0,
              border: '1.5px solid var(--tb-border-2)',
            }} />
            <span style={{
              flex: 1, minWidth: 0,
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>{p.content}</span>
            <span style={{ fontSize: 10, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              waiting
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Live anomaly banner ───────────────────────────────────────────────────────

function AnomalyBanner({ anomaly, paused, onPauseToggle }: {
  anomaly: AnomalyStateUI; paused?: boolean; onPauseToggle?: () => void;
}) {
  const high  = (anomaly.severity ?? 'high') === 'high';
  const color = high ? '#f85149' : '#d29922';
  const Icon  = anomaly.type === 'stall' ? ClockIcon : AlertIcon;
  const title = (anomaly.title ?? String(anomaly.type ?? 'Anomaly').replace(/_/g, ' '));

  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 14,
      padding: '14px 16px',
      borderRadius: 10,
      border: `1px solid ${color}66`,
      background: high ? 'rgba(120,20,18,0.35)' : 'rgba(90,66,12,0.35)',
    }}>
      <div style={{
        width: 40, height: 40, borderRadius: 9, flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: `${color}22`,
        color,
      }}>
        <Icon size={20} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: 14, fontWeight: 700, color, lineHeight: 1.3, marginBottom: 3,
        }}>
          {title} detected
        </div>
        <div style={{
          fontSize: 12.5, lineHeight: 1.5,
          color: high ? '#ffb3ac' : '#e8c877',
          wordBreak: 'break-word',
        }}>
          {anomaly.description ?? anomaly.reason}
        </div>
      </div>
      {onPauseToggle && (
        <button
          onClick={onPauseToggle}
          style={{
            flexShrink: 0, alignSelf: 'center',
            background: 'transparent',
            border: `1px solid ${color}88`,
            borderRadius: 7,
            color,
            fontSize: 13, fontWeight: 600, fontFamily: 'var(--tb-ui-font)',
            padding: '8px 16px', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: 6,
          }}
        >
          {paused ? <PlayIcon size={12} /> : <PauseIcon size={12} />}
          {paused ? 'Resume' : 'Pause'}
        </button>
      )}
    </div>
  );
}

// ── Task block ────────────────────────────────────────────────────────────────

function TaskBlock({ group, isLive, expandedId, onToggle }: {
  group:      ChapterTaskGroup;
  isLive:     boolean;
  expandedId: string | null;
  onToggle:   (id: string) => void;
}) {
  const isDone   = group.status === 'completed' && group.errorCount === 0;
  const isActive = group.status === 'in_progress';
  const hasError = group.errorCount > 0;
  const [open, setOpen] = useState(!isDone);

  // Ungrouped actions (no objective recorded) — render cards directly.
  if (!group.objective) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {group.nodes.map((n) => (
          <ActionCardRow key={n.id} node={n} expandedId={expandedId} onToggle={onToggle} />
        ))}
      </div>
    );
  }

  return (
    <div style={{
      border: `1px solid ${isActive ? 'var(--tb-border-2)' : 'var(--tb-border)'}`,
      borderRadius: 10,
      background: 'var(--tb-surface)',
      overflow: 'hidden',
    }}>
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '13px 15px',
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <StatusMark isDone={isDone} isActive={isActive} hasError={hasError} isLive={isLive} />
        <span style={{
          fontSize: 13.5, fontWeight: isActive ? 650 : 550,
          color: isDone ? 'var(--tb-text-muted)' : 'var(--tb-text)',
          flex: 1, minWidth: 0,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {group.objective}
        </span>
        {isActive ? (
          <span style={{ fontSize: 11.5, color: 'var(--tb-blue)', flexShrink: 0 }}>
            active · {group.nodes.length} action{group.nodes.length === 1 ? '' : 's'}
          </span>
        ) : (
          <span style={{ fontSize: 11.5, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
            {group.nodes.length} action{group.nodes.length === 1 ? '' : 's'}
          </span>
        )}
        <span style={{
          color: 'var(--tb-text-dim)', display: 'flex', flexShrink: 0,
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.15s',
        }}>
          <ChevronIcon size={13} />
        </span>
      </div>

      {/* Done tasks collapse to a one-line "TOOL path · TOOL path" summary */}
      {!open && collapsedSummary(group).length > 0 && (
        <div style={{
          padding: '0 15px 11px 39px',
          fontSize: 11,
          fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {collapsedSummary(group).map((e, i) => (
            <span key={i}>
              {i > 0 && <span style={{ color: 'var(--tb-text-dim)' }}> · </span>}
              <span style={{ color: 'var(--tb-text-muted)', fontWeight: 600 }}>{e.tool}</span>
              <span style={{ color: 'var(--tb-text-dim)' }}> {e.path}</span>
            </span>
          ))}
        </div>
      )}

      {open && (
        <div style={{
          borderTop: '1px solid var(--tb-border)',
          padding: '12px 14px',
          display: 'flex', flexDirection: 'column', gap: 8,
          animation: 'cardBodyIn 0.15s ease-out',
        }}>
          {group.nodes.map((n) => (
            <ActionCardRow key={n.id} node={n} expandedId={expandedId} onToggle={onToggle} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Round status marker: green check (done), blue spinner (active), red alert
 *  (errored), or an empty ring (pending) — matches the target task rail. */
function StatusMark({ isDone, isActive, hasError, isLive }: {
  isDone: boolean; isActive: boolean; hasError: boolean; isLive: boolean;
}) {
  if (hasError) {
    return (
      <span style={{
        width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(248,81,73,0.15)', color: '#f85149',
      }}>
        <AlertIcon size={13} />
      </span>
    );
  }
  if (isDone) {
    return (
      <span style={{
        width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(63,185,80,0.15)', color: '#3fb950',
      }}>
        <CheckIcon size={13} />
      </span>
    );
  }
  if (isActive) {
    return (
      <span
        className={isLive ? 'tb-spin' : ''}
        style={{
          width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
          border: '2px solid rgba(88,166,255,0.25)',
          borderTopColor: '#58a6ff',
          boxSizing: 'border-box',
        }}
      />
    );
  }
  return (
    <span style={{
      width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
      border: '1.5px solid var(--tb-border-2)',
    }} />
  );
}

/** Action card — the shared TimelineCard, whose leading dot doubles as the
 *  status indicator inside a task block. */
function ActionCardRow({ node, expandedId, onToggle }: {
  node: TimelineNode; expandedId: string | null; onToggle: (id: string) => void;
}) {
  return (
    <TimelineCard
      node={node}
      expanded={expandedId === node.id}
      onToggle={onToggle}
    />
  );
}

function ControlButton({ onClick, active, activeColor, icon, label }: {
  onClick: () => void; active: boolean; activeColor: string;
  icon: React.ReactNode; label: string;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 7,
        background: active ? `${activeColor}1f` : hovered ? 'var(--tb-surface-2)' : 'var(--tb-surface)',
        border: `1px solid ${active ? activeColor : 'var(--tb-border-2)'}`,
        borderRadius: 8,
        color: active ? activeColor : 'var(--tb-text)',
        fontSize: 13, fontWeight: 600, fontFamily: 'var(--tb-ui-font)',
        padding: '8px 16px', cursor: 'pointer',
        transition: 'background 0.1s, border-color 0.1s, color 0.1s',
      }}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

/** "TOOL path" entries for a collapsed done task — one per file-bearing action,
 *  first-seen order, deduped. Bash and other fileless calls are omitted. */
function collapsedSummary(group: ChapterTaskGroup): { tool: string; path: string }[] {
  const out: { tool: string; path: string }[] = [];
  const seen = new Set<string>();
  for (const node of group.nodes) {
    const items = node.isBatch && node.batchItems ? node.batchItems : [node];
    for (const item of items) {
      const raw = (item.toolInput?.file_path ?? item.toolInput?.path ??
                   item.toolInput?.notebook_path) as string | undefined;
      if (!raw) continue;
      const path = raw.split('/').filter(Boolean).slice(-2).join('/');
      const key  = `${node.toolName}:${path}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ tool: node.toolName.toUpperCase(), path });
    }
  }
  return out;
}

function HeaderStat({ value, label, color }: { value: string; label: string; color?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{
        fontSize: 17, fontWeight: 650,
        fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
        color: color ?? 'var(--tb-text)',
        lineHeight: 1.05,
      }}>{value}</span>
      <span style={{
        fontSize: 10, fontWeight: 500,
        color: 'var(--tb-text-muted)',
        letterSpacing: '0.02em',
      }}>{label}</span>
    </div>
  );
}
