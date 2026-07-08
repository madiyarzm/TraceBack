import { useMemo, useState } from 'react';
import TimelineCard, { TimelineNode, formatDuration } from './TimelineCard';
import { ChapterTaskGroup, computePhaseBlocks, pendingPlanItems, PhaseBlock, PromptChapter } from '../chapters';
import { summarizeVerification, verifyChanges } from '../fileChanges';
import { computeMetrics, formatTokens } from '../metrics';
import { AnomalyStateUI } from './SessionOdometer';
import {
  AlertIcon, CheckIcon, ChevronIcon, ClockIcon, DotsIcon, FileIcon,
  PauseIcon, PencilIcon, PlayIcon, TerminalIcon,
} from './Icons';

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
  // Evidence line: changed files never exercised after their last edit. Only
  // shown once the turn is over — mid-flight everything is trivially unverified.
  const verifyLine = useMemo(
    () => (isLive ? '' : summarizeVerification(verifyChanges(chapter.nodes))),
    [chapter.nodes, isLive],
  );

  const showControls = isLive && (onPauseToggle || onRedirect);
  const [redirectOpen, setRedirectOpen] = useState(false);
  const [redirect, setRedirect] = useState('');

  // Live edge: the most recent action card gets a one-shot arrival flash.
  const newestId = isLive
    ? [...chapter.nodes].reverse().find(
        (n) => !n.toolName.startsWith('__') && !n.isPlanUpdate)?.id ?? null
    : null;

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
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: 12,
          flexWrap: 'wrap',
        }}>
          <div style={{
            flex: '1 1 180px', minWidth: 140,
            fontSize: 16, fontWeight: 650, lineHeight: 1.35,
            color: 'var(--tb-text)',
            whiteSpace: 'pre-wrap', overflowWrap: 'anywhere',
          }}>
            “{chapter.text}”
          </div>

          {showControls && (
            <div style={{ display: 'flex', gap: 8, flexShrink: 0, marginLeft: 'auto' }}>
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
                fontSize: 13, padding: '8px 11px',
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
                fontSize: 13, fontWeight: 600,
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
            <TaskSteps plan={chapter.plan} done={done} total={total} isLive={isLive} />
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

        {/* ── Verification evidence line ── */}
        {verifyLine && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            marginTop: 12,
            fontSize: 12.5, color: '#d29922',
          }}>
            <AlertIcon size={12} />
            <span>{verifyLine}</span>
          </div>
        )}
      </div>

      {/* ── Body ── */}
      <div style={{ padding: '14px 16px 28px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* ── Live anomaly banner ── */}
        {anomaly?.isAnomalous && (
          <AnomalyBanner anomaly={anomaly} paused={paused} onPauseToggle={onPauseToggle} />
        )}

        {chapter.nodes.length === 0 && (
          <div style={{ fontSize: 13, color: 'var(--tb-text-dim)', padding: '8px 2px' }}>
            {isLive ? 'Waiting — no actions for this prompt yet.' : 'No actions recorded for this prompt.'}
          </div>
        )}

        {chapter.taskGroups.map((g, i) => (
          <TaskBlock
            key={i}
            group={g}
            isLast={i === chapter.taskGroups.length - 1}
            isLive={isLive && isLast}
            newestId={newestId}
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
            fontSize: 13.5,
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
            <span style={{ fontSize: 11, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              waiting
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Segmented task steps ──────────────────────────────────────────────────────

/**
 * Discrete progress, never a percent bar: agent tasks have no honest
 * denominator (sizes differ, plans grow mid-flight). Each segment claims only
 * what is true — done, active, or pending — and a plan that grows appends a
 * hollow segment instead of moving anything backwards.
 */
function TaskSteps({ plan, done, total, isLive }: {
  plan: PromptChapter['plan']; done: number; total: number; isLive: boolean;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, alignSelf: 'flex-end' }}>
      <div style={{ display: 'flex', gap: 3 }}>
        {plan.map((p, i) => {
          const isDone   = p.status === 'completed';
          const isActive = p.status === 'in_progress';
          return (
            <span
              key={i}
              className={isActive && isLive ? 'tb-seg-active' : ''}
              title={p.content}
              style={{
                width: 16, height: 5, borderRadius: 3,
                background: isDone ? 'var(--tb-green)' : isActive ? 'var(--tb-blue)' : 'transparent',
                border: `1px solid ${isDone ? 'var(--tb-green)' : isActive ? 'var(--tb-blue)' : 'var(--tb-border-2)'}`,
                boxSizing: 'border-box',
              }}
            />
          );
        })}
      </div>
      <span style={{
        fontSize: 12,
        color: done === total ? 'var(--tb-green)' : 'var(--tb-text-muted)',
      }}>
        {done} of {total} tasks
      </span>
    </div>
  );
}

// ── Live anomaly banner ───────────────────────────────────────────────────────

function AnomalyBanner({ anomaly, paused, onPauseToggle }: {
  anomaly: AnomalyStateUI; paused?: boolean; onPauseToggle?: () => void;
}) {
  // A stall is "Claude is waiting on you", not misbehavior — quiet notice,
  // no alarm styling, no pause button (pausing a waiting agent is meaningless).
  if (anomaly.type === 'stall') {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '12px 15px',
        borderRadius: 10,
        border: '1px solid var(--tb-border-2)',
        background: 'var(--tb-surface)',
      }}>
        <span style={{
          width: 32, height: 32, borderRadius: 8, flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--tb-surface-2)', color: 'var(--tb-blue)',
        }}>
          <ClockIcon size={16} />
        </span>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 13.5, fontWeight: 650, color: 'var(--tb-text)', marginBottom: 2 }}>
            {anomaly.title ?? 'Waiting'}
          </div>
          <div style={{ fontSize: 12.5, lineHeight: 1.5, color: 'var(--tb-text-muted)', wordBreak: 'break-word' }}>
            {anomaly.description ?? anomaly.reason}
          </div>
        </div>
      </div>
    );
  }

  const high  = (anomaly.severity ?? 'high') === 'high';
  const color = high ? '#f85149' : '#d29922';
  const Icon  = AlertIcon;
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
          fontSize: 13.5, lineHeight: 1.5,
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
            fontSize: 14, fontWeight: 600, fontFamily: 'var(--tb-ui-font)',
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

function TaskBlock({ group, isLast, isLive, newestId, expandedId, onToggle }: {
  group:      ChapterTaskGroup;
  /** Whether this is the chapter's final group — the live edge. */
  isLast:     boolean;
  isLive:     boolean;
  newestId:   string | null;
  expandedId: string | null;
  onToggle:   (id: string) => void;
}) {
  // Task state comes ONLY from the todo/task tools; a failed action inside the
  // group shows on its own card and never demotes the task itself.
  const isDone   = group.status === 'completed';
  const isActive = group.status === 'in_progress';
  const [open, setOpen] = useState(!isDone);

  // Unattributed actions (no task was in_progress when they ran) are still
  // CONTAINED — phase blocks, never bare floating cards. This covers pre-plan
  // exploration and sessions where the agent skipped in_progress updates.
  if (!group.objective) {
    const blocks = computePhaseBlocks(group.nodes);
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {blocks.map((b, i) => (
          <PhaseBlockRow
            key={i}
            block={b}
            defaultOpen={isLive && isLast && i === blocks.length - 1}
            newestId={newestId}
            expandedId={expandedId}
            onToggle={onToggle}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={isActive && isLive ? 'tb-task-active' : ''}
      style={{
        border: `1px solid ${isActive ? 'var(--tb-border-2)' : 'var(--tb-border)'}`,
        borderRadius: 10,
        background: 'var(--tb-surface)',
        overflow: 'hidden',
      }}
    >
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '13px 15px',
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <StatusMark isDone={isDone} isActive={isActive} isLive={isLive} />
        <span style={{
          fontSize: 14.5, fontWeight: isActive ? 650 : 550,
          color: isDone ? 'var(--tb-text-muted)' : 'var(--tb-text)',
          flex: 1, minWidth: 0,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {group.objective}
        </span>
        {group.errorCount > 0 && (
          <span style={{ fontSize: 12.5, color: '#f85149', flexShrink: 0 }}>
            {group.errorCount} error{group.errorCount === 1 ? '' : 's'}
          </span>
        )}
        {isActive ? (
          <span style={{ fontSize: 12.5, color: 'var(--tb-blue)', flexShrink: 0 }}>
            active · {group.nodes.length} action{group.nodes.length === 1 ? '' : 's'}
          </span>
        ) : (
          <span style={{ fontSize: 12.5, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
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
          fontSize: 12,
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
            <ActionCardRow key={n.id} node={n} highlight={n.id === newestId}
                           expandedId={expandedId} onToggle={onToggle} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Phase block (fallback grouping when the agent made no plan) ──────────────

/** Verb + glyph per phase — same card anatomy as TaskBlock, but a tool glyph
 *  where the status ring would be: phases are activity, not declared tasks,
 *  and must never fake a todo state. */
const PHASE_STYLE: Record<PhaseBlock['kind'], { verb: string; icon: React.ReactNode }> = {
  reading: { verb: 'Read',    icon: <FileIcon size={12} /> },
  editing: { verb: 'Edited',  icon: <PencilIcon size={12} /> },
  running: { verb: 'Ran',     icon: <TerminalIcon size={12} /> },
  actions: { verb: 'Actions', icon: <DotsIcon size={12} /> },
};

function PhaseBlockRow({ block, defaultOpen, newestId, expandedId, onToggle }: {
  block:       PhaseBlock;
  defaultOpen: boolean;
  newestId:    string | null;
  expandedId:  string | null;
  onToggle:    (id: string) => void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const duration = formatDuration(block.durationMs);
  const ps = PHASE_STYLE[block.kind];

  return (
    <div style={{
      border: '1px solid var(--tb-border)',
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
        <span style={{
          width: 22, height: 22, borderRadius: 6, flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--tb-surface-2)',
          color: 'var(--tb-text-muted)',
        }}>
          {ps.icon}
        </span>
        <span style={{
          fontSize: 14.5, fontWeight: 550,
          color: 'var(--tb-text)',
          flex: 1, minWidth: 0,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {block.kind === 'actions' || !block.summary ? (
            block.label
          ) : (
            <>
              <span style={{ color: 'var(--tb-text-muted)' }}>{ps.verb} </span>
              <span style={{ fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)', fontSize: 13.5 }}>
                {block.summary}
              </span>
            </>
          )}
        </span>
        {block.errorCount > 0 && (
          <span style={{ fontSize: 12.5, color: '#f85149', flexShrink: 0 }}>
            {block.errorCount} error{block.errorCount === 1 ? '' : 's'}
          </span>
        )}
        <span style={{ fontSize: 12.5, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
          {block.actionCount} action{block.actionCount === 1 ? '' : 's'}
          {duration ? ` · ${duration}` : ''}
        </span>
        <span style={{
          color: 'var(--tb-text-dim)', display: 'flex', flexShrink: 0,
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.15s',
        }}>
          <ChevronIcon size={13} />
        </span>
      </div>

      {open && (
        <div style={{
          borderTop: '1px solid var(--tb-border)',
          padding: '12px 14px',
          display: 'flex', flexDirection: 'column', gap: 8,
          animation: 'cardBodyIn 0.15s ease-out',
        }}>
          {block.nodes.map((n) => (
            <ActionCardRow key={n.id} node={n} highlight={n.id === newestId}
                           expandedId={expandedId} onToggle={onToggle} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Round status marker: green check (done), blue spinner (active), or an empty
 *  ring (pending). Reflects the todo status only — action errors show on their
 *  own cards, never here. */
function StatusMark({ isDone, isActive, isLive }: {
  isDone: boolean; isActive: boolean; isLive: boolean;
}) {
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
 *  status indicator inside a task block. The newest live card flashes in. */
function ActionCardRow({ node, highlight = false, expandedId, onToggle }: {
  node: TimelineNode; highlight?: boolean;
  expandedId: string | null; onToggle: (id: string) => void;
}) {
  return (
    <div className={highlight ? 'tb-node-arrive' : ''}>
      <TimelineCard
        bare
        node={node}
        expanded={expandedId === node.id}
        onToggle={onToggle}
      />
    </div>
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
        fontSize: 14, fontWeight: 600, fontFamily: 'var(--tb-ui-font)',
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
        fontSize: 11, fontWeight: 500,
        color: 'var(--tb-text-muted)',
        letterSpacing: '0.02em',
      }}>{label}</span>
    </div>
  );
}
