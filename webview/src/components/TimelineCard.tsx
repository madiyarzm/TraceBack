import { memo, useRef, useState } from 'react';
import DiffViewer from './DiffViewer';
import { parseToolPayload, formatBytes, summarizeOutput, copyText } from '../payloadParser';
import { AlertIcon, CheckIcon, ChevronIcon, CopyIcon, FileIcon, ListChecksIcon, PencilIcon } from './Icons';
import ScrambleText from './ScrambleText';

export type NodeStatus = 'pending' | 'success' | 'error' | 'thinking';

export interface TimelineBatchItem {
  label:       string;
  detail?:     string;
  status:      NodeStatus;
  durationMs?: number;
  toolInput?:  Record<string, unknown>;
  intent?:     string;
}

export interface TimelineNode {
  id:          string;
  toolName:    string;
  status:      NodeStatus;
  label:       string;
  count:       number;
  detail?:     string;
  toolInput?:  Record<string, unknown>;
  eventIds?:   string[];
  timestamp:   number;
  durationMs?: number;
  isBatch?:    boolean;
  batchItems?: TimelineBatchItem[];
  /** In-progress plan item this call ran under. */
  objective?:  string;
  /** TodoWrite call — rendered as a quiet "plan updated" divider. */
  isPlanUpdate?: boolean;
  /** Why the agent made this call — first sentence of the assistant text
   *  preceding it in the transcript. Optional everywhere; may arrive late. */
  intent?:     string;
}

export const STATUS_COLOR: Record<NodeStatus, string> = {
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

/** Tool-type accent color for the pill (independent of run status). */
const TOOL_COLOR: Record<string, string> = {
  Read: '#58a6ff', Grep: '#58a6ff', Glob: '#58a6ff', LS: '#58a6ff', NotebookRead: '#58a6ff',
  Bash: '#3fb950',
  Edit: '#d29922', Write: '#d29922', MultiEdit: '#d29922', NotebookEdit: '#d29922', FileWrite: '#d29922',
  WebSearch: '#a371f7', WebFetch: '#a371f7',
  Agent: '#a371f7',
};

function toolColor(name: string): string {
  return TOOL_COLOR[name] ?? '#7d8590';
}

export function formatDuration(ms?: number): string | null {
  if (ms === undefined || ms <= 0) return null;
  if (ms < 1000)   return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

interface Props {
  node:     TimelineNode;
  expanded: boolean;
  /** True when this card's events are implicated in the current anomaly. */
  flagged?: boolean;
  /** Set when this card was part of a PAST anomaly — permanent evidence tag. */
  historyReason?: string;
  onToggle: (id: string) => void;
}

function TimelineCard({ node, expanded, flagged, historyReason, onToggle }: Props) {
  const [hovered, setHovered] = useState(false);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleMouseEnter() {
    hoverTimer.current = setTimeout(() => setHovered(true), 30);
  }
  function handleMouseLeave() {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    setHovered(false);
  }

  // User prompts render as a distinct "speech" block — the conversation's
  // structure (prompt → actions → prompt) is what makes exports shareable.
  if (node.toolName === '__prompt__') {
    return (
      <div
        onClick={() => onToggle(node.id)}
        style={{
          margin: '6px 0 4px 26px',
          padding: '7px 12px',
          borderRadius: 6,
          background: 'rgba(88,166,255,0.07)',
          border: '1px solid rgba(88,166,255,0.25)',
          borderLeft: '3px solid var(--tb-blue)',
          cursor: 'pointer',
          fontFamily: 'var(--tb-ui-font)',
        }}
      >
        <div style={{
          fontSize: 8.5, fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          color: 'var(--tb-blue)',
          marginBottom: 3,
        }}>
          You
        </div>
        <div style={{
          fontSize: 11.5, lineHeight: 1.5,
          color: 'var(--tb-text)',
          ...(expanded ? { whiteSpace: 'pre-wrap', wordBreak: 'break-word' } : {
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }),
        }}>
          {expanded ? (node.detail ?? node.label) : node.label}
        </div>
      </div>
    );
  }

  // TodoWrite rows render as a quiet "plan updated" divider — the plan itself
  // lives in the ObjectiveHeader, so a full card here would be noise.
  if (node.isPlanUpdate) {
    const input = node.toolInput ?? {};
    let text = 'plan updated';
    if (node.toolName === 'TaskCreate' && typeof input.subject === 'string') {
      text = `task added — ${input.subject}`;
    } else if (node.toolName === 'TaskUpdate') {
      text = typeof input.status === 'string'
        ? `task ${String(input.status).replace('_', ' ')}`
        : 'task updated';
    } else if (Array.isArray(input.todos)) {
      const n = (input.todos as unknown[]).length;
      text = `plan updated — ${n} task${n === 1 ? '' : 's'}`;
    }
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 7,
        padding: '3px 0 3px 26px',
        color: 'var(--tb-text-dim)',
        fontSize: 9.5, fontFamily: 'var(--tb-ui-font)',
        userSelect: 'none',
      }}>
        <ListChecksIcon size={10} />
        <span style={{
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>{text}</span>
      </div>
    );
  }

  // "Thinking" rows render as a quiet divider, not a card
  if (node.toolName === '__thinking__') {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '2px 0 2px 26px',
        color: 'rgba(88,166,255,0.35)',
        fontSize: 9.5, fontFamily: 'var(--tb-ui-font)',
        userSelect: 'none',
      }}>
        <span style={{ letterSpacing: 3 }}>· · ·</span>
      </div>
    );
  }

  const color    = STATUS_COLOR[node.status];
  const tcolor   = toolColor(node.toolName);
  const icon     = TOOL_ICON[node.toolName] ?? '·';
  const duration = formatDuration(node.durationMs);
  const isError  = node.status === 'error';

  return (
    <div
      style={{ display: 'flex', gap: 10, position: 'relative' }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Timeline dot, sitting on the vertical line drawn by the parent */}
      <div style={{
        width: 9, height: 9, borderRadius: '50%',
        background: color,
        border: '2px solid var(--tb-bg, #07090d)',
        marginTop: 11,
        flexShrink: 0,
        zIndex: 1,
        ...(node.status === 'pending'
          ? { animation: 'pendingPulse 1.6s ease-in-out infinite' } : {}),
      }} />

      {/* Card */}
      <div
        onClick={() => onToggle(node.id)}
        style={{
          flex: 1,
          minWidth: 0,
          background: isError
            ? (hovered ? 'rgba(248,81,73,0.1)' : 'rgba(248,81,73,0.06)')
            : hovered ? 'var(--tb-surface-2)' : 'var(--tb-surface)',
          border: `1px solid ${
            flagged ? 'rgba(248,81,73,0.7)'
            : isError ? 'rgba(248,81,73,0.45)'
            : hovered ? 'var(--tb-border-2)'
            : 'var(--tb-border)'
          }`,
          borderLeft: `3px solid ${flagged ? '#f85149' : color}`,
          borderRadius: 4,
          cursor: 'pointer',
          fontFamily: 'var(--tb-ui-font)',
          transition: 'background 0.12s, border-color 0.12s',
          ...(flagged ? {
            boxShadow: '0 0 10px rgba(248,81,73,0.35)',
            animation: 'stumbleHalo 1.2s ease-in-out infinite',
          } : {}),
        }}
      >
        {/* ── Permanent anomaly evidence strip (survives recovery) ── */}
        {historyReason && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '3px 10px',
            borderBottom: '1px solid rgba(210,153,34,0.3)',
            background: flagged ? 'rgba(248,81,73,0.12)' : 'rgba(210,153,34,0.08)',
            color: flagged ? '#ffa198' : '#d29922',
            fontSize: 9.5, fontWeight: 600,
          }}>
            <AlertIcon size={11} />
            <span style={{
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {historyReason}
            </span>
          </div>
        )}

        {/* ── Compact header row ── */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '7px 10px',
        }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 4, flexShrink: 0,
            fontSize: 9.5, fontWeight: 700,
            letterSpacing: '0.04em', textTransform: 'uppercase',
            color: tcolor,
            background: `${tcolor}1f`,
            border: `1px solid ${tcolor}3d`,
            borderRadius: 4, padding: '2px 7px',
          }}>
            <span style={{ fontSize: 10 }}>{icon}</span>
            {node.toolName}
          </span>
          <span style={{
            fontSize: 11.5, flex: 1, minWidth: 0,
            color: isError ? '#ffa198' : 'var(--tb-text)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            <ScrambleText text={node.isBatch ? `${node.count} steps` : node.label} />
          </span>

          {node.count > 1 && !node.isBatch && (
            <span style={{
              fontSize: 9, fontWeight: 600, flexShrink: 0,
              color: 'var(--tb-text-muted)',
              background: 'var(--tb-surface-2)',
              border: '1px solid var(--tb-border)',
              borderRadius: 3, padding: '0 4px', lineHeight: '14px',
            }}>×{node.count}</span>
          )}
          {node.isBatch && (
            <span style={{
              fontSize: 9, fontWeight: 600, flexShrink: 0,
              color, background: `${color}18`,
              border: `1px solid ${color}30`,
              borderRadius: 3, padding: '0 5px', lineHeight: '14px',
            }}>×{node.count}</span>
          )}

          {duration && (
            <span style={{
              fontSize: 9.5, flexShrink: 0,
              fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
              color: (node.durationMs ?? 0) > 10_000 ? '#d29922' : 'var(--tb-text-muted)',
            }}>{duration}</span>
          )}

          <span style={{
            color: 'var(--tb-text-dim)', flexShrink: 0,
            display: 'flex',
            transform: expanded ? 'rotate(180deg)' : 'none',
            transition: 'transform 0.15s',
          }}>
            <ChevronIcon size={11} />
          </span>
        </div>

        {/* ── Intent subtitle: why the agent made this call ── */}
        {node.intent && !node.isBatch && (
          <div style={{
            padding: '0 10px 6px 28px',
            fontSize: 10.5, lineHeight: 1.4,
            color: 'var(--tb-text-muted)',
            overflow: 'hidden', textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
          }}>
            {node.intent}
          </div>
        )}

        {/* ── Expanded accordion body ── */}
        {expanded && (
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              borderTop: '1px solid var(--tb-border)',
              padding: '8px 10px',
              display: 'flex', flexDirection: 'column', gap: 8,
              cursor: 'default',
              animation: 'cardBodyIn 0.15s ease-out',
            }}
          >
            {node.isBatch && node.batchItems && (
              <div>
                <SectionLabel>steps</SectionLabel>
                {node.batchItems.map((item, i) => (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'center', gap: 7, padding: '2px 0',
                  }}>
                    <span style={{
                      width: 5, height: 5, borderRadius: '50%',
                      background: STATUS_COLOR[item.status], flexShrink: 0,
                    }} />
                    <span style={{
                      fontSize: 11, color: 'var(--tb-text)', flex: 1, minWidth: 0,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>{item.label}</span>
                    {formatDuration(item.durationMs) && (
                      <span style={{
                        fontSize: 9, color: 'var(--tb-text-muted)',
                        fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                      }}>{formatDuration(item.durationMs)}</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            <CuratedBody node={node} />
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Curated expanded view: a deterministic parser (payloadParser.ts) classifies
 * the call, and each category gets a purpose-built panel instead of a raw
 * JSON/log dump. The unedited payload stays one click away in the raw panel.
 */
function CuratedBody({ node }: { node: TimelineNode }) {
  const [showRaw, setShowRaw] = useState(false);
  const parsed  = parseToolPayload(node.toolName, node.toolInput, node.detail);
  const isError = node.status === 'error';
  const hasRaw  = !!node.detail || !!(node.toolInput && Object.keys(node.toolInput).length);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* ── Bash: command trace + exit pill, logs masked ── */}
      {parsed.kind === 'bash' && parsed.command && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: '#0d1117',
          border: '1px solid var(--tb-border)',
          borderRadius: 4, padding: '6px 10px',
        }}>
          <span style={{
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
            fontSize: 10.5, color: '#c9d1d9',
            flex: 1, minWidth: 0,
            wordBreak: 'break-all',
          }}>
            <span style={{ color: 'var(--tb-text-muted)', userSelect: 'none' }}>$ </span>
            {parsed.command}
          </span>
          <CopyBtn getText={() => parsed.command ?? ''} title="copy command" />
          <Pill color={isError ? '#f85149' : '#3fb950'}>
            {isError ? 'failed' : 'exit 0'}
          </Pill>
        </div>
      )}

      {/* ── Outcome: deterministic one-line "what happened" ── */}
      {(() => {
        const outcome = summarizeOutput(node.detail, isError);
        if (!outcome) return null;
        return (
          <div style={{
            display: 'flex', alignItems: 'baseline', gap: 6,
            fontSize: 10.5, lineHeight: 1.45,
            color: isError ? '#ffa198' : 'var(--tb-text)',
            wordBreak: 'break-word',
          }}>
            <span style={{ color: 'var(--tb-text-dim)', flexShrink: 0, userSelect: 'none' }}>→</span>
            <span>{outcome}</span>
          </div>
        );
      })()}

      {/* ── File read: path + size metrics, contents masked ── */}
      {parsed.kind === 'file-read' && (
        <FileRow icon={<FileIcon size={11} />} path={parsed.filePath} verb="read"
                 lines={parsed.lines} bytes={parsed.bytes} />
      )}

      {/* ── File write/edit: metrics + the diff (the curated view for mods) ── */}
      {parsed.kind === 'file-write' && (
        <>
          <FileRow icon={<PencilIcon size={11} />} path={parsed.filePath} verb="written"
                   lines={parsed.lines} bytes={parsed.bytes} />
          {renderDiff(node)}
        </>
      )}

      {/* ── Web: query / domain chips ── */}
      {parsed.kind === 'web' && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
          {parsed.query  && <Chip>⌕ {parsed.query}</Chip>}
          {parsed.domain && <Chip>↗ {parsed.domain}</Chip>}
          {!parsed.query && !parsed.domain && parsed.url && <Chip>↗ {parsed.url}</Chip>}
        </div>
      )}

      {parsed.kind === 'generic' && !hasRaw && (
        <span style={{ fontSize: 10.5, color: 'var(--tb-text-dim)' }}>
          no recorded input/output for this step
        </span>
      )}

      {/* ── Raw panel fallback ── */}
      {hasRaw && (
        <div>
          <div style={{ display: 'flex', gap: 5 }}>
            <button
              onClick={() => setShowRaw((v) => !v)}
              style={{
                background: 'none',
                border: '1px solid var(--tb-border)',
                borderRadius: 3,
                color: 'var(--tb-text-muted)',
                fontSize: 9.5,
                fontFamily: 'var(--tb-ui-font)',
                padding: '2px 8px',
                cursor: 'pointer',
              }}
            >
              {showRaw ? 'hide' : 'view'} raw output
            </button>
            <CopyBtn getText={() => rawText(node)} title="copy raw input/output" label="copy" />
          </div>

          {showRaw && (
            <pre style={{
              margin: '6px 0 0',
              background: '#0a0e14',
              border: '1px solid #21262d',
              borderRadius: 4,
              padding: 8,
              fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
              fontSize: 10,
              lineHeight: 1.5,
              color: '#7ee787',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              maxHeight: 144,
              overflowY: 'auto',
            }}>
              {rawText(node)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function rawText(node: TimelineNode): string {
  let out = '';
  if (node.toolInput && Object.keys(node.toolInput).length > 0) {
    out += `── arguments ──\n${JSON.stringify(node.toolInput, null, 2)}\n\n`;
  }
  if (node.detail) out += `── output ──\n${node.detail}`;
  return out;
}

/** Small copy button with transient ✓ feedback. */
function CopyBtn({ getText, title, label }: {
  getText: () => string; title: string; label?: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      title={title}
      onClick={async (e) => {
        e.stopPropagation();
        if (await copyText(getText())) {
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        }
      }}
      style={{
        background: 'none',
        border: '1px solid var(--tb-border)',
        borderRadius: 3,
        color: copied ? '#3fb950' : 'var(--tb-text-muted)',
        fontSize: 9.5,
        fontFamily: 'var(--tb-ui-font)',
        padding: '2px 7px',
        cursor: 'pointer',
        flexShrink: 0,
        display: 'flex', alignItems: 'center', gap: 4,
      }}
    >
      {copied
        ? <CheckIcon size={10} />
        : <>{label ? <>{label}</> : null}<CopyIcon size={10} /></>}
    </button>
  );
}

function renderDiff(node: TimelineNode) {
  const input = node.toolInput;
  if (!input) return null;
  const filePath = (input.file_path ?? input.path) as string | undefined;

  if (node.toolName === 'Edit' &&
      typeof input.old_string === 'string' && typeof input.new_string === 'string') {
    return <DiffViewer oldText={input.old_string} newText={input.new_string} filePath={filePath} />;
  }
  if ((node.toolName === 'Write' || node.toolName === 'FileWrite' || node.toolName === 'NotebookEdit') &&
      typeof input.content === 'string') {
    return <DiffViewer oldText="" newText={input.content} filePath={filePath} />;
  }
  if (node.toolName === 'MultiEdit' && Array.isArray(input.edits)) {
    const edits = input.edits as { old_string?: string; new_string?: string }[];
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {edits.map((e, i) => (
          <DiffViewer key={i} oldText={e.old_string ?? ''} newText={e.new_string ?? ''}
                      filePath={i === 0 ? filePath : undefined} />
        ))}
      </div>
    );
  }
  return null;
}

function FileRow({ icon, path, verb, lines, bytes }: {
  icon: React.ReactNode; path?: string; verb: string; lines?: number; bytes?: number;
}) {
  const metrics = [
    lines !== undefined ? `${lines} lines` : null,
    formatBytes(bytes),
  ].filter(Boolean).join(' · ');

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 7,
      background: 'var(--tb-surface-2)',
      border: '1px solid var(--tb-border)',
      borderRadius: 4, padding: '5px 10px',
    }}>
      <span style={{ flexShrink: 0, display: 'flex', color: 'var(--tb-text-muted)' }}>{icon}</span>
      <span style={{
        fontSize: 10.5,
        fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
        color: 'var(--tb-text)',
        flex: 1, minWidth: 0,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        direction: 'rtl', textAlign: 'left',
      }}>
        {path ?? 'unknown file'}
      </span>
      {metrics && (
        <span style={{ fontSize: 9.5, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
          {metrics} {verb}
        </span>
      )}
    </div>
  );
}

function Pill({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span style={{
      flexShrink: 0,
      fontSize: 9, fontWeight: 600,
      color,
      background: `${color}18`,
      border: `1px solid ${color}40`,
      borderRadius: 3,
      padding: '1px 6px',
      lineHeight: '14px',
    }}>
      {children}
    </span>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontSize: 10,
      color: 'var(--tb-text)',
      background: 'var(--tb-surface-2)',
      border: '1px solid var(--tb-border)',
      borderRadius: 10,
      padding: '2px 9px',
      maxWidth: '100%',
      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
    }}>
      {children}
    </span>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 9, fontWeight: 600,
      color: 'var(--tb-text-dim)',
      letterSpacing: '0.08em', textTransform: 'uppercase',
      marginBottom: 4,
    }}>{children}</div>
  );
}

export default memo(TimelineCard);
