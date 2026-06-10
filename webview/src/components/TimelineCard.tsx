import { memo } from 'react';
import DiffViewer from './DiffViewer';

export type NodeStatus = 'pending' | 'success' | 'error' | 'thinking';

export interface TimelineBatchItem {
  label:       string;
  detail?:     string;
  status:      NodeStatus;
  durationMs?: number;
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

export function formatDuration(ms?: number): string | null {
  if (ms === undefined || ms <= 0) return null;
  if (ms < 1000)   return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

const MONO_BLOCK: React.CSSProperties = {
  background:   '#0d1117',
  border:       '1px solid var(--tb-border)',
  borderRadius: 4,
  padding:      '8px 10px',
  fontFamily:   'var(--tb-mono-font, ui-monospace, monospace)',
  fontSize:     10.5,
  lineHeight:   1.5,
  color:        '#c9d1d9',
  whiteSpace:   'pre-wrap',
  wordBreak:    'break-word',
  overflowY:    'auto',
  maxHeight:    220,
  margin:       0,
};

interface Props {
  node:     TimelineNode;
  expanded: boolean;
  /** True when this card's events are implicated in the current anomaly. */
  flagged?: boolean;
  onToggle: (id: string) => void;
}

function TimelineCard({ node, expanded, flagged, onToggle }: Props) {
  // "Thinking" rows render as a quiet divider, not a card
  if (node.toolName === '__thinking__') {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '2px 0 2px 26px',
        color: 'rgba(88,166,255,0.55)',
        fontSize: 10, fontFamily: 'var(--tb-ui-font)',
        userSelect: 'none',
      }}>
        <span style={{ letterSpacing: 2 }}>· · ·</span>
        <span>thinking</span>
      </div>
    );
  }

  const color    = STATUS_COLOR[node.status];
  const icon     = TOOL_ICON[node.toolName] ?? '·';
  const duration = formatDuration(node.durationMs);
  const isError  = node.status === 'error';

  return (
    <div style={{ display: 'flex', gap: 10, position: 'relative' }}>
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
          background: isError ? 'rgba(248,81,73,0.06)' : 'var(--tb-surface)',
          border: `1px solid ${
            flagged ? 'rgba(248,81,73,0.7)'
            : isError ? 'rgba(248,81,73,0.45)'
            : 'var(--tb-border)'
          }`,
          borderLeft: `3px solid ${flagged ? '#f85149' : color}`,
          borderRadius: 4,
          cursor: 'pointer',
          fontFamily: 'var(--tb-ui-font)',
          ...(flagged ? {
            boxShadow: '0 0 10px rgba(248,81,73,0.35)',
            animation: 'stumbleHalo 1.2s ease-in-out infinite',
          } : {}),
        }}
      >
        {/* ── Compact header row ── */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          padding: '7px 10px',
        }}>
          <span style={{ fontSize: 11, color, flexShrink: 0 }}>{icon}</span>
          <span style={{
            fontSize: 10, fontWeight: 600, flexShrink: 0,
            color: 'var(--tb-text-muted)',
            letterSpacing: '0.04em', textTransform: 'uppercase',
          }}>
            {node.toolName}
          </span>
          <span style={{
            fontSize: 11.5, flex: 1, minWidth: 0,
            color: isError ? '#ffa198' : 'var(--tb-text)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {node.isBatch ? `${node.count} steps` : node.label}
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
            fontSize: 8, color: 'var(--tb-text-dim)', flexShrink: 0,
            transform: expanded ? 'rotate(180deg)' : 'none',
            transition: 'transform 0.15s',
          }}>▼</span>
        </div>

        {/* ── Expanded accordion body ── */}
        {expanded && (
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              borderTop: '1px solid var(--tb-border)',
              padding: '8px 10px',
              display: 'flex', flexDirection: 'column', gap: 8,
              cursor: 'default',
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

            {renderArguments(node)}

            {node.detail && (
              <div>
                <SectionLabel>output</SectionLabel>
                <pre style={MONO_BLOCK}>{node.detail}</pre>
              </div>
            )}

            {!node.detail && !node.toolInput && !node.isBatch && (
              <span style={{ fontSize: 10.5, color: 'var(--tb-text-dim)' }}>
                no recorded input/output for this step
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * File-modification tools get a visual diff instead of raw JSON:
 *  - Edit:      old_string → new_string (real before/after diff)
 *  - Write:     '' → content            (all-additions diff)
 *  - MultiEdit: one diff section per entry in edits[]
 * Everything else falls back to pretty-printed JSON arguments.
 */
function renderArguments(node: TimelineNode) {
  const input = node.toolInput;
  if (!input || Object.keys(input).length === 0) return null;

  const filePath = (input.file_path ?? input.path) as string | undefined;

  if (node.toolName === 'Edit' &&
      typeof input.old_string === 'string' && typeof input.new_string === 'string') {
    return (
      <div>
        <SectionLabel>changes</SectionLabel>
        <DiffViewer oldText={input.old_string} newText={input.new_string} filePath={filePath} />
      </div>
    );
  }

  if (node.toolName === 'Write' && typeof input.content === 'string') {
    return (
      <div>
        <SectionLabel>file written</SectionLabel>
        <DiffViewer oldText="" newText={input.content} filePath={filePath} />
      </div>
    );
  }

  if (node.toolName === 'MultiEdit' && Array.isArray(input.edits)) {
    const edits = input.edits as { old_string?: string; new_string?: string }[];
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <SectionLabel>changes ({edits.length} edits)</SectionLabel>
        {edits.map((e, i) => (
          <DiffViewer
            key={i}
            oldText={e.old_string ?? ''}
            newText={e.new_string ?? ''}
            filePath={i === 0 ? filePath : undefined}
          />
        ))}
      </div>
    );
  }

  return (
    <div>
      <SectionLabel>arguments</SectionLabel>
      <pre style={MONO_BLOCK}>{JSON.stringify(input, null, 2)}</pre>
    </div>
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
