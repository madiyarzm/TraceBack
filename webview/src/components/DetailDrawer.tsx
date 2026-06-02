import { useEffect, useRef, useState } from 'react';
import vscode from '../vscodeApi';

type NodeStatus = 'pending' | 'success' | 'error' | 'thinking';

export interface DrawerPayload {
  toolName: string;
  label: string;
  status: NodeStatus;
  detail?: string;
  toolInput?: Record<string, unknown>;
  isBatch?: boolean;
  batchItems?: { label: string; detail?: string; status: NodeStatus }[];
}

interface Props {
  node: DrawerPayload | null;
  onClose: () => void;
  onChat?: (question: string) => void;
  chatAnswer?: string;
  chatLoading?: boolean;
}

const STATUS_COLOR: Record<NodeStatus, string> = {
  pending:  '#d29922',
  thinking: '#58a6ff',
  success:  '#3fb950',
  error:    '#f85149',
};

const STATUS_LABEL: Record<NodeStatus, string> = {
  pending:  'running',
  thinking: 'thinking',
  success:  'success',
  error:    'error',
};

const TOOL_ICON: Record<string, string> = {
  Read:      '↘',  Edit: '✎', Write: '✦', Bash: '$',
  WebSearch: '⌕',  WebFetch: '↗', TodoRead: '☐', TodoWrite: '☑', Agent: '◈',
};

// Strip ANSI escape codes from terminal output
function stripAnsi(s: string): string {
  return s.replace(/\x1B\[[0-9;]*[A-Za-z]/g, '').replace(/\x1B\][^\x07]*\x07/g, '');
}

// Return first N lines with truncation metadata
function smartTruncate(raw: string, maxLines = 60): { text: string; truncated: boolean; totalLines: number } {
  const cleaned = stripAnsi(raw);
  const lines   = cleaned.split('\n');
  if (lines.length <= maxLines) return { text: cleaned, truncated: false, totalLines: lines.length };
  return { text: lines.slice(0, maxLines).join('\n'), truncated: true, totalLines: lines.length };
}

// Format toolInput as readable key–value pairs, skipping noise
function formatInputPairs(input: Record<string, unknown>): Array<{ key: string; value: string }> {
  const SKIP = new Set(['description']); // already shown as the node label
  return Object.entries(input)
    .filter(([k, v]) => !SKIP.has(k) && v !== undefined && v !== null && v !== '')
    .map(([k, v]) => ({
      key: k,
      value: typeof v === 'string' ? v : JSON.stringify(v),
    }));
}

const canOpenFile = (toolName: string) =>
  toolName === 'Read' || toolName === 'Edit' || toolName === 'Write';

export default function DetailDrawer({ node, onClose, onChat, chatAnswer, chatLoading }: Props) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (node && contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
    setExpanded(false);
  }, [node]);

  // Scroll to bottom when a chat answer arrives
  useEffect(() => {
    if (chatAnswer && contentRef.current) {
      setTimeout(() => {
        contentRef.current!.scrollTop = contentRef.current!.scrollHeight;
      }, 50);
    }
  }, [chatAnswer]);

  if (!node) return null;

  const color     = STATUS_COLOR[node.status];
  const icon      = TOOL_ICON[node.toolName] ?? '·';
  const inputPairs = node.toolInput ? formatInputPairs(node.toolInput) : [];

  function handleOpenFile() {
    vscode.postMessage({ type: 'open_file', filePath: node!.label });
  }

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 0, left: 0, right: 0,
        background: 'var(--tb-surface)',
        borderTop: `1px solid ${color}40`,
        fontFamily: 'var(--tb-ui-font)',
        display: 'flex',
        flexDirection: 'column',
        maxHeight: '52%',
        boxShadow: '0 -4px 24px rgba(0,0,0,0.5)',
        zIndex: 50,
        animation: 'drawerSlideUp 0.18s cubic-bezier(0.16,1,0.3,1) both',
      }}
    >
      {/* ── Header ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '7px 12px',
        borderBottom: '1px solid var(--tb-border)',
        flexShrink: 0,
      }}>
        {/* Icon + tool */}
        <span style={{ fontSize: 13, color, lineHeight: 1 }}>{icon}</span>
        <span style={{
          fontSize: 11, fontWeight: 600,
          color: 'var(--tb-text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
        }}>
          {node.toolName}
        </span>

        {/* Separator */}
        <span style={{ color: 'var(--tb-border-2)', fontSize: 12 }}>/</span>

        {/* Label */}
        <span style={{
          fontSize: 12, color: 'var(--tb-text)',
          flex: 1, overflow: 'hidden',
          textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {node.label}
        </span>

        {/* Status pill */}
        <span style={{
          fontSize: 10, fontWeight: 600,
          color, background: `${color}18`,
          border: `1px solid ${color}30`,
          borderRadius: 3,
          padding: '1px 6px',
          flexShrink: 0,
          letterSpacing: '0.03em',
        }}>
          {STATUS_LABEL[node.status]}
        </span>

        {/* Actions */}
        <div style={{ display: 'flex', gap: 8, flexShrink: 0, alignItems: 'center' }}>
          {canOpenFile(node.toolName) && (
            <button
              onClick={handleOpenFile}
              style={{
                fontSize: 11, color: 'var(--tb-blue)',
                background: 'none', border: 'none',
                cursor: 'pointer', padding: 0,
                fontFamily: 'var(--tb-ui-font)',
              }}
            >
              open ↗
            </button>
          )}
          <button
            onClick={onClose}
            style={{
              fontSize: 14, color: 'var(--tb-text-muted)',
              background: 'none', border: 'none',
              cursor: 'pointer', padding: 0, lineHeight: 1,
              fontFamily: 'var(--tb-ui-font)',
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* ── Scrollable body ── */}
      <div ref={contentRef} style={{ overflow: 'auto', flex: 1 }}>

        {/* INPUT section — only when toolInput has useful data */}
        {inputPairs.length > 0 && (
          <Section label="INPUT">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                {inputPairs.map(({ key, value }) => (
                  <tr key={key}>
                    <td style={{
                      fontSize: 10, fontWeight: 600,
                      color: 'var(--tb-text-muted)',
                      paddingRight: 12, paddingBottom: 4,
                      verticalAlign: 'top',
                      whiteSpace: 'nowrap',
                      letterSpacing: '0.02em',
                    }}>
                      {key}
                    </td>
                    <td style={{
                      fontSize: 11,
                      color: 'var(--tb-text)',
                      paddingBottom: 4,
                      wordBreak: 'break-all',
                      fontFamily: key === 'command' || key === 'file_path' || key === 'path'
                        ? 'var(--tb-font)'
                        : 'var(--tb-ui-font)',
                    }}>
                      {value}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>
        )}

        {/* BATCH ITEMS — shown instead of OUTPUT for batch nodes */}
        {node.isBatch && node.batchItems && node.batchItems.length > 0 ? (
          <Section label={`${node.batchItems.length} STEPS`}>
            {node.batchItems.map((item, i) => (
              <BatchItemRow key={i} index={i} item={item} total={node.batchItems!.length} />
            ))}
          </Section>
        ) : (
          /* OUTPUT section */
          node.detail ? (
            <OutputSection detail={node.detail} expanded={expanded} onToggleExpand={() => setExpanded(e => !e)} />
          ) : (
            <Section label="OUTPUT">
              <span style={{ fontSize: 11, color: 'var(--tb-text-muted)', fontStyle: 'italic' }}>
                No output recorded
              </span>
            </Section>
          )
        )}

        {/* CHAT section */}
        {onChat && (
          <ChatSection
            key={`chat-${node.toolName}-${node.label}`}
            onChat={onChat}
            chatAnswer={chatAnswer}
            chatLoading={chatLoading ?? false}
          />
        )}
      </div>
    </div>
  );
}

/* ─── Sub-components ─────────────────────────────────────────── */

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ padding: '8px 12px' }}>
      <div style={{
        fontSize: 9, fontWeight: 700,
        color: 'var(--tb-text-dim)',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        marginBottom: 8,
        paddingBottom: 4,
        borderBottom: '1px solid var(--tb-border)',
      }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function OutputSection({
  detail, expanded, onToggleExpand,
}: {
  detail: string;
  expanded: boolean;
  onToggleExpand: () => void;
}) {
  const { text, truncated, totalLines } = smartTruncate(detail, expanded ? Infinity : 60);

  return (
    <Section label={`OUTPUT${totalLines > 1 ? ` · ${totalLines} lines` : ''}`}>
      <pre style={{
        margin: 0,
        fontSize: 11,
        lineHeight: 1.6,
        color: 'var(--tb-text)',
        fontFamily: 'var(--tb-font)',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        overflow: 'visible',
      }}>
        {text}
      </pre>

      {truncated && !expanded && (
        <button
          onClick={onToggleExpand}
          style={{
            marginTop: 8,
            fontSize: 10,
            color: 'var(--tb-blue)',
            background: 'none', border: 'none',
            cursor: 'pointer', padding: 0,
            fontFamily: 'var(--tb-ui-font)',
          }}
        >
          show full output ({totalLines} lines) ↓
        </button>
      )}
    </Section>
  );
}

function ChatSection({
  onChat,
  chatAnswer,
  chatLoading,
}: {
  onChat: (q: string) => void;
  chatAnswer?: string;
  chatLoading: boolean;
}) {
  const [input, setInput] = useState('');

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || chatLoading) return;
    onChat(q);
    setInput('');
  }

  return (
    <Section label="ASK">
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 6, marginBottom: chatAnswer ? 8 : 0 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this action…"
          style={{
            flex: 1,
            fontSize: 11,
            fontFamily: 'var(--tb-ui-font)',
            background: 'var(--tb-bg)',
            border: '1px solid var(--tb-border)',
            borderRadius: 3,
            color: 'var(--tb-text)',
            padding: '4px 8px',
            outline: 'none',
          }}
        />
        <button
          type="submit"
          disabled={chatLoading || !input.trim()}
          style={{
            fontSize: 10,
            fontFamily: 'var(--tb-ui-font)',
            color: chatLoading || !input.trim() ? 'var(--tb-text-dim)' : 'var(--tb-blue)',
            background: 'none',
            border: 'none',
            cursor: chatLoading ? 'wait' : 'pointer',
            padding: '0 4px',
            flexShrink: 0,
          }}
        >
          {chatLoading ? '···' : 'ask ↗'}
        </button>
      </form>
      {chatAnswer && (
        <div style={{
          fontSize: 11,
          lineHeight: 1.65,
          color: 'var(--tb-text)',
          background: 'var(--tb-bg)',
          border: '1px solid var(--tb-border)',
          borderRadius: 3,
          padding: '7px 9px',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {chatAnswer}
        </div>
      )}
    </Section>
  );
}

function BatchItemRow({
  index, item, total,
}: {
  index: number;
  item: { label: string; detail?: string; status: NodeStatus };
  total: number;
}) {
  const [open, setOpen] = useState(false);
  const color = STATUS_COLOR[item.status];

  return (
    <div style={{ marginBottom: 6 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          width: '100%', background: 'none', border: 'none',
          cursor: 'pointer', padding: 0, textAlign: 'left',
          fontFamily: 'var(--tb-ui-font)',
        }}
      >
        <span style={{
          width: 6, height: 6, borderRadius: '50%',
          background: color, flexShrink: 0,
        }} />
        <span style={{
          fontSize: 10, color: 'var(--tb-text-muted)',
          flexShrink: 0, fontWeight: 600,
        }}>
          {index + 1}/{total}
        </span>
        <span style={{
          fontSize: 11, color: 'var(--tb-text)',
          overflow: 'hidden', textOverflow: 'ellipsis',
          whiteSpace: 'nowrap', flex: 1,
        }}>
          {item.label}
        </span>
        <span style={{ fontSize: 9, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
          {open ? '▲' : '▼'}
        </span>
      </button>

      {open && item.detail && (
        <pre style={{
          margin: '6px 0 0 14px',
          padding: '6px 8px',
          background: 'var(--tb-bg)',
          borderLeft: `2px solid ${color}40`,
          fontSize: 10,
          lineHeight: 1.5,
          color: 'var(--tb-text)',
          fontFamily: 'var(--tb-font)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          maxHeight: 200,
          overflow: 'auto',
        }}>
          {stripAnsi(item.detail).split('\n').slice(0, 40).join('\n')}
        </pre>
      )}
    </div>
  );
}

