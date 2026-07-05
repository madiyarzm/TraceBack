import { useState } from 'react';
import FileChangesPanel from './FileChangesPanel';
import { TimelineNode } from './TimelineCard';
import { anomaliesFor, PromptChapter } from '../chapters';
import { AnomalyRecordUI, BuiltinGuardUI, GuardsStateUI } from '../useSessionFeed';
import { AlertIcon, ChevronIcon, ClockIcon } from './Icons';

type Tab = 'anomalies' | 'files' | 'guards';

interface Props {
  nodes:          TimelineNode[];
  chapters:       PromptChapter[];
  selectedIndex:  number;
  records:        AnomalyRecordUI[];
  filesClickable: boolean;
  guards:         GuardsStateUI;
  onSetGuard:     (key: string, enabled: boolean) => void;
  onAddGuard:     (pattern: string) => void;
  onRemoveGuard:  (pattern: string) => void;
}

/**
 * Right panel: Anomalies (per-prompt evidence trail), Files (what changed on
 * disk), Guards (the policy rules that auto-deny tool calls).
 */
export default function RightPanel(props: Props) {
  const [tab, setTab] = useState<Tab>('anomalies');
  const anomalyTotal = props.records.length;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', minHeight: 0,
      fontFamily: 'var(--tb-ui-font)',
    }}>
      <div style={{
        display: 'flex', flexShrink: 0,
        borderBottom: '1px solid var(--tb-border)',
        background: 'var(--tb-surface)',
      }}>
        <TabButton active={tab === 'anomalies'} onClick={() => setTab('anomalies')}>
          Anomalies{anomalyTotal > 0 ? ` ${anomalyTotal}` : ''}
        </TabButton>
        <TabButton active={tab === 'files'} onClick={() => setTab('files')}>
          Files
        </TabButton>
        <TabButton active={tab === 'guards'} onClick={() => setTab('guards')}>
          Guards
        </TabButton>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
        {tab === 'anomalies' && (
          <AnomaliesTab
            chapters={props.chapters}
            selectedIndex={props.selectedIndex}
            records={props.records}
          />
        )}
        {tab === 'files' && (
          <FileChangesPanel nodes={props.nodes} defaultOpen clickable={props.filesClickable} />
        )}
        {tab === 'guards' && (
          <GuardsTab
            guards={props.guards}
            onSetGuard={props.onSetGuard}
            onAddGuard={props.onAddGuard}
            onRemoveGuard={props.onRemoveGuard}
          />
        )}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        fontSize: 10.5, fontWeight: active ? 700 : 400,
        fontFamily: 'var(--tb-ui-font)',
        padding: '8px 4px',
        cursor: 'pointer',
        background: 'transparent',
        color: active ? 'var(--tb-text)' : 'var(--tb-text-muted)',
        border: 'none',
        borderBottom: `2px solid ${active ? 'var(--tb-blue)' : 'transparent'}`,
        transition: 'color 0.1s, border-color 0.1s',
      }}
    >
      {children}
    </button>
  );
}

// ─── Anomalies ────────────────────────────────────────────────────────────────

const SEV_COLOR = { high: '#f85149', medium: '#d29922' } as const;

function severityOf(r: AnomalyRecordUI): 'high' | 'medium' {
  return r.severity ?? (r.type === 'stall' ? 'medium' : 'high');
}

function AnomaliesTab({ chapters, selectedIndex, records }: {
  chapters: PromptChapter[]; selectedIndex: number; records: AnomalyRecordUI[];
}) {
  const selected = chapters.find((c) => c.index === selectedIndex);
  const current  = selected ? sortBySeverity(anomaliesFor(selected, records)) : [];
  const others   = chapters
    .filter((c) => c.index !== selectedIndex)
    .map((c) => ({ chapter: c, recs: sortBySeverity(anomaliesFor(c, records)) }))
    .filter((g) => g.recs.length > 0)
    .reverse(); // newest chapter first

  const empty = current.length === 0 && others.length === 0;

  return (
    <div style={{ padding: '10px 12px 20px' }}>
      <SectionLabel>This prompt</SectionLabel>
      {current.length === 0 ? (
        <div style={{ fontSize: 10.5, color: 'var(--tb-text-dim)', padding: '4px 2px 12px' }}>
          No anomalies in this prompt.
        </div>
      ) : (
        current.map((r, i) => <AnomalyItem key={i} record={r} />)
      )}

      {others.map(({ chapter, recs }) => (
        <HistorySection key={chapter.index} title={`P${chapter.index}`} count={recs.length}>
          {recs.map((r, i) => <AnomalyItem key={i} record={r} />)}
        </HistorySection>
      ))}

      {empty && (
        <div style={{ fontSize: 10.5, color: 'var(--tb-text-dim)', padding: '10px 2px' }}>
          Clean session so far — detections will show up here and stay as a
          permanent evidence trail.
        </div>
      )}
    </div>
  );
}

function sortBySeverity(recs: AnomalyRecordUI[]): AnomalyRecordUI[] {
  return [...recs].sort((a, b) =>
    severityOf(a) === severityOf(b)
      ? b.detectedAt - a.detectedAt
      : severityOf(a) === 'high' ? -1 : 1
  );
}

function AnomalyItem({ record }: { record: AnomalyRecordUI }) {
  const sev   = severityOf(record);
  const color = SEV_COLOR[sev];
  const title = record.title ?? record.type.replace(/_/g, ' ');
  const desc  = record.description ?? record.reason;
  const Icon  = record.type === 'stall' ? ClockIcon : AlertIcon;

  return (
    <div style={{
      border: `1px solid ${color}45`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 5,
      background: `${color}0d`,
      padding: '8px 10px',
      marginBottom: 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
        <span style={{ color, display: 'flex' }}><Icon size={12} /></span>
        <span style={{ fontSize: 11, fontWeight: 600, color }}>{title}</span>
      </div>
      <div style={{ fontSize: 10.5, lineHeight: 1.45, color: 'var(--tb-text)', wordBreak: 'break-word' }}>
        {desc}
      </div>
    </div>
  );
}

function HistorySection({ title, count, children }: {
  title: string; count: number; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: 6 }}>
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '5px 2px', cursor: 'pointer', userSelect: 'none',
        }}
      >
        <span style={{
          fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
          textTransform: 'uppercase', color: 'var(--tb-text-dim)',
        }}>
          All time · {title}
        </span>
        <span style={{ fontSize: 9, color: 'var(--tb-text-dim)' }}>({count})</span>
        <span style={{
          color: 'var(--tb-text-dim)', display: 'flex',
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.15s',
        }}>
          <ChevronIcon size={9} />
        </span>
      </div>
      {open && <div style={{ animation: 'cardBodyIn 0.15s ease-out' }}>{children}</div>}
    </div>
  );
}

// ─── Guards ───────────────────────────────────────────────────────────────────

function GuardsTab({ guards, onSetGuard, onAddGuard, onRemoveGuard }: {
  guards:        GuardsStateUI;
  onSetGuard:    (key: string, enabled: boolean) => void;
  onAddGuard:    (pattern: string) => void;
  onRemoveGuard: (pattern: string) => void;
}) {
  const [draft, setDraft] = useState('');
  const [adding, setAdding] = useState(false);

  function submit() {
    const p = draft.trim();
    if (!p) return;
    onAddGuard(p);
    setDraft('');
    setAdding(false);
  }

  return (
    <div style={{ padding: '10px 12px 20px' }}>
      <SectionLabel>Built-in</SectionLabel>
      {guards.builtins.map((g) => (
        <BuiltinRow key={g.key} guard={g} onToggle={(v) => onSetGuard(g.key, v)} />
      ))}
      {guards.builtins.length === 0 && (
        <div style={{ fontSize: 10.5, color: 'var(--tb-text-dim)', padding: '4px 2px 8px' }}>
          Loading guards…
        </div>
      )}

      <div style={{ marginTop: 14 }}>
        <SectionLabel>Custom</SectionLabel>
        {guards.custom.map((pattern) => (
          <div key={pattern} style={{
            display: 'flex', alignItems: 'center', gap: 8,
            border: '1px solid var(--tb-border)',
            borderRadius: 5,
            padding: '6px 9px',
            marginBottom: 6,
            background: 'var(--tb-surface)',
          }}>
            <span style={{
              flex: 1, minWidth: 0,
              fontSize: 10.5,
              fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
              color: 'var(--tb-blue)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              /{pattern}/
            </span>
            <button
              title="Delete this rule"
              onClick={() => onRemoveGuard(pattern)}
              style={{
                background: 'none',
                border: '1px solid var(--tb-border)',
                borderRadius: 3,
                color: 'var(--tb-text-muted)',
                fontSize: 10, padding: '1px 7px',
                cursor: 'pointer', flexShrink: 0,
              }}
            >
              ✕
            </button>
          </div>
        ))}

        {adding ? (
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submit();
                if (e.key === 'Escape') { setAdding(false); setDraft(''); }
              }}
              placeholder="regex, e.g. curl.*prod"
              style={{
                flex: 1, minWidth: 0,
                background: 'var(--tb-surface)',
                border: '1px solid var(--tb-border-2)',
                borderRadius: 3,
                color: 'var(--tb-text)',
                fontSize: 10.5,
                fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                padding: '4px 8px',
                outline: 'none',
              }}
            />
            <button
              onClick={submit}
              disabled={!draft.trim()}
              style={{
                background: 'rgba(88,166,255,0.1)',
                border: '1px solid rgba(88,166,255,0.4)',
                borderRadius: 3,
                color: 'var(--tb-blue)',
                fontSize: 10, fontWeight: 600,
                padding: '0 10px',
                cursor: draft.trim() ? 'pointer' : 'not-allowed',
              }}
            >
              add
            </button>
          </div>
        ) : (
          <button
            onClick={() => setAdding(true)}
            style={{
              width: '100%',
              background: 'transparent',
              border: '1px dashed var(--tb-border-2)',
              borderRadius: 5,
              color: 'var(--tb-text-muted)',
              fontSize: 10.5, fontFamily: 'var(--tb-ui-font)',
              padding: '7px 0',
              cursor: 'pointer',
            }}
          >
            + Add custom rule
          </button>
        )}
      </div>

      <div style={{
        marginTop: 12,
        fontSize: 9.5, lineHeight: 1.5,
        color: 'var(--tb-text-dim)',
      }}>
        A matching call is auto-denied before it runs; the rule name is sent
        back to the agent as context. Custom rules are regexes matched against
        the full tool call (name + arguments).
      </div>
    </div>
  );
}

function BuiltinRow({ guard, onToggle }: {
  guard: BuiltinGuardUI; onToggle: (enabled: boolean) => void;
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 10,
      border: '1px solid var(--tb-border)',
      borderRadius: 5,
      padding: '8px 10px',
      marginBottom: 6,
      background: 'var(--tb-surface)',
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tb-text)' }}>
          {guard.label}
        </div>
        <div style={{ fontSize: 9.5, color: 'var(--tb-text-muted)', marginTop: 2, lineHeight: 1.4 }}>
          {guard.description}
        </div>
      </div>
      <Toggle checked={guard.enabled} onChange={onToggle} />
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      style={{
        width: 28, height: 16,
        borderRadius: 8,
        background: checked ? 'var(--tb-green)' : 'var(--tb-surface-2)',
        border: `1px solid ${checked ? 'var(--tb-green)' : 'var(--tb-border-2)'}`,
        position: 'relative',
        cursor: 'pointer',
        flexShrink: 0,
        marginTop: 2,
        transition: 'background 0.15s, border-color 0.15s',
      }}
    >
      <span style={{
        position: 'absolute',
        top: 1, left: checked ? 13 : 1,
        width: 12, height: 12,
        borderRadius: '50%',
        background: checked ? '#0d1117' : 'var(--tb-text-muted)',
        transition: 'left 0.15s',
      }} />
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 9, fontWeight: 700,
      letterSpacing: '0.1em', textTransform: 'uppercase',
      color: 'var(--tb-text-dim)',
      padding: '2px 0 7px',
    }}>{children}</div>
  );
}
