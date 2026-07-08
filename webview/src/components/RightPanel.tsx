import { useState } from 'react';
import FileChangesPanel from './FileChangesPanel';
import { TimelineNode } from './TimelineCard';
import { anomaliesFor, PromptChapter } from '../chapters';
import { AnomalyRecordUI, BuiltinGuardUI, GuardsStateUI, LedgerItemUI } from '../useSessionFeed';
import { AlertIcon, BranchIcon, ChevronIcon, ClockIcon, QuestionIcon } from './Icons';

type Tab = 'anomalies' | 'files' | 'decisions' | 'guards';

interface Props {
  nodes:          TimelineNode[];
  chapters:       PromptChapter[];
  selectedIndex:  number;
  records:        AnomalyRecordUI[];
  ledger:         LedgerItemUI[];
  filesClickable: boolean;
  guards:         GuardsStateUI;
  onSetGuard:     (key: string, enabled: boolean) => void;
  onAddGuard:     (pattern: string) => void;
  onRemoveGuard:  (pattern: string) => void;
}

/**
 * Right panel: Anomalies (per-prompt evidence trail), Files (what changed on
 * disk), Decisions (judgment calls mined from the transcript), Guards (the
 * policy rules that auto-deny tool calls).
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
        display: 'flex', alignItems: 'stretch', gap: 2, flexShrink: 0,
        padding: '0 8px',
        borderBottom: '1px solid var(--tb-border)',
        background: 'var(--tb-surface)',
        overflowX: 'auto',
      }}>
        <TabButton
          active={tab === 'anomalies'} onClick={() => setTab('anomalies')}
          label="Anomalies" count={anomalyTotal} countColor="#f85149"
        />
        <TabButton
          active={tab === 'files'} onClick={() => setTab('files')}
          label="Files"
        />
        <TabButton
          active={tab === 'decisions'} onClick={() => setTab('decisions')}
          label="Decisions" count={props.ledger.length}
        />
        <TabButton
          active={tab === 'guards'} onClick={() => setTab('guards')}
          label="Guards"
        />
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
        {tab === 'decisions' && (
          <DecisionsTab chapters={props.chapters} ledger={props.ledger} />
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

function TabButton({ active, onClick, label, count = 0, countColor }: {
  active: boolean; onClick: () => void; label: string;
  count?: number; countColor?: string;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        fontSize: 12.5, fontWeight: active ? 600 : 450,
        fontFamily: 'var(--tb-ui-font)',
        padding: '10px 10px 9px',
        cursor: 'pointer',
        background: 'transparent',
        color: active || hovered ? 'var(--tb-text)' : 'var(--tb-text-muted)',
        border: 'none',
        boxShadow: active ? 'inset 0 -2px 0 var(--tb-blue)' : 'inset 0 -2px 0 transparent',
        transition: 'color 0.12s ease, box-shadow 0.15s ease',
        whiteSpace: 'nowrap',
      }}
    >
      {label}
      {count > 0 && (
        <span style={{
          fontSize: 10, fontWeight: 650, lineHeight: 1,
          fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
          color: countColor ?? 'var(--tb-text-muted)',
          background: countColor ? `${countColor}1a` : 'var(--tb-surface-2)',
          border: `1px solid ${countColor ? `${countColor}44` : 'var(--tb-border-2)'}`,
          borderRadius: 8, padding: '2px 6px',
        }}>
          {count}
        </span>
      )}
    </button>
  );
}

// ─── Decisions ────────────────────────────────────────────────────────────────

/**
 * The judgment calls the agent made in prose — decisions ("instead of…",
 * "went with…") and assumptions ("I'll assume…") — grouped by the prompt
 * whose time slice they fell into. This is where sessions silently go wrong;
 * an assumption caught live is a Redirect before it calcifies.
 */
function DecisionsTab({ chapters, ledger }: {
  chapters: PromptChapter[]; ledger: LedgerItemUI[];
}) {
  if (ledger.length === 0) {
    return (
      <div style={{ padding: '14px 14px 20px', fontSize: 11.5, color: 'var(--tb-text-dim)', lineHeight: 1.6 }}>
        No judgment calls detected yet. When the agent writes "I'll assume…",
        "instead of…", or "went with…", those sentences surface here — the
        choices being made on your behalf.
      </div>
    );
  }

  // Group items into chapters by timestamp; anything before the first prompt
  // (or in an unmatched gap) pools under its nearest chapter.
  const groups = chapters
    .map((c) => ({
      chapter: c,
      items: ledger.filter((it) => it.timestamp >= c.timestamp && it.timestamp < c.endTimestamp),
    }))
    .filter((g) => g.items.length > 0)
    .reverse(); // newest prompt first
  const matched  = new Set(groups.flatMap((g) => g.items));
  const orphaned = ledger.filter((it) => !matched.has(it));

  return (
    <div style={{ padding: '10px 12px 20px' }}>
      {groups.map(({ chapter, items }) => (
        <div key={chapter.index} style={{ marginBottom: 12 }}>
          <SectionLabel>P{chapter.index} — {chapter.text.replace(/\s+/g, ' ').slice(0, 40)}{chapter.text.length > 40 ? '…' : ''}</SectionLabel>
          {items.map((it, i) => <LedgerRow key={i} item={it} />)}
        </div>
      ))}
      {orphaned.length > 0 && (
        <div>
          <SectionLabel>Earlier</SectionLabel>
          {orphaned.map((it, i) => <LedgerRow key={i} item={it} />)}
        </div>
      )}
    </div>
  );
}

function LedgerRow({ item }: { item: LedgerItemUI }) {
  const isAssumption = item.kind === 'assumption';
  const color = isAssumption ? '#d29922' : '#58a6ff';
  const Icon  = isAssumption ? QuestionIcon : BranchIcon;
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 9,
      padding: '8px 10px',
      border: '1px solid var(--tb-border)',
      borderRadius: 8,
      background: 'var(--tb-surface-2)',
      marginBottom: 7,
    }}>
      <span title={isAssumption ? 'Assumption' : 'Decision'}
            style={{ color, display: 'flex', flexShrink: 0, marginTop: 1 }}>
        <Icon size={13} />
      </span>
      <div style={{ minWidth: 0 }}>
        <div style={{
          fontSize: 10, fontWeight: 700, letterSpacing: '0.07em',
          textTransform: 'uppercase', color, marginBottom: 3,
        }}>
          {item.kind}
        </div>
        <div style={{
          fontSize: 12.5, lineHeight: 1.5, color: 'var(--tb-text)',
          wordBreak: 'break-word',
        }}>
          {item.text}
        </div>
      </div>
    </div>
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
      <SectionLabel>This session</SectionLabel>
      {current.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--tb-text-dim)', padding: '4px 2px 12px' }}>
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
        <div style={{ fontSize: 11.5, color: 'var(--tb-text-dim)', padding: '10px 2px' }}>
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
      border: `1px solid ${color}55`,
      borderRadius: 8,
      background: sev === 'high' ? 'rgba(120,20,18,0.32)' : 'rgba(90,66,12,0.3)',
      padding: '11px 13px',
      marginBottom: 9,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
        <span style={{ color, display: 'flex' }}><Icon size={14} /></span>
        <span style={{ fontSize: 13.5, fontWeight: 700, color }}>{title}</span>
      </div>
      <div style={{
        fontSize: 12.5, lineHeight: 1.45,
        color: sev === 'high' ? '#ffb3ac' : '#e8c877',
        wordBreak: 'break-word',
      }}>
        {desc}
      </div>
    </div>
  );
}

function HistorySection({ title, count, children }: {
  title: string; count: number; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div style={{ marginTop: 12 }}>
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '5px 2px 8px', cursor: 'pointer', userSelect: 'none',
        }}
      >
        <span style={{
          fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
          textTransform: 'uppercase', color: 'var(--tb-text-dim)',
        }}>
          All time · {title}
        </span>
        <span style={{ fontSize: 10, color: 'var(--tb-text-dim)' }}>({count})</span>
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
        <div style={{ fontSize: 11.5, color: 'var(--tb-text-dim)', padding: '4px 2px 8px' }}>
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
              fontSize: 11.5,
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
                fontSize: 11, padding: '1px 7px',
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
                fontSize: 11.5,
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
                fontSize: 11, fontWeight: 600,
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
              fontSize: 11.5, fontFamily: 'var(--tb-ui-font)',
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
        fontSize: 10.5, lineHeight: 1.5,
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
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--tb-text)' }}>
          {guard.label}
        </div>
        <div style={{ fontSize: 10.5, color: 'var(--tb-text-muted)', marginTop: 2, lineHeight: 1.4 }}>
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
      fontSize: 10, fontWeight: 700,
      letterSpacing: '0.1em', textTransform: 'uppercase',
      color: 'var(--tb-text-dim)',
      padding: '2px 0 7px',
    }}>{children}</div>
  );
}
