import { useEffect, useMemo, useRef, useState } from 'react';

import EmptyState from './components/EmptyState';
import SessionPicker from './components/SessionPicker';
import { DotsIcon, PanelIcon } from './components/Icons';
import { agentIdentity } from './codename';
import { computeChapters } from './chapters';
import { computeFileChanges, summarizeChanges } from './fileChanges';
import { useSessionFeed } from './useSessionFeed';

/**
 * Lean sidebar: a glanceable status + launcher, NOT a squished copy of the
 * panel. It answers "is my agent OK, and what is it on?" at a glance — status,
 * the current chapter, a couple of counts, an anomaly badge, a live pause —
 * then hands off to the roomy Action Map for chapters, files, decisions, and
 * review. The panel auto-opens on first activity (traceback.autoOpenMap), so in
 * practice this stays a quiet status strip while the real work happens there.
 */
export default function SidebarApp() {
  const feed = useSessionFeed();
  const { sessions, archived, display, pinnedId } = feed;

  const traceNodes = display?.nodes ?? [];
  const hasData    = traceNodes.length > 0;
  const isLive     = display ? !display.stopped && !archived : false;
  const anomaly    = archived ? undefined : display?.anomaly;
  const isAnomalous = !!anomaly?.isAnomalous && anomaly.type !== 'stall';
  const realCount  = traceNodes.filter((n) => !n.toolName.startsWith('__')).length;

  const chapters = useMemo(() => computeChapters(traceNodes), [traceNodes]);
  const liveChapter = chapters[chapters.length - 1] ?? null;
  const chapterTitle = firstLine(liveChapter?.text) || (isLive ? 'Working…' : 'Session');

  const changes = useMemo(() => computeFileChanges(traceNodes), [traceNodes]);
  const changesSummary = summarizeChanges(changes);

  const statusLabel = archived ? 'REPLAY' : display?.paused ? 'PAUSED' : isLive ? 'LIVE' : 'DONE';
  const statusColor = display?.paused ? 'var(--tb-amber)'
    : archived ? 'var(--tb-purple, #a371f7)'
    : isLive ? 'var(--tb-green)' : 'var(--tb-text-dim)';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100vw', height: '100vh' }}>
      <div className="tb-scanlines" />

      {/* ── Compact header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '0 10px', height: 30,
        borderBottom: '1px solid var(--tb-border)',
        background: 'var(--tb-surface)',
        fontFamily: 'var(--tb-ui-font)',
        flexShrink: 0,
      }}>
        <div
          className={isLive && !display?.paused ? 'live-dot' : ''}
          style={{ width: 6, height: 6, borderRadius: '50%', flexShrink: 0, background: statusColor }}
        />
        <span
          title={display ? `${agentIdentity(display.id).name} — ${display.label}` : undefined}
          style={{
            fontSize: 12, fontWeight: 600,
            color: display ? agentIdentity(display.id).color : 'var(--tb-text)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0,
          }}
        >
          {display ? agentIdentity(display.id).name : 'TraceBack'}
        </span>
        <div style={{ flex: 1 }} />
        {hasData && <OverflowMenu feed={feed} />}
        <HeaderButton title="Open Action Map" onClick={feed.openFullPanel}><PanelIcon /></HeaderButton>
      </div>

      {sessions.length > 1 && (
        <SessionPicker
          sessions={sessions.map((s) => ({
            id: s.id, label: s.label, startedAt: s.startedAt,
            nodeCount: s.nodeCount, stopped: s.stopped,
            anomalous: !!s.anomaly?.isAnomalous && s.anomaly?.type !== 'stall',
          }))}
          displayId={display?.id ?? null}
          pinnedId={pinnedId}
          onSelect={feed.selectSession}
        />
      )}

      {/* ── Body: glance card + launcher ── */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: hasData ? 12 : 0 }}>
        {!hasData ? (
          <EmptyState />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontFamily: 'var(--tb-ui-font)' }}>
            {/* status pill */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                fontSize: 10, fontWeight: 700, letterSpacing: 0.6,
                color: statusColor, padding: '2px 7px', borderRadius: 4,
                border: `1px solid ${statusColor}`, flexShrink: 0,
              }}>
                {statusLabel}
              </span>
              <span style={{ fontSize: 12, color: 'var(--tb-text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {display?.label}
              </span>
            </div>

            {/* current chapter */}
            <div>
              <div style={{ fontSize: 10, color: 'var(--tb-text-dim)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 3 }}>
                Current chapter
              </div>
              <div style={{ fontSize: 13, color: 'var(--tb-text)', fontWeight: 500, lineHeight: 1.35 }}>
                {chapterTitle}
              </div>
            </div>

            {/* key counts */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: 12, color: 'var(--tb-text-muted)' }}>
              <span><b style={{ color: 'var(--tb-text)' }}>{realCount}</b> actions</span>
              {changes.length > 0 && (
                <span><b style={{ color: 'var(--tb-text)' }}>{changes.length}</b> files changed</span>
              )}
              {changesSummary && <span style={{ color: 'var(--tb-text-dim)' }}>{changesSummary}</span>}
            </div>

            {/* anomaly badge */}
            {isAnomalous && (
              <div style={{
                display: 'flex', alignItems: 'flex-start', gap: 6,
                fontSize: 12, color: 'var(--tb-red, #f85149)',
                background: 'rgba(248,81,73,0.08)', border: '1px solid rgba(248,81,73,0.3)',
                borderRadius: 5, padding: '6px 8px', lineHeight: 1.35,
              }}>
                <span style={{ flexShrink: 0 }}>⚠</span>
                <span>{anomaly?.title ?? 'Anomaly detected'}</span>
              </div>
            )}

            {/* live control: freeze without leaving the sidebar */}
            {isLive && !archived && (
              <button
                onClick={feed.pauseToggle}
                style={{
                  ...ctaBase,
                  background: 'transparent',
                  color: display?.paused ? 'var(--tb-green)' : 'var(--tb-amber)',
                  border: `1px solid ${display?.paused ? 'var(--tb-green)' : 'var(--tb-amber)'}`,
                }}
              >
                {display?.paused ? '▶  Resume agent' : '⏸  Pause agent'}
              </button>
            )}

            {/* primary CTA */}
            <button onClick={feed.openFullPanel} style={{ ...ctaBase, ...ctaPrimary }}>
              Open Action Map&nbsp;▸
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

const ctaBase: React.CSSProperties = {
  width: '100%', padding: '9px 12px', borderRadius: 6,
  fontSize: 13, fontWeight: 600, cursor: 'pointer',
  fontFamily: 'var(--tb-ui-font)', lineHeight: 1,
};

const ctaPrimary: React.CSSProperties = {
  background: 'var(--tb-accent, #58a6ff)',
  color: '#07090d',
  border: '1px solid var(--tb-accent, #58a6ff)',
};

/** First non-empty line of a prompt, trimmed and length-capped for a chip. */
function firstLine(text?: string): string {
  if (!text) return '';
  const line = text.split('\n').map((l) => l.trim()).find(Boolean) ?? '';
  return line.length > 90 ? line.slice(0, 89).trimEnd() + '…' : line;
}

function HeaderButton({ children, title, onClick }: {
  children: React.ReactNode; title: string; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      title={title}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? 'var(--tb-surface-2)' : 'none',
        border: 'none',
        borderRadius: 3,
        color: hovered ? 'var(--tb-text)' : 'var(--tb-text-muted)',
        fontSize: 13,
        padding: '2px 5px',
        cursor: 'pointer',
        flexShrink: 0,
        lineHeight: 1,
      }}
    >
      {children}
    </button>
  );
}

/** ⋯ dropdown holding the export + clear actions in the compact sidebar. */
function OverflowMenu({ feed }: { feed: ReturnType<typeof useSessionFeed> }) {
  const [open, setOpen] = useState(false);
  const [confirmingClear, setConfirmingClear] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setConfirmingClear(false);
      }
    }
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [open]);

  function item(label: string, action: () => void, danger = false) {
    return (
      <div
        onClick={() => { action(); setOpen(false); }}
        style={{
          padding: '5px 12px',
          fontSize: 12,
          color: danger ? '#f85149' : 'var(--tb-text)',
          cursor: 'pointer',
          whiteSpace: 'nowrap',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--tb-surface-2)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
      >
        {label}
      </div>
    );
  }

  return (
    <div ref={ref} style={{ position: 'relative', flexShrink: 0 }}>
      <HeaderButton title="More actions" onClick={() => setOpen((v) => !v)}><DotsIcon /></HeaderButton>
      {open && (
        <div style={{
          position: 'absolute', right: 0, top: 22,
          background: 'var(--tb-surface)',
          border: '1px solid var(--tb-border-2)',
          borderRadius: 5,
          boxShadow: '0 6px 20px rgba(0,0,0,0.5)',
          zIndex: 100,
          padding: '4px 0',
          animation: 'chatPanelIn 0.12s ease-out',
        }}>
          {item('Export PNG', feed.exportPng)}
          {item('Export HTML', feed.exportHtml)}
          {item('Export JSON', feed.exportJson)}
          {item('Copy MD report', feed.copyReport)}
          <div style={{ height: 1, background: 'var(--tb-border)', margin: '4px 0' }} />
          {confirmingClear
            ? item('✕ Confirm clear?', () => { feed.clear(); setConfirmingClear(false); }, true)
            : (
              <div
                onClick={() => setConfirmingClear(true)}
                style={{
                  padding: '5px 12px', fontSize: 12,
                  color: '#f85149', cursor: 'pointer', whiteSpace: 'nowrap',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--tb-surface-2)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
              >
                Clear session…
              </div>
            )}
        </div>
      )}
    </div>
  );
}
