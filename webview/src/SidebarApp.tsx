import { useEffect, useMemo, useRef, useState } from 'react';

import EmptyState from './components/EmptyState';
import SessionPicker from './components/SessionPicker';
import FileChangesPanel from './components/FileChangesPanel';
import PromptChapterView from './components/PromptChapterView';
import { DotsIcon, PanelIcon } from './components/Icons';
import { agentIdentity } from './codename';
import { computeChapters } from './chapters';
import { useSessionFeed } from './useSessionFeed';

/**
 * Compact sidebar: one 30px header row (status + label + actions), session
 * pills only when several agents run, then the live prompt-chapter view — the
 * same single view as the full panel, no timeline/map/zoom toggle.
 */
export default function SidebarApp() {
  const feed = useSessionFeed();
  const {
    sessions, archived, display, pinnedId,
    expandedId, setExpandedId,
    timelineRef, scrollRef, handleScroll,
  } = feed;

  const traceNodes = display?.nodes ?? [];
  const hasData    = traceNodes.length > 0;
  const isLive     = display ? !display.stopped && !archived : false;
  const anomaly    = archived ? undefined : display?.anomaly;
  const realCount  = traceNodes.filter((n) => !n.toolName.startsWith('__')).length;

  // The live chapter = the current (last) prompt's slice.
  const chapters = useMemo(() => computeChapters(traceNodes), [traceNodes]);
  const liveChapter = chapters[chapters.length - 1] ?? null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100vw', height: '100vh' }}>
      <div className="tb-scanlines" />
      {/* ── Single compact header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '0 10px', height: 30,
        borderBottom: '1px solid var(--tb-border)',
        background: 'var(--tb-surface)',
        fontFamily: 'var(--tb-ui-font)',
        flexShrink: 0,
      }}>
        <div
          className={isLive ? 'live-dot' : ''}
          style={{
            width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
            background: isLive ? 'var(--tb-green)' : 'var(--tb-text-dim)',
          }}
        />
        <span
          title={display ? `${agentIdentity(display.id).name} — ${display.label}` : undefined}
          style={{
            fontSize: 11, fontWeight: 600,
            color: display ? agentIdentity(display.id).color : 'var(--tb-text)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            minWidth: 0,
          }}
        >
          {display ? agentIdentity(display.id).name : 'TraceBack'}
          {display && (
            <span style={{ fontWeight: 400, color: 'var(--tb-text-muted)' }}>
              {' '}· {display.label}
            </span>
          )}
        </span>
        {hasData && (
          <span style={{ fontSize: 10, color: 'var(--tb-text-muted)', flexShrink: 0 }}>
            {realCount}
          </span>
        )}
        <div style={{ flex: 1 }} />
        {hasData && <OverflowMenu feed={feed} />}
        <HeaderButton title="Open full view" onClick={feed.openFullPanel}><PanelIcon /></HeaderButton>
      </div>

      {sessions.length > 1 && (
        <SessionPicker
          sessions={sessions.map((s) => ({
            id: s.id, label: s.label, startedAt: s.startedAt,
            nodeCount: s.nodeCount, stopped: s.stopped,
            anomalous: !!s.anomaly?.isAnomalous,
          }))}
          displayId={display?.id ?? null}
          pinnedId={pinnedId}
          onSelect={feed.selectSession}
        />
      )}

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        style={{ position: 'relative', flex: 1, minHeight: 0, overflowY: 'auto' }}
      >
        {!hasData || !liveChapter ? (
          <EmptyState />
        ) : (
          <div ref={timelineRef}>
            <PromptChapterView
              chapter={liveChapter}
              isLive={isLive}
              isLast
              realTokens={display?.contextTokens}
              expandedId={expandedId}
              onToggle={(id) => setExpandedId(expandedId === id ? null : id)}
              anomaly={anomaly}
              paused={display?.paused ?? false}
              onPauseToggle={archived ? undefined : feed.pauseToggle}
              onRedirect={archived ? undefined : feed.redirect}
            />

            <FileChangesPanel nodes={traceNodes} />
          </div>
        )}

        {anomaly?.isAnomalous && (
          <div style={{
            position: 'fixed', inset: 0,
            background: 'radial-gradient(ellipse at center, rgba(248,81,73,0.07) 0%, rgba(248,81,73,0.02) 60%, transparent 100%)',
            pointerEvents: 'none',
            zIndex: 10,
          }} />
        )}
      </div>
    </div>
  );
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
        fontSize: 12,
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
          fontSize: 11,
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
                  padding: '5px 12px', fontSize: 11,
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
