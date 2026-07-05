import { useEffect, useMemo, useState } from 'react';

import EmptyState from './components/EmptyState';
import FileChangesPanel from './components/FileChangesPanel';
import PromptList from './components/PromptList';
import PromptChapterView from './components/PromptChapterView';
import RightPanel from './components/RightPanel';
import { formatDuration } from './components/TimelineCard';
import { AlertIcon, CheckIcon, FileIcon } from './components/Icons';
import { agentIdentity } from './codename';
import { anomaliesFor, computeChapters } from './chapters';
import { useSessionFeed, FullSessionData, ArchivedMeta } from './useSessionFeed';

/**
 * Full editor panel — a single prompt-chapter view: prompts on the left, the
 * focused prompt's tasks + action cards in the middle, and the
 * Anomalies/Files/Guards tabs on the right. There is no other view.
 */
export default function PanelApp() {
  // Wide layout: enough room for the chapter view plus the right tab panel.
  const [wide, setWide] = useState(window.innerWidth >= 1150);
  useEffect(() => {
    const onResize = () => setWide(window.innerWidth >= 1150);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const feed = useSessionFeed();
  const {
    sessions, history, archived, display,
    expandedId, setExpandedId,
    timelineRef, scrollRef, handleScroll,
  } = feed;

  const traceNodes = display?.nodes ?? [];
  const hasData    = traceNodes.length > 0;
  const isLive     = display ? !display.stopped && !archived : false;
  const anomaly    = archived ? undefined : display?.anomaly;
  const records    = display?.anomalyHistory ?? [];

  // ── Prompt chapters ──
  const chapters = useMemo(() => computeChapters(traceNodes), [traceNodes]);
  const anomalyCounts = useMemo(() => {
    const map = new Map<number, number>();
    for (const c of chapters) map.set(c.index, anomaliesFor(c, records).length);
    return map;
  }, [chapters, records]);

  // Focused chapter: user-picked, else follow the latest prompt.
  const [pickedChapter, setPickedChapter] = useState<number | null>(null);
  useEffect(() => setPickedChapter(null), [display?.id]); // reset per session
  const selectedIndex =
    pickedChapter !== null && pickedChapter <= chapters.length
      ? pickedChapter
      : chapters.length;
  const selectedChapter = chapters.find((c) => c.index === selectedIndex) ?? null;

  // Archived metas whose session isn't currently live (avoid double listing)
  const liveIds     = new Set(sessions.map((s) => s.id));
  const pastEntries = history.filter((h) => !liveIds.has(h.id));

  return (
    <div style={{
      display: 'flex', width: '100vw', height: '100vh',
      fontFamily: 'var(--tb-ui-font)',
    }}>
      <div className="tb-scanlines" />
      {/* ── Left rail ── */}
      <div style={{
        width: 240, flexShrink: 0,
        borderRight: '1px solid var(--tb-border)',
        background: 'var(--tb-surface)',
        display: 'flex', flexDirection: 'column',
        minHeight: 0,
      }}>
        <div style={{
          padding: '16px 16px 14px',
          borderBottom: '1px solid var(--tb-border)',
          display: 'flex', alignItems: 'center', gap: 10,
          flexShrink: 0,
        }}>
          <div
            className={isLive ? 'live-dot' : ''}
            style={{
              width: 9, height: 9, borderRadius: '50%', flexShrink: 0,
              background: isLive ? 'var(--tb-green)' : 'var(--tb-text-dim)',
            }}
          />
          <div style={{ minWidth: 0 }}>
            <div style={{
              fontSize: 15, fontWeight: 700, lineHeight: 1.2,
              color: display ? agentIdentity(display.id).color : 'var(--tb-text)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {display ? agentIdentity(display.id).name : 'TraceBack'}
            </div>
            {display && (
              <div style={{
                fontSize: 11.5, color: 'var(--tb-text-muted)', marginTop: 2,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {display.label}
              </div>
            )}
          </div>
        </div>

        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          {sessions.length > 1 && (
            <>
              <RailLabel>Live agents</RailLabel>
              {sessions.map((s) => (
                <SessionRailCard
                  key={s.id}
                  session={s}
                  active={!archived && s.id === display?.id}
                  onClick={() => feed.selectSession(s.id)}
                />
              ))}
            </>
          )}

          {hasData && chapters.length > 0 && (
            <>
              <RailLabel>Prompts</RailLabel>
              <PromptList
                chapters={chapters}
                selectedIndex={selectedIndex}
                isLive={isLive}
                anomalyCounts={anomalyCounts}
                onSelect={(i) => setPickedChapter(i === chapters.length ? null : i)}
              />
            </>
          )}

          {pastEntries.length > 0 && (
            <>
              <RailLabel>History</RailLabel>
              {pastEntries.map((h) => (
                <HistoryRailCard
                  key={h.id}
                  meta={h}
                  active={archived?.id === h.id}
                  onClick={() => feed.selectArchived(h.id)}
                />
              ))}
            </>
          )}

          {sessions.length === 0 && pastEntries.length === 0 && (
            <div style={{ padding: '10px 14px', fontSize: 10.5, color: 'var(--tb-text-dim)', lineHeight: 1.5 }}>
              No sessions yet. Start Claude Code in a terminal and its actions stream in here.
            </div>
          )}
        </div>

        {/* ── Export actions ── */}
        {hasData && (
          <div style={{
            flexShrink: 0,
            borderTop: '1px solid var(--tb-border)',
            padding: 12,
            display: 'flex', flexDirection: 'column', gap: 8,
          }}>
            <RailButton onClick={feed.exportJson} icon={<DownloadIcon size={15} />}>
              Export session
            </RailButton>
            <CopyRailButton onCopy={feed.copyReport} />
            <RailButton onClick={feed.exportHtml} icon={<ShareIcon size={15} />}>
              Share as HTML
            </RailButton>
          </div>
        )}
      </div>

      {/* ── Main column ── */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        {/* Header bar */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '0 16px', height: 40,
          borderBottom: '1px solid var(--tb-border)',
          background: 'var(--tb-surface)',
          flexShrink: 0,
        }}>
          {archived && (
            <span style={{
              fontSize: 9, fontWeight: 700, letterSpacing: '0.06em',
              color: 'var(--tb-text-muted)',
              border: '1px solid var(--tb-border-2)',
              borderRadius: 3, padding: '1px 6px',
              textTransform: 'uppercase',
            }}>
              archived
            </span>
          )}
          <div style={{ flex: 1 }} />
          {hasData && !archived && (
            <div style={{ display: 'flex', gap: 4, marginLeft: 8 }}>
              <ClearPanelButton onClear={feed.clear} />
            </div>
          )}
        </div>

        {/* Scroll area */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          style={{ position: 'relative', flex: 1, minHeight: 0, overflowY: 'auto' }}
        >
          {!hasData ? (
            <EmptyState />
          ) : (
            <div ref={timelineRef} style={{
              width: '100%',
              display: 'flex',
              alignItems: 'stretch',
            }}>
              {/* ── Main column: fills all available width ── */}
              <div style={{ flex: 1, minWidth: 0 }}>
                {selectedChapter && (
                  <PromptChapterView
                    chapter={selectedChapter}
                    isLive={isLive && selectedIndex === chapters.length}
                    isLast={selectedIndex === chapters.length}
                    realTokens={display?.contextTokens}
                    expandedId={expandedId}
                    onToggle={(id) => setExpandedId(expandedId === id ? null : id)}
                    anomaly={selectedIndex === chapters.length ? anomaly : undefined}
                    paused={display?.paused ?? false}
                    onPauseToggle={archived ? undefined : feed.pauseToggle}
                    onRedirect={archived ? undefined : feed.redirect}
                  />
                )}

                {!wide && <FileChangesPanel nodes={traceNodes} defaultOpen clickable={!archived} />}
              </div>

              {/* ── Right panel (wide screens): Anomalies / Files / Guards ── */}
              {wide && (
                <div style={{
                  width: 340, flexShrink: 0,
                  position: 'sticky', top: 0,
                  alignSelf: 'flex-start',
                  height: '100vh',
                  borderLeft: '1px solid var(--tb-border)',
                  background: 'var(--tb-surface)',
                }}>
                  <RightPanel
                    nodes={traceNodes}
                    chapters={chapters}
                    selectedIndex={selectedIndex}
                    records={records}
                    filesClickable={!archived}
                    guards={feed.guards}
                    onSetGuard={feed.setGuard}
                    onAddGuard={feed.addGuard}
                    onRemoveGuard={feed.removeGuard}
                  />
                </div>
              )}
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
    </div>
  );
}

// ── Rail pieces ─────────────────────────────────────────────────────────────

function RailLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      padding: '12px 14px 5px',
      fontSize: 9, fontWeight: 700,
      letterSpacing: '0.1em', textTransform: 'uppercase',
      color: 'var(--tb-text-dim)',
    }}>{children}</div>
  );
}

function railCardStyle(active: boolean, hovered: boolean): React.CSSProperties {
  return {
    margin: '0 8px 4px',
    padding: '7px 10px',
    borderRadius: 5,
    border: `1px solid ${active ? 'var(--tb-border-2)' : 'transparent'}`,
    background: active ? 'var(--tb-surface-2)' : hovered ? 'rgba(22,27,34,0.6)' : 'transparent',
    cursor: 'pointer',
    transition: 'background 0.1s, border-color 0.1s',
  };
}

function SessionRailCard({ session, active, onClick }: {
  session: FullSessionData; active: boolean; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const anomalous = !!session.anomaly?.isAnomalous;
  const live = !session.stopped;
  const identity = agentIdentity(session.id);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...railCardStyle(active, hovered),
        borderLeft: `2px solid ${active ? identity.color : 'transparent'}`,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        <div
          className={live && !anomalous ? 'live-dot' : ''}
          style={{
            width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
            background: anomalous ? '#f85149' : live ? '#3fb950' : 'var(--tb-text-dim)',
            ...(anomalous ? { animation: 'pendingPulse 1.2s ease-in-out infinite' } : {}),
          }}
        />
        <span style={{
          fontSize: 11.5, fontWeight: active ? 600 : 400,
          color: identity.color,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          flex: 1, minWidth: 0,
        }}>
          {identity.name}
        </span>
        {anomalous && <span style={{ color: '#f85149', display: 'flex' }}><AlertIcon size={10} /></span>}
      </div>
      <div style={{
        fontSize: 9.5, color: 'var(--tb-text-muted)', paddingLeft: 13, marginTop: 1,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>
        {session.label} · {session.nodeCount} actions{live ? ' · live' : ''}
      </div>
    </div>
  );
}

function HistoryRailCard({ meta, active, onClick }: {
  meta: ArchivedMeta; active: boolean; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const dur = formatDuration(meta.endedAt - meta.startedAt);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={railCardStyle(active, hovered)}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        <span style={{
          fontSize: 11.5, fontWeight: active ? 600 : 400,
          color: 'var(--tb-text-muted)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          flex: 1, minWidth: 0,
        }}>
          <span style={{ color: agentIdentity(meta.id).color, opacity: 0.75 }}>
            {agentIdentity(meta.id).name}
          </span>
          {' '}· {meta.label}
        </span>
        {meta.errorCount > 0 && (
          <span style={{ fontSize: 9, color: '#f85149' }}>{meta.errorCount}✕</span>
        )}
        {meta.anomalyCount > 0 && (
          <span style={{ color: '#d29922', display: 'flex' }}><AlertIcon size={10} /></span>
        )}
      </div>
      <div style={{ fontSize: 9.5, color: 'var(--tb-text-dim)', marginTop: 1 }}>
        {new Date(meta.startedAt).toLocaleDateString()} · {meta.nodeCount} actions{dur ? ` · ${dur}` : ''}
      </div>
    </div>
  );
}

// ── Buttons ─────────────────────────────────────────────────────────────────

function RailButton({ children, onClick, icon }: {
  children: React.ReactNode; onClick: () => void; icon?: React.ReactNode;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        width: '100%',
        fontSize: 12, fontFamily: 'var(--tb-ui-font)', fontWeight: 550,
        padding: '10px 13px', cursor: 'pointer',
        textAlign: 'left',
        background: hovered ? 'rgba(88,166,255,0.08)' : 'var(--tb-surface-2)',
        color: hovered ? 'var(--tb-blue)' : 'var(--tb-text)',
        border: `1px solid ${hovered ? 'rgba(88,166,255,0.4)' : 'var(--tb-border)'}`,
        borderRadius: 8,
        transition: 'color 0.1s, border-color 0.1s, background 0.1s',
        whiteSpace: 'nowrap',
      }}
    >
      {icon && <span style={{ display: 'flex', flexShrink: 0, opacity: 0.85 }}>{icon}</span>}
      <span>{children}</span>
    </button>
  );
}

function CopyRailButton({ onCopy }: { onCopy: () => Promise<void> }) {
  const [copied, setCopied] = useState(false);
  return (
    <RailButton
      icon={copied ? <CheckIcon size={15} /> : <FileIcon size={15} />}
      onClick={async () => {
        await onCopy();
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }}
    >
      {copied ? 'Copied to clipboard' : 'Markdown report'}
    </RailButton>
  );
}

function DownloadIcon({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round"
         style={{ display: 'block' }}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 10l5 5 5-5" />
      <path d="M12 15V3" />
    </svg>
  );
}

function ShareIcon({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round"
         style={{ display: 'block' }}>
      <circle cx="18" cy="5" r="3" />
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="19" r="3" />
      <path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4" />
    </svg>
  );
}

function ClearPanelButton({ onClear }: { onClear: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [hovered, setHovered]       = useState(false);

  return (
    <button
      title={confirming ? 'Click again to confirm clear' : 'Clear current session'}
      onClick={() => {
        if (!confirming) {
          setConfirming(true);
          setTimeout(() => setConfirming(false), 2200);
          return;
        }
        onClear();
        setConfirming(false);
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        fontSize: 10, fontFamily: 'var(--tb-ui-font)',
        fontWeight: confirming ? 700 : 500,
        padding: '3px 9px', cursor: 'pointer',
        background: confirming ? 'rgba(248,81,73,0.12)' : hovered ? 'rgba(248,81,73,0.07)' : 'transparent',
        color: confirming || hovered ? '#f85149' : 'var(--tb-text-muted)',
        border: `1px solid ${confirming ? 'rgba(248,81,73,0.55)' : hovered ? 'rgba(248,81,73,0.35)' : 'var(--tb-border)'}`,
        borderRadius: 3,
        transition: 'color 0.1s, border-color 0.1s, background 0.1s',
        whiteSpace: 'nowrap',
      }}
    >
      {confirming ? '✕ confirm?' : 'clear'}
    </button>
  );
}
