import { useEffect, useMemo, useState } from 'react';

import Timeline from './components/Timeline';
import SessionOdometer from './components/SessionOdometer';
import EmptyState from './components/EmptyState';
import FileChangesPanel from './components/FileChangesPanel';
import ObjectiveHeader from './components/ObjectiveHeader';
import SessionMap from './components/SessionMap';
import PromptList from './components/PromptList';
import PromptChapterView from './components/PromptChapterView';
import RightPanel from './components/RightPanel';
import ZoomControl, { ZoomLevel } from './components/ZoomControl';
import { formatDuration } from './components/TimelineCard';
import { AlertIcon } from './components/Icons';
import { agentIdentity } from './codename';
import { anomaliesFor, computeChapters } from './chapters';
import { useSessionFeed, FullSessionData, ArchivedMeta } from './useSessionFeed';

/**
 * Full editor panel, structured around prompt chapters: prompts on the left,
 * the focused prompt's tasks + actions in the middle, and the
 * Anomalies/Files/Guards tabs on the right. The classic Map/Steps/Detail
 * zooms stay one click away.
 */
export default function PanelApp() {
  const [zoom, setZoom] = useState<ZoomLevel>('chapters');
  // Wide layout: enough room for the timeline plus the right tab panel.
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
    chatAnswer, chatLoading,
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
          padding: '14px 14px 10px',
          fontSize: 13, fontWeight: 700, letterSpacing: '0.02em',
          color: 'var(--tb-text)',
          display: 'flex', alignItems: 'center', gap: 7,
          flexShrink: 0,
        }}>
          <span style={{ color: 'var(--tb-blue)' }}>◉</span> TraceBack
        </div>

        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          {sessions.length > 0 && (
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
                onSelect={(i) => {
                  setPickedChapter(i === chapters.length ? null : i);
                  if (zoom !== 'chapters') setZoom('chapters');
                }}
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
            padding: 10,
            display: 'flex', flexDirection: 'column', gap: 6,
          }}>
            <RailButton onClick={feed.exportJson}>Export session</RailButton>
            <CopyRailButton onCopy={feed.copyReport} />
            <RailButton onClick={feed.exportHtml}>Share as HTML</RailButton>
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
          <div
            className={isLive ? 'live-dot' : ''}
            style={{
              width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
              background: isLive ? 'var(--tb-green)' : 'var(--tb-text-dim)',
            }}
          />
          <span style={{
            fontSize: 13, fontWeight: 600,
            color: display ? agentIdentity(display.id).color : 'var(--tb-text)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {display ? agentIdentity(display.id).name : 'No session'}
            {display && (
              <span style={{ fontWeight: 400, color: 'var(--tb-text-muted)' }}>
                {' '}· {display.label}
              </span>
            )}
          </span>
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
          {hasData && <ZoomControl withChapters zoom={zoom} onChange={setZoom} />}
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
                <SessionOdometer
                  nodes={traceNodes}
                  awaitingInput={archived ? undefined : display?.awaitingInput}
                  isLive={isLive}
                  anomaly={anomaly}
                  anomalyCount={display?.anomalyHistory?.length ?? 0}
                  paused={display?.paused ?? false}
                  onPauseToggle={archived ? undefined : feed.pauseToggle}
                  onRedirect={archived ? undefined : feed.redirect}
                  realTokens={display?.contextTokens}
                  aiSummary={display?.aiSummary}
                  chatAnswer={chatAnswer}
                  chatLoading={chatLoading}
                  onChat={feed.chat}
                />

                {zoom === 'chapters' && selectedChapter && (
                  <PromptChapterView
                    chapter={selectedChapter}
                    isLive={isLive && selectedIndex === chapters.length}
                    isLast={selectedIndex === chapters.length}
                    realTokens={display?.contextTokens}
                    expandedId={expandedId}
                    onToggle={(id) => setExpandedId(expandedId === id ? null : id)}
                  />
                )}

                {zoom !== 'chapters' && (
                  <>
                    {zoom !== 'map' && (
                      <ObjectiveHeader
                        plan={display?.plan}
                        nodes={traceNodes}
                        isLive={isLive}
                      />
                    )}

                    {zoom === 'map' ? (
                      <SessionMap
                        grid={wide}
                        nodes={traceNodes}
                        plan={display?.plan}
                        history={display?.anomalyHistory}
                        isLive={isLive}
                      />
                    ) : (
                      <Timeline
                        nodes={traceNodes}
                        anomaly={anomaly}
                        history={display?.anomalyHistory}
                        expandedId={expandedId}
                        expandAll={zoom === 'detail'}
                        isLive={isLive}
                        onToggle={(id) => setExpandedId(expandedId === id ? null : id)}
                      />
                    )}
                  </>
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

function RailButton({ children, onClick }: {
  children: React.ReactNode; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        fontSize: 10.5, fontFamily: 'var(--tb-ui-font)', fontWeight: 500,
        padding: '6px 10px', cursor: 'pointer',
        textAlign: 'left',
        background: hovered ? 'rgba(88,166,255,0.07)' : 'transparent',
        color: hovered ? 'var(--tb-blue)' : 'var(--tb-text-muted)',
        border: `1px solid ${hovered ? 'rgba(88,166,255,0.35)' : 'var(--tb-border)'}`,
        borderRadius: 4,
        transition: 'color 0.1s, border-color 0.1s, background 0.1s',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </button>
  );
}

function CopyRailButton({ onCopy }: { onCopy: () => Promise<void> }) {
  const [copied, setCopied] = useState(false);
  return (
    <RailButton onClick={async () => {
      await onCopy();
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }}>
      {copied ? '✓ Copied to clipboard' : 'Markdown report'}
    </RailButton>
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
