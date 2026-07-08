import { useEffect, useMemo, useState } from 'react';

import EmptyState from './components/EmptyState';
import FileChangesPanel from './components/FileChangesPanel';
import PromptList from './components/PromptList';
import PromptChapterView from './components/PromptChapterView';
import ReviewPanel from './components/ReviewPanel';
import RightPanel from './components/RightPanel';
import { formatDuration } from './components/TimelineCard';
import { AlertIcon } from './components/Icons';
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

  // ── Replay: step through the session as it happened ──
  // A cursor over the real (non-thinking) nodes; every derived view below is
  // computed from the sliced list, so the whole UI time-travels together.
  const replayNodes = useMemo(
    () => traceNodes.filter((n) => n.toolName !== '__thinking__'),
    [traceNodes],
  );
  const [replayCursor, setReplayCursor] = useState<number | null>(null);
  useEffect(() => setReplayCursor(null), [display?.id]);
  const replayActive = replayCursor !== null && !isLive;
  const effectiveNodes = replayActive
    ? replayNodes.slice(0, replayCursor)
    : traceNodes;

  // ── Prompt chapters ──
  const chapters = useMemo(() => computeChapters(effectiveNodes), [effectiveNodes]);
  const anomalyCounts = useMemo(() => {
    const map = new Map<number, number>();
    for (const c of chapters) map.set(c.index, anomaliesFor(c, records).length);
    return map;
  }, [chapters, records]);

  // Focused chapter: user-picked, else follow the latest prompt.
  const [pickedChapter, setPickedChapter] = useState<number | null>(null);
  useEffect(() => setPickedChapter(null), [display?.id]); // reset per session

  // Net-change review mode: replaces the chapter body in the main column.
  const [reviewOpen, setReviewOpen] = useState(false);
  useEffect(() => setReviewOpen(false), [display?.id]);
  const hasEdits = useMemo(
    () => traceNodes.some((n) =>
      ['Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'FileWrite'].includes(n.toolName)),
    [traceNodes],
  );
  function openReview() {
    feed.requestReview();
    setReviewOpen(true);
  }
  const selectedIndex =
    replayActive ? chapters.length   // replay always follows the moving edge
    : pickedChapter !== null && pickedChapter <= chapters.length
      ? pickedChapter
      : chapters.length;
  const selectedChapter = chapters.find((c) => c.index === selectedIndex) ?? null;

  // ── Replay transport ──
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replaySpeed, setReplaySpeed]     = useState(1);
  useEffect(() => {
    if (!replayActive || !replayPlaying) return;
    const t = setInterval(() => {
      setReplayCursor((c) => {
        if (c === null || c >= replayNodes.length) { setReplayPlaying(false); return c; }
        return c + 1;
      });
    }, 900 / replaySpeed);
    return () => clearInterval(t);
  }, [replayActive, replayPlaying, replaySpeed, replayNodes.length]);

  useEffect(() => {
    if (!replayActive) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'ArrowRight') setReplayCursor((c) => Math.min((c ?? 0) + 1, replayNodes.length));
      if (e.key === 'ArrowLeft')  setReplayCursor((c) => Math.max((c ?? 1) - 1, 1));
      if (e.key === 'Escape')     setReplayCursor(null);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [replayActive, replayNodes.length]);

  function toggleReplay() {
    setReviewOpen(false);
    setReplayPlaying(false);
    setReplayCursor((c) => (c === null ? Math.min(1, replayNodes.length) : null));
  }

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
              {display && (
                <span style={{
                  fontSize: 11.5, fontWeight: 500,
                  fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
                  color: 'var(--tb-text-dim)', marginLeft: 7,
                  letterSpacing: '0.04em',
                }}>
                  {agentIdentity(display.id).tag}
                </span>
              )}
            </div>
            {display && (
              <div style={{
                fontSize: 12.5, color: 'var(--tb-text-muted)', marginTop: 2,
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
            <div style={{ padding: '10px 14px', fontSize: 11.5, color: 'var(--tb-text-dim)', lineHeight: 1.5 }}>
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
            <ExportMenu
              onJson={feed.exportJson}
              onMarkdown={feed.copyReport}
              onHtml={feed.exportHtml}
              onPng={feed.exportPng}
            />
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
              fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
              color: 'var(--tb-text-muted)',
              border: '1px solid var(--tb-border-2)',
              borderRadius: 3, padding: '1px 6px',
              textTransform: 'uppercase',
            }}>
              archived
            </span>
          )}
          <div style={{ flex: 1 }} />
          {hasData && !isLive && (
            <HeaderToggle
              label={replayActive ? 'exit replay' : 'replay'}
              active={replayActive}
              title="Step through the session as it happened (←/→ keys)"
              onClick={toggleReplay}
            />
          )}
          {hasData && !archived && hasEdits && !replayActive && (
            <ReviewButton active={reviewOpen} onClick={() => reviewOpen ? setReviewOpen(false) : openReview()} />
          )}
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
                {reviewOpen ? (
                  <ReviewPanel
                    files={feed.reviewFiles}
                    nodes={traceNodes}
                    clickable={!archived}
                    onClose={() => setReviewOpen(false)}
                  />
                ) : selectedChapter && (
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

                {!wide && <FileChangesPanel nodes={effectiveNodes} defaultOpen clickable={!archived} />}

                {replayActive && (
                  <ReplayBar
                    cursor={replayCursor ?? 0}
                    total={replayNodes.length}
                    playing={replayPlaying}
                    speed={replaySpeed}
                    currentLabel={
                      replayCursor
                        ? `${replayNodes[replayCursor - 1]?.toolName ?? ''} — ${replayNodes[replayCursor - 1]?.label ?? ''}`
                        : ''
                    }
                    onStep={(d) => setReplayCursor((c) =>
                      Math.max(1, Math.min((c ?? 1) + d, replayNodes.length)))}
                    onJump={(to) => setReplayCursor(Math.max(1, Math.min(to, replayNodes.length)))}
                    onPlayToggle={() => setReplayPlaying((v) => !v)}
                    onSpeed={() => setReplaySpeed((s) => (s === 4 ? 1 : s * 2))}
                  />
                )}
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
                    nodes={effectiveNodes}
                    chapters={chapters}
                    selectedIndex={selectedIndex}
                    records={records}
                    ledger={display?.ledger ?? []}
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

          {anomaly?.isAnomalous && anomaly.type !== 'stall' && (
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
      fontSize: 10, fontWeight: 700,
      letterSpacing: '0.1em', textTransform: 'uppercase',
      color: 'var(--tb-text-dim)',
    }}>{children}</div>
  );
}

function railCardStyle(active: boolean, hovered: boolean): React.CSSProperties {
  return {
    margin: '0 8px 4px',
    padding: '8px 11px',
    borderRadius: 6,
    background: active ? 'var(--tb-surface-2)' : hovered ? 'rgba(22,27,34,0.55)' : 'transparent',
    cursor: 'pointer',
    transition: 'background 0.12s ease, box-shadow 0.12s ease',
  };
}

function SessionRailCard({ session, active, onClick }: {
  session: FullSessionData; active: boolean; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  // A stall is a waiting state, not an alarm — keep the rail dot calm for it.
  const anomalous = !!session.anomaly?.isAnomalous && session.anomaly?.type !== 'stall';
  const live = !session.stopped;
  const identity = agentIdentity(session.id);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...railCardStyle(active, hovered),
        boxShadow: active ? `inset 2.5px 0 0 ${identity.color}` : 'none',
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
          fontSize: 13.5, fontWeight: active ? 600 : 400,
          color: identity.color,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          flex: 1, minWidth: 0,
        }}>
          {identity.name}
          <span style={{
            fontSize: 11, fontWeight: 500,
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
            color: 'var(--tb-text-dim)', marginLeft: 6,
          }}>
            {identity.tag}
          </span>
        </span>
        {anomalous && <span style={{ color: '#f85149', display: 'flex' }}><AlertIcon size={10} /></span>}
      </div>
      <div style={{
        fontSize: 10.5, color: 'var(--tb-text-muted)', paddingLeft: 13, marginTop: 1,
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
          fontSize: 13.5, fontWeight: active ? 600 : 400,
          color: 'var(--tb-text-muted)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          flex: 1, minWidth: 0,
        }}>
          <span style={{ color: agentIdentity(meta.id).color, opacity: 0.75 }}>
            {agentIdentity(meta.id).name}
          </span>
          <span style={{
            fontSize: 11, fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
            color: 'var(--tb-text-dim)', marginLeft: 5,
          }}>
            {agentIdentity(meta.id).tag}
          </span>
          {' '}· {meta.label}
        </span>
        {meta.errorCount > 0 && (
          <span style={{ fontSize: 10, color: '#f85149' }}>{meta.errorCount}✕</span>
        )}
        {meta.anomalyCount > 0 && (
          <span style={{ color: '#d29922', display: 'flex' }}><AlertIcon size={10} /></span>
        )}
      </div>
      <div style={{ fontSize: 10.5, color: 'var(--tb-text-dim)', marginTop: 1 }}>
        {new Date(meta.startedAt).toLocaleDateString()} · {meta.nodeCount} actions{dur ? ` · ${dur}` : ''}
      </div>
    </div>
  );
}

// ── Buttons ─────────────────────────────────────────────────────────────────


/**
 * One export entry point: a rail button opening an upward menu with the four
 * formats. Markdown copies (with inline confirmation); the rest download.
 */
function ExportMenu({ onJson, onMarkdown, onHtml, onPng }: {
  onJson: () => void;
  onMarkdown: () => Promise<void>;
  onHtml: () => void;
  onPng: () => void;
}) {
  const [open, setOpen]     = useState(false);
  const [copied, setCopied] = useState(false);

  function run(action: () => void) {
    action();
    setOpen(false);
  }

  async function runMarkdown() {
    await onMarkdown();
    setCopied(true);
    setTimeout(() => { setCopied(false); setOpen(false); }, 900);
  }

  return (
    <div style={{ position: 'relative' }}>
      {open && (
        <div
          onClick={() => setOpen(false)}
          style={{ position: 'fixed', inset: 0, zIndex: 30 }}
        />
      )}

      {open && (
        <div style={{
          position: 'absolute', bottom: 'calc(100% + 6px)', left: 0, right: 0,
          zIndex: 31,
          background: 'var(--tb-surface-2)',
          border: '1px solid var(--tb-border-2)',
          borderRadius: 8,
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          overflow: 'hidden',
          animation: 'cardBodyIn 0.12s ease-out',
        }}>
          <ExportMenuItem label="JSON file"       hint=".json"  onClick={() => run(onJson)} />
          <ExportMenuItem
            label={copied ? 'Copied to clipboard' : 'Markdown report'}
            hint={copied ? '✓' : 'copy'}
            onClick={() => void runMarkdown()}
          />
          <ExportMenuItem label="Shareable HTML"  hint=".html" onClick={() => run(onHtml)} />
          <ExportMenuItem label="PNG snapshot"    hint=".png"  onClick={() => run(onPng)} />
        </div>
      )}

      <ExportTrigger open={open} onClick={() => setOpen((v) => !v)} />
    </div>
  );
}

function ExportTrigger({ open, onClick }: { open: boolean; onClick: () => void }) {
  const [hovered, setHovered] = useState(false);
  const lit = open || hovered;
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 9,
        width: '100%', boxSizing: 'border-box',
        padding: '9px 12px',
        fontSize: 13, fontWeight: 600, fontFamily: 'var(--tb-ui-font)',
        textAlign: 'left',
        background: lit ? 'var(--tb-surface-2)' : 'transparent',
        color: lit ? 'var(--tb-text)' : 'var(--tb-text-muted)',
        border: `1px solid ${open ? 'var(--tb-border-2)' : 'var(--tb-border)'}`,
        borderRadius: 8,
        cursor: 'pointer',
        transition: 'background 0.12s ease, color 0.12s ease, border-color 0.12s ease',
      }}
    >
      <span style={{ display: 'flex', flexShrink: 0, opacity: 0.85 }}>
        <DownloadIcon size={14} />
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>Export session</span>
      <span style={{
        display: 'flex', flexShrink: 0,
        color: 'var(--tb-text-dim)',
        transform: open ? 'rotate(180deg)' : 'none',
        transition: 'transform 0.18s ease',
      }}>
        <ChevronUpGlyph />
      </span>
    </button>
  );
}

function ChevronUpGlyph() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round"
         style={{ display: 'block' }}>
      <path d="m18 15-6-6-6 6" />
    </svg>
  );
}

function ExportMenuItem({ label, hint, onClick }: {
  label: string; hint: string; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '9px 12px',
        fontSize: 13.5, fontFamily: 'var(--tb-ui-font)', fontWeight: 500,
        color: hovered ? 'var(--tb-text)' : 'var(--tb-text-muted)',
        background: hovered ? 'rgba(88,166,255,0.08)' : 'transparent',
        cursor: 'pointer',
        transition: 'background 0.1s ease, color 0.1s ease',
      }}
    >
      <span style={{ flex: 1 }}>{label}</span>
      <span style={{
        fontSize: 11, fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
        color: 'var(--tb-text-dim)',
      }}>
        {hint}
      </span>
    </div>
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


function HeaderToggle({ label, active, title, onClick }: {
  label: string; active: boolean; title: string; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const accent = '#a371f7'; // replay = time travel, keep it distinct from blue
  return (
    <button
      title={title}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        fontSize: 11.5, fontFamily: 'var(--tb-ui-font)', fontWeight: 600,
        padding: '3px 10px', cursor: 'pointer', marginRight: 4,
        background: active ? `${accent}1f` : hovered ? `${accent}12` : 'transparent',
        color: active || hovered ? accent : 'var(--tb-text-muted)',
        border: `1px solid ${active ? `${accent}88` : hovered ? `${accent}55` : 'var(--tb-border)'}`,
        borderRadius: 3,
        transition: 'color 0.1s, border-color 0.1s, background 0.1s',
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </button>
  );
}

/** Sticky transport bar for replay: step / play / speed, with the current
 *  action named so the viewer can predict-then-check. */
function ReplayBar({ cursor, total, playing, speed, currentLabel, onStep, onJump, onPlayToggle, onSpeed }: {
  cursor: number; total: number; playing: boolean; speed: number; currentLabel: string;
  onStep: (delta: number) => void;
  onJump: (to: number) => void;
  onPlayToggle: () => void;
  onSpeed: () => void;
}) {
  const btn: React.CSSProperties = {
    background: 'transparent',
    border: '1px solid var(--tb-border-2)',
    borderRadius: 6,
    color: 'var(--tb-text)',
    fontSize: 13, fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
    padding: '5px 10px', cursor: 'pointer',
    lineHeight: 1,
  };
  return (
    <div style={{
      position: 'sticky', bottom: 0, zIndex: 5,
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '10px 16px',
      background: 'var(--tb-surface)',
      borderTop: '1px solid var(--tb-border-2)',
      fontFamily: 'var(--tb-ui-font)',
    }}>
      <button style={btn} title="Jump to start" onClick={() => onJump(1)}>⏮</button>
      <button style={btn} title="Previous step (←)" onClick={() => onStep(-1)}>◀</button>
      <button
        style={{ ...btn, color: '#a371f7', borderColor: '#a371f788', minWidth: 34 }}
        title={playing ? 'Pause' : 'Auto-play'}
        onClick={onPlayToggle}
      >
        {playing ? '❚❚' : '▶'}
      </button>
      <button style={btn} title="Next step (→)" onClick={() => onStep(1)}>▶▶</button>
      <button style={btn} title="Jump to end" onClick={() => onJump(total)}>⏭</button>
      <button style={btn} title="Playback speed" onClick={onSpeed}>{speed}×</button>
      <span style={{
        fontSize: 12, fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
        color: 'var(--tb-text-muted)', flexShrink: 0,
      }}>
        {cursor} / {total}
      </span>
      <span style={{
        fontSize: 12.5, color: 'var(--tb-text)',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        minWidth: 0,
      }}>
        {currentLabel}
      </span>
    </div>
  );
}

function ReviewButton({ active, onClick }: { active: boolean; onClick: () => void }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      title="Net effect of this session: baseline → now diff per file, with reasoning"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        fontSize: 11.5, fontFamily: 'var(--tb-ui-font)', fontWeight: 600,
        padding: '3px 10px', cursor: 'pointer',
        background: active ? 'rgba(88,166,255,0.12)' : hovered ? 'rgba(88,166,255,0.07)' : 'transparent',
        color: active || hovered ? 'var(--tb-blue)' : 'var(--tb-text-muted)',
        border: `1px solid ${active ? 'rgba(88,166,255,0.55)' : hovered ? 'rgba(88,166,255,0.35)' : 'var(--tb-border)'}`,
        borderRadius: 3,
        transition: 'color 0.1s, border-color 0.1s, background 0.1s',
        whiteSpace: 'nowrap',
      }}
    >
      {active ? 'timeline' : 'review changes'}
    </button>
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
        fontSize: 11, fontFamily: 'var(--tb-ui-font)',
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
