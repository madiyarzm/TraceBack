import { useEffect, useRef, useState } from 'react';
import { toPng } from 'html-to-image';

import TimelineCard, { TimelineNode } from './components/TimelineCard';
import SessionOdometer, { AnomalyStateUI } from './components/SessionOdometer';
import EmptyState from './components/EmptyState';
import SessionPicker, { SessionSummary } from './components/SessionPicker';
import Toolbar from './components/Toolbar';
import { summarizeOutput, copyText } from './payloadParser';
import { formatTokens, formatCost, computeMetrics } from './metrics';
import { formatDuration } from './components/TimelineCard';
import vscode from './vscodeApi';

interface AnomalyRecordUI {
  type:            'repeater' | 'error_thrash' | 'stall';
  reason:          string;
  flaggedEventIds: string[];
  detectedAt:      number;
}

interface FullSessionData extends SessionSummary {
  nodes:           TimelineNode[];
  anomaly?:        AnomalyStateUI;
  anomalyHistory?: AnomalyRecordUI[];
  aiSummary?:      string;
  contextTokens?:  number;
  cwd?:            string;
}

interface SessionUpdateMessage {
  type: 'session_update';
  session: { id: string };
  allSessions: FullSessionData[];
}

interface LlmResponseMessage {
  type:   'llm_response';
  answer: string;
}

/**
 * Serializes a session as a markdown post-mortem: pasteable into GitHub
 * issues, PR descriptions, or a fresh agent session as handoff context.
 */
function buildSessionReport(s: FullSessionData): string {
  const real = s.nodes.filter((n) => n.toolName !== '__thinking__');
  const m    = computeMetrics(s.nodes);

  const lines: string[] = [
    `# TraceBack session report — ${s.label}`,
    '',
    `| total time | actions | errors | anomalies | tokens | est. cost |`,
    `|---|---|---|---|---|---|`,
    `| ${formatDuration(m.totalDurationMs) ?? '—'} | ${m.toolCount} | ${m.errorCount} | ${s.anomalyHistory?.length ?? 0} | ${formatTokens(s.contextTokens ?? m.estTokens)} | ${formatCost(((s.contextTokens ?? m.estTokens) / 1_000_000) * 6)} |`,
    '',
    '## Timeline',
    '',
  ];

  real.forEach((n, i) => {
    const icon = n.status === 'error' ? '❌' : n.status === 'pending' ? '⏳' : '✅';
    const dur  = formatDuration(n.durationMs);
    lines.push(`${i + 1}. ${icon} **${n.toolName}** — ${n.isBatch ? `${n.count} steps` : n.label}${dur ? ` _(${dur})_` : ''}`);
    const outcome = summarizeOutput(n.detail, n.status === 'error');
    if (outcome) lines.push(`   - ${outcome}`);
    if (n.isBatch && n.batchItems) {
      for (const item of n.batchItems) lines.push(`   - ${item.status === 'error' ? '❌' : '·'} ${item.label}`);
    }
  });

  if (s.anomalyHistory && s.anomalyHistory.length > 0) {
    lines.push('', '## Anomalies', '');
    for (const rec of s.anomalyHistory) {
      lines.push(`- ⚠ ${rec.reason} _(${new Date(rec.detectedAt).toLocaleTimeString()})_`);
    }
  }

  if (s.aiSummary) lines.push('', '## Summary', '', s.aiSummary);

  return lines.join('\n');
}

export default function App() {
  // Source of truth: the full fleet snapshot from the extension, plus which
  // session the user pinned (null = follow the most recently updated one).
  const [sessions, setSessions]     = useState<FullSessionData[]>([]);
  const [latestId, setLatestId]     = useState<string | null>(null);
  const [pinnedId, setPinnedId]     = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [chatAnswer, setChatAnswer]   = useState<string | undefined>(undefined);
  const [chatLoading, setChatLoading] = useState(false);

  const timelineRef = useRef<HTMLDivElement>(null);
  const scrollRef   = useRef<HTMLDivElement>(null);
  const followRef   = useRef(true);    // stick to bottom while live
  const pinnedRef   = useRef<string | null>(null);
  pinnedRef.current = pinnedId;

  useEffect(() => {
    function handleMessage(event: MessageEvent) {
      const message = event.data as SessionUpdateMessage | LlmResponseMessage;

      if (message.type === 'llm_response') {
        setChatAnswer(message.answer);
        setChatLoading(false);
        return;
      }

      if (message.type !== 'session_update') return;
      const msg = message as SessionUpdateMessage;

      setSessions(msg.allSessions ?? []);
      setLatestId(msg.session.id);

      // Auto-scroll only if the update belongs to the session being shown
      const showingId = pinnedRef.current ?? msg.session.id;
      if (followRef.current && msg.session.id === showingId) {
        requestAnimationFrame(() => {
          scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
        });
      }
    }

    window.addEventListener('message', handleMessage);
    vscode.postMessage({ type: 'ready' });
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  // ── Derive the displayed session ──────────────────────────────────────────
  const displayId = pinnedId ?? latestId;
  const display   =
    sessions.find((s) => s.id === displayId) ??
    sessions.find((s) => s.id === latestId) ??
    sessions[0] ??
    null;

  const traceNodes = display?.nodes ?? [];
  const hasData    = traceNodes.length > 0;
  const isLive     = display ? !display.stopped : false;
  const anomaly    = display?.anomaly;
  const realCount  = traceNodes.filter((n) => n.toolName !== '__thinking__').length;
  const flaggedIds = new Set(anomaly?.flaggedEventIds ?? []);

  // Permanent evidence trail: eventId → reason of the anomaly it was part of.
  // Cards keep their ⚠ tag even after the live anomaly self-clears.
  const history = display?.anomalyHistory ?? [];
  const historyByEvent = new Map<string, string>();
  for (const rec of history) {
    for (const id of rec.flaggedEventIds) {
      if (!historyByEvent.has(id)) historyByEvent.set(id, rec.reason);
    }
  }
  function historyReasonFor(node: TimelineNode): string | undefined {
    for (const id of node.eventIds ?? []) {
      const reason = historyByEvent.get(id);
      if (reason) return reason;
    }
    return undefined;
  }

  const pickerSessions: SessionSummary[] = sessions.map((s) => ({
    id:        s.id,
    label:     s.label,
    startedAt: s.startedAt,
    nodeCount: s.nodeCount,
    stopped:   s.stopped,
    anomalous: !!s.anomaly?.isAnomalous,
  }));

  // ── Handlers ──────────────────────────────────────────────────────────────
  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  }

  function handleSelectSession(id: string | null) {
    setPinnedId(id);
    setExpandedId(null);
    followRef.current = true;
  }

  function handleChat(question: string) {
    setChatLoading(true);
    setChatAnswer(undefined);
    const nodeContext = traceNodes
      .filter((n) => n.toolName !== '__thinking__')
      .map((n) => `${n.toolName}: ${n.label} [${n.status}]`)
      .join('\n');
    vscode.postMessage({ type: 'llm_query', question, nodeContext });
  }

  async function handleExportPng() {
    const container = timelineRef.current;
    if (!container) return;
    try {
      const url = await toPng(container, { backgroundColor: '#07090d', pixelRatio: 2 });
      const a = document.createElement('a');
      a.download = `traceback-${Date.now()}.png`;
      a.href = url;
      a.click();
    } catch (err) {
      console.error('PNG export failed:', err);
    }
  }

  function handleExportJson() {
    if (!display) return;
    const blob = new Blob([JSON.stringify(display, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.download = `traceback-${Date.now()}.json`;
    a.href = url;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleCopyReport() {
    if (!display) return;
    await copyText(buildSessionReport(display));
  }

  function handleClear() {
    vscode.postMessage({ type: 'clear_session' });
    setSessions([]);
    setLatestId(null);
    setPinnedId(null);
    setExpandedId(null);
    setChatAnswer(undefined);
    followRef.current = true;
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100vw', height: '100vh' }}>
      <SessionPicker
        sessions={pickerSessions}
        displayId={display?.id ?? null}
        pinnedId={pinnedId}
        onSelect={handleSelectSession}
      />

      {hasData && (
        <Toolbar
          isLive={isLive}
          nodeCount={realCount}
          onExportPng={handleExportPng}
          onExportJson={handleExportJson}
          onCopyReport={handleCopyReport}
          onClear={handleClear}
        />
      )}

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        style={{ position: 'relative', flex: 1, minHeight: 0, overflowY: 'auto' }}
      >
        {!hasData ? (
          <EmptyState />
        ) : (
          <div ref={timelineRef}>
            <SessionOdometer
              nodes={traceNodes}
              isLive={isLive}
              anomaly={anomaly}
              anomalyCount={history.length}
              realTokens={display?.contextTokens}
              aiSummary={display?.aiSummary}
              chatAnswer={chatAnswer}
              chatLoading={chatLoading}
              onChat={handleChat}
            />

            {/* ── Timeline ── */}
            <div style={{ position: 'relative', padding: '10px 12px 24px 14px' }}>
              <div style={{
                position: 'absolute',
                left: 18, top: 10, bottom: 24,
                width: 1,
                background: 'var(--tb-border)',
              }} />

              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {traceNodes.map((node) => (
                  <TimelineCard
                    key={node.id}
                    node={node}
                    expanded={expandedId === node.id}
                    flagged={
                      !!anomaly?.isAnomalous &&
                      (node.eventIds?.some((id) => flaggedIds.has(id)) ?? false)
                    }
                    historyReason={historyReasonFor(node)}
                    onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Anomaly red-tint overlay */}
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
