import { useEffect, useRef, useState } from 'react';
import { toPng } from 'html-to-image';

import { TimelineNode, formatDuration } from './components/TimelineCard';
import { AnomalyStateUI } from './components/SessionOdometer';
import { SessionPlanUI } from './components/ObjectiveHeader';
import { SessionSummary } from './components/SessionPicker';
import { summarizeOutput, copyText } from './payloadParser';
import { formatTokens, formatCost, computeMetrics } from './metrics';
import { anomaliesFor, computeChapters, pendingPlanItems } from './chapters';
import { buildSessionHtml } from './htmlExport';
import vscode from './vscodeApi';

export interface AnomalyRecordUI {
  type:            string;
  severity?:       'high' | 'medium';
  title?:          string;
  description?:    string;
  reason:          string;
  flaggedEventIds: string[];
  detectedAt:      number;
}

export interface BuiltinGuardUI {
  key:         string;
  label:       string;
  description: string;
  enabled:     boolean;
}

export interface GuardsStateUI {
  builtins: BuiltinGuardUI[];
  custom:   string[];
}

export interface LedgerItemUI {
  text:      string;
  kind:      'decision' | 'assumption';
  timestamp: number;
}

export interface FullSessionData extends SessionSummary {
  nodes:           TimelineNode[];
  anomaly?:        AnomalyStateUI;
  anomalyHistory?: AnomalyRecordUI[];
  aiSummary?:      string;
  contextTokens?:  number;
  cwd?:            string;
  paused?:         boolean;
  awaitingInput?:  string;
  plan?:           SessionPlanUI;
  ledger?:         LedgerItemUI[];
}

/** One file in the net-change review: pre-session baseline vs. disk now. */
export interface ReviewFile {
  path:     string;
  /** Content before the agent's first touch; null = file was created. */
  baseline: string | null;
  /** Content on disk at review time; null = file was deleted. */
  current:  string | null;
}

export interface ArchivedMeta {
  id:           string;
  label:        string;
  startedAt:    number;
  endedAt:      number;
  nodeCount:    number;
  errorCount:   number;
  anomalyCount: number;
  tokens?:      number;
}

/**
 * Serializes a session as a markdown post-mortem, structured as prompt
 * chapters: pasteable into GitHub issues, PR descriptions, or a fresh agent
 * session as handoff context.
 */
export function buildSessionReport(s: FullSessionData): string {
  const m        = computeMetrics(s.nodes);
  const chapters = computeChapters(s.nodes);
  const records  = s.anomalyHistory ?? [];

  const lines: string[] = [
    `# TraceBack session report — ${s.label}`,
    '',
    `| prompts | total time | actions | errors | anomalies | tokens | est. cost |`,
    `|---|---|---|---|---|---|---|`,
    `| ${chapters.length} | ${formatDuration(m.totalDurationMs) ?? '—'} | ${m.toolCount} | ${m.errorCount} | ${records.length} | ${formatTokens(s.contextTokens ?? m.estTokens)} | ${formatCost(((s.contextTokens ?? m.estTokens) / 1_000_000) * 6)} |`,
  ];

  for (const c of chapters) {
    const done  = c.plan.filter((p) => p.status === 'completed').length;
    const total = c.plan.length;
    const dur   = formatDuration(c.durationMs);
    const meta  = [
      `${c.actionCount} actions`,
      c.errorCount ? `${c.errorCount} errors` : null,
      dur,
      total ? `${done}/${total} tasks` : null,
    ].filter(Boolean).join(' · ');

    lines.push('', `## P${c.index} — 🧑 ${c.text.replace(/\s+/g, ' ')}`, '', `_${meta}_`, '');

    for (const rec of anomaliesFor(c, records)) {
      const sev = rec.severity ?? (rec.type === 'stall' ? 'medium' : 'high');
      lines.push(`> ⚠ **${rec.title ?? rec.type}** (${sev}): ${rec.description ?? rec.reason}`, '');
    }

    for (const g of c.taskGroups) {
      if (g.objective) {
        const badge = g.status === 'completed' ? ' ✅' : g.status === 'in_progress' ? ' 🔵' : '';
        lines.push(`### ${g.objective}${badge}`);
        if (g.files.length) lines.push(`_files: ${g.files.join(', ')}_`);
        lines.push('');
      }
      let step = 0;
      for (const n of g.nodes) {
        step++;
        const icon = n.status === 'error' ? '❌' : n.status === 'pending' ? '⏳' : '✅';
        const d    = formatDuration(n.durationMs);
        lines.push(`${step}. ${icon} **${n.toolName}** — ${n.isBatch ? `${n.count} steps` : n.label}${d ? ` _(${d})_` : ''}`);
        if (n.intent) lines.push(`   - _${n.intent}_`);
        const outcome = summarizeOutput(n.detail, n.status === 'error', n);
        if (outcome) lines.push(`   - ${outcome}`);
        if (n.isBatch && n.batchItems) {
          for (const item of n.batchItems) lines.push(`   - ${item.status === 'error' ? '❌' : '·'} ${item.label}`);
        }
      }
      lines.push('');
    }

    for (const p of pendingPlanItems(c)) {
      lines.push(`- ⬜ ${p.content} _(pending)_`);
    }
  }

  if (s.aiSummary) lines.push('', '## Summary', '', s.aiSummary);

  return lines.join('\n');
}

function download(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.download = filename;
  a.href = url;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * All webview↔extension state in one place, shared by the sidebar and panel
 * layouts: live session feed, archived history, chat, and the export actions.
 */
export function useSessionFeed() {
  const [sessions, setSessions]     = useState<FullSessionData[]>([]);
  const [history, setHistory]       = useState<ArchivedMeta[]>([]);
  const [archived, setArchived]     = useState<FullSessionData | null>(null);
  const [latestId, setLatestId]     = useState<string | null>(null);
  const [pinnedId, setPinnedId]     = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [chatAnswer, setChatAnswer]   = useState<string | undefined>(undefined);
  const [chatLoading, setChatLoading] = useState(false);
  const [guards, setGuards]           = useState<GuardsStateUI>({ builtins: [], custom: [] });
  const [reviewFiles, setReviewFiles] = useState<ReviewFile[] | null>(null);

  const timelineRef = useRef<HTMLDivElement>(null);
  const scrollRef   = useRef<HTMLDivElement>(null);
  const followRef   = useRef(true);    // stick to bottom while live
  const pinnedRef   = useRef<string | null>(null);
  const archivedRef = useRef(false);
  pinnedRef.current   = pinnedId;
  archivedRef.current = archived !== null;

  useEffect(() => {
    function handleMessage(event: MessageEvent) {
      const message = event.data as { type: string; [k: string]: unknown };

      if (message.type === 'llm_response') {
        setChatAnswer(message.answer as string);
        setChatLoading(false);
        return;
      }

      if (message.type === 'guards_update') {
        setGuards((message.guards as GuardsStateUI) ?? { builtins: [], custom: [] });
        return;
      }

      if (message.type === 'history_update') {
        setHistory((message.history as ArchivedMeta[]) ?? []);
        return;
      }

      if (message.type === 'archived_session') {
        const s = message.session as FullSessionData & { stopped?: boolean };
        setArchived({ ...s, stopped: true });
        setExpandedId(null);
        return;
      }

      if (message.type === 'review_data') {
        setReviewFiles((message.files as ReviewFile[]) ?? []);
        return;
      }

      if (message.type !== 'session_update') return;

      setSessions((message.allSessions as FullSessionData[]) ?? []);
      setHistory((message.history as ArchivedMeta[]) ?? []);
      const id = (message.session as { id: string }).id;
      setLatestId(id);

      // Auto-scroll only when live-following the session being shown
      const showingId = pinnedRef.current ?? id;
      if (!archivedRef.current && followRef.current && id === showingId) {
        requestAnimationFrame(() => {
          scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
        });
      }
    }

    window.addEventListener('message', handleMessage);
    vscode.postMessage({ type: 'ready' });
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  // ── Derive the displayed session ────────────────────────────────────────
  const displayId = pinnedId ?? latestId;
  const liveDisplay =
    sessions.find((s) => s.id === displayId) ??
    sessions.find((s) => s.id === latestId) ??
    sessions[0] ??
    null;
  const display = archived ?? liveDisplay;

  // ── Handlers ────────────────────────────────────────────────────────────
  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  }

  function selectSession(id: string | null) {
    setArchived(null);
    setPinnedId(id);
    setExpandedId(null);
    setReviewFiles(null);
    followRef.current = true;
  }

  /** Ask the extension host for baseline-vs-now contents of every touched file. */
  function requestReview() {
    if (!display || archived) return;
    setReviewFiles(null); // stale data out — the panel shows a loading state
    vscode.postMessage({ type: 'request_review', sessionId: display.id });
  }

  function selectArchived(id: string) {
    vscode.postMessage({ type: 'load_archived', sessionId: id });
  }

  function chat(question: string) {
    if (!display) return;
    setChatLoading(true);
    setChatAnswer(undefined);
    const nodeContext = display.nodes
      .filter((n) => n.toolName !== '__thinking__')
      .map((n) => n.toolName === '__prompt__'
        ? `USER PROMPT: "${n.label}"`
        : `${n.toolName}: ${n.label} [${n.status}]`)
      .join('\n');
    vscode.postMessage({ type: 'llm_query', question, nodeContext });
  }

  async function exportPng() {
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

  function exportJson() {
    if (!display) return;
    download(`traceback-${Date.now()}.json`, JSON.stringify(display, null, 2), 'application/json');
  }

  function exportHtml() {
    if (!display) return;
    download(
      `traceback-${display.label.replace(/[^\w.-]/g, '_')}-${Date.now()}.html`,
      buildSessionHtml(display),
      'text/html',
    );
  }

  async function copyReport() {
    if (!display) return;
    await copyText(buildSessionReport(display));
  }

  function pauseToggle() {
    if (!display || archived) return;
    vscode.postMessage({
      type: display.paused ? 'resume_session' : 'pause_session',
      sessionId: display.id,
    });
  }

  function redirect(message: string) {
    if (!display || archived) return;
    vscode.postMessage({ type: 'redirect_session', sessionId: display.id, message });
  }

  function clear() {
    vscode.postMessage({ type: 'clear_session' });
    setSessions([]);
    setLatestId(null);
    setPinnedId(null);
    setArchived(null);
    setExpandedId(null);
    setChatAnswer(undefined);
    followRef.current = true;
  }

  function openFullPanel() {
    vscode.postMessage({ type: 'open_full_panel' });
  }

  function setGuard(key: string, enabled: boolean) {
    vscode.postMessage({ type: 'set_guard', key, enabled });
  }

  function addGuard(pattern: string) {
    vscode.postMessage({ type: 'add_custom_guard', pattern });
  }

  function removeGuard(pattern: string) {
    vscode.postMessage({ type: 'remove_custom_guard', pattern });
  }

  return {
    sessions, history, archived, display, displayId: display?.id ?? null, pinnedId,
    expandedId, setExpandedId,
    chatAnswer, chatLoading,
    guards, setGuard, addGuard, removeGuard,
    reviewFiles, requestReview,
    timelineRef, scrollRef,
    handleScroll, selectSession, selectArchived, chat,
    exportPng, exportJson, exportHtml, copyReport,
    pauseToggle, redirect, clear, openFullPanel,
  };
}
