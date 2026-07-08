import { TimelineNode, formatDuration } from './components/TimelineCard';
import { computeFileChanges, summarizeChanges } from './fileChanges';
import { computeMetrics, formatTokens } from './metrics';
import { anomaliesFor, computeChapters, pendingPlanItems, PromptChapter } from './chapters';
import type { AnomalyRecordUI } from './useSessionFeed';

/**
 * Self-contained shareable session page: no JS, no external assets, inline
 * CSS matching the TraceBack palette. Structured as prompt chapters — each
 * user prompt is a section with its tasks, actions (with intents), and any
 * anomalies flagged inline — so it reads as a document, not a data dump.
 */

interface ExportSession {
  label:      string;
  startedAt:  number;
  stopped:    boolean;
  nodes:      TimelineNode[];
  aiSummary?: string;
  contextTokens?: number;
  anomalyHistory?: AnomalyRecordUI[];
}

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const STATUS_DOT: Record<string, string> = {
  success: '#3fb950',
  error:   '#f85149',
  pending: '#d29922',
  thinking:'#58a6ff',
};

function actionRow(n: TimelineNode): string {
  const dot = STATUS_DOT[n.status] ?? '#7d8590';
  const dur = formatDuration(n.durationMs);
  const intent = n.intent ? `<div class="intent">${esc(n.intent)}</div>` : '';
  const raw = n.detail
    ? `<details><summary>output</summary><pre>${esc(n.detail.slice(0, 4000))}</pre></details>`
    : '';
  const batch = n.isBatch && n.batchItems
    ? `<div class="batch">${n.batchItems.map((b) =>
        `<div><i style="background:${STATUS_DOT[b.status] ?? '#7d8590'}"></i>${esc(b.label)}${
          b.intent ? ` <span class="bintent">— ${esc(b.intent)}</span>` : ''
        }</div>`
      ).join('')}</div>`
    : '';
  return `<div class="row${n.status === 'error' ? ' err' : ''}">
  <span class="dot" style="background:${dot}"></span>
  <span class="tool">${esc(n.toolName)}</span>
  <span class="lbl">${esc(n.isBatch ? `${n.count} steps` : n.label)}</span>
  ${n.count > 1 && !n.isBatch ? `<span class="x">×${n.count}</span>` : ''}
  ${dur ? `<span class="dur">${esc(dur)}</span>` : ''}
  ${intent}${batch}${raw}
</div>`;
}

function anomalyBox(a: AnomalyRecordUI): string {
  const sev = a.severity ?? (a.type === 'stall' ? 'medium' : 'high');
  return `<div class="anombox ${sev}">
  <b>⚠ ${esc(a.title ?? a.type.replace(/_/g, ' '))}</b>
  <span>${esc(a.description ?? a.reason)}</span>
  <small>${new Date(a.detectedAt).toLocaleTimeString()}</small>
</div>`;
}

function chapterSection(c: PromptChapter, records: AnomalyRecordUI[]): string {
  const anomalies = anomaliesFor(c, records);
  const done  = c.plan.filter((p) => p.status === 'completed').length;
  const total = c.plan.length;
  const dur   = formatDuration(c.durationMs);

  const meta = [
    `${c.actionCount} action${c.actionCount === 1 ? '' : 's'}`,
    c.errorCount ? `${c.errorCount} error${c.errorCount === 1 ? '' : 's'}` : null,
    dur,
    total ? `${done}/${total} tasks` : null,
  ].filter(Boolean).join(' · ');

  const groups = c.taskGroups.map((g) => {
    const rows = g.nodes.map(actionRow).join('\n');
    if (!g.objective) return rows;
    const badge =
      g.status === 'completed' ? '<span class="tk done">done</span>'
      : g.status === 'in_progress' ? '<span class="tk act">active</span>'
      : '';
    return `<div class="task">
  <div class="taskhead">${esc(g.objective)} ${badge}
    ${g.files.length ? `<span class="files">${esc(g.files.join(' · '))}</span>` : ''}
  </div>
  ${rows}
</div>`;
  }).join('\n');

  const pending = pendingPlanItems(c).map((p) =>
    `<div class="task pending"><div class="taskhead">${esc(p.content)} <span class="tk">pending</span></div></div>`
  ).join('\n');

  return `<section>
  <div class="prompt">
    <div class="who">Prompt ${c.index}</div>
    <div class="ptext">${esc(c.text)}</div>
    <div class="pmeta">${esc(meta)}</div>
  </div>
  ${anomalies.map(anomalyBox).join('\n')}
  ${groups}
  ${pending}
</section>`;
}

export function buildSessionHtml(s: ExportSession): string {
  const m        = computeMetrics(s.nodes);
  const changes  = computeFileChanges(s.nodes);
  const chapters = computeChapters(s.nodes);
  const records  = s.anomalyHistory ?? [];
  const date     = new Date(s.startedAt).toLocaleString();

  const stats: [string, string][] = [
    ['prompts', String(chapters.length)],
    ['time',    formatDuration(m.totalDurationMs) ?? '—'],
    ['actions', String(m.toolCount)],
    ['errors',  String(m.errorCount)],
    ['anomalies', String(records.length)],
    ['tokens',  formatTokens(s.contextTokens ?? m.estTokens)],
  ];

  const statHtml = stats
    .map(([k, v]) => `<div class="stat"><b>${esc(v)}</b><span>${esc(k)}</span></div>`)
    .join('');

  const sections = chapters.map((c) => chapterSection(c, records)).join('\n');

  const changesHtml = changes.length
    ? `<h2>Files changed <small>${esc(summarizeChanges(changes))}</small></h2>
<table>${changes.map((c) =>
    `<tr><td class="k ${c.kind}">${c.kind}</td><td class="p">${esc(c.path)}</td><td class="n"><em>+${c.linesAdded}</em> <del>−${c.linesRemoved}</del></td></tr>`
  ).join('')}</table>`
    : '';

  const summaryHtml = s.aiSummary
    ? `<h2>Summary</h2><p class="sum">${esc(s.aiSummary)}</p>`
    : '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>TraceBack — ${esc(s.label)}</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#07090d; color:#e6edf3;
         font:13px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 40px 24px 80px; }
  header { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  h1 { font-size:18px; margin:0; }
  h1 small { color:#7d8590; font-weight:400; font-size:12px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
       color:#7d8590; margin:36px 0 10px; }
  h2 small { text-transform:none; letter-spacing:0; font-weight:400; }
  .stats { display:flex; gap:24px; margin:20px 0 8px; flex-wrap:wrap; }
  .stat b { display:block; font-size:16px; font-family:ui-monospace,monospace; }
  .stat span { font-size:9px; text-transform:uppercase; letter-spacing:.08em; color:#3d444d; }
  section { margin-top:34px; }
  .prompt { padding:12px 14px; border-radius:6px;
            background:rgba(88,166,255,0.07); border:1px solid rgba(88,166,255,0.25);
            border-left:3px solid #58a6ff; margin-bottom:12px; }
  .who { font-size:9px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
         color:#58a6ff; margin-bottom:4px; }
  .ptext { font-size:13.5px; line-height:1.55; white-space:pre-wrap; word-break:break-word; }
  .pmeta { margin-top:6px; font-size:10px; color:#7d8590; font-family:ui-monospace,monospace; }
  .task { border:1px solid #21262d; border-radius:6px; padding:8px 10px; margin-bottom:10px; }
  .task.pending { border-style:dashed; color:#7d8590; }
  .taskhead { font-size:12px; font-weight:600; margin-bottom:6px; }
  .tk { font-size:8.5px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
        border:1px solid #30363d; border-radius:3px; padding:1px 5px; color:#7d8590; margin-left:6px; }
  .tk.done { color:#3fb950; border-color:rgba(63,185,80,.4); }
  .tk.act { color:#58a6ff; border-color:rgba(88,166,255,.4); }
  .files { font-size:10px; color:#7d8590; font-weight:400; font-family:ui-monospace,monospace; margin-left:8px; }
  .row { position:relative; background:#0d1117; border:1px solid #21262d; border-radius:5px;
         padding:8px 12px 8px 30px; margin-bottom:6px; }
  .row.err { background:rgba(248,81,73,0.06); border-color:rgba(248,81,73,0.4); }
  .dot { position:absolute; left:12px; top:14px; width:8px; height:8px; border-radius:50%; }
  .tool { font-size:9.5px; font-weight:600; text-transform:uppercase; letter-spacing:.05em;
          color:#7d8590; margin-right:8px; }
  .lbl { font-size:12.5px; }
  .x, .dur { font-size:10px; color:#7d8590; font-family:ui-monospace,monospace; margin-left:8px; }
  .intent { margin-top:4px; font-size:11px; color:#7d8590; }
  .batch { margin-top:6px; padding-left:2px; }
  .batch div { font-size:11px; color:#7d8590; padding:1px 0; }
  .batch i { display:inline-block; width:5px; height:5px; border-radius:50%; margin-right:7px; }
  .bintent { color:#57606a; }
  .anombox { border-radius:5px; padding:8px 12px; margin-bottom:10px; font-size:11.5px;
             display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }
  .anombox.high { background:rgba(248,81,73,0.08); border:1px solid rgba(248,81,73,0.4); color:#ffa198; }
  .anombox.medium { background:rgba(210,153,34,0.08); border:1px solid rgba(210,153,34,0.4); color:#d29922; }
  .anombox small { color:#7d8590; margin-left:auto; }
  details { margin-top:6px; }
  summary { font-size:10px; color:#58a6ff; cursor:pointer; }
  pre { background:#0a0e14; border:1px solid #21262d; border-radius:4px; padding:8px;
        font-size:10.5px; color:#7ee787; white-space:pre-wrap; word-break:break-word;
        max-height:240px; overflow:auto; }
  table { width:100%; border-collapse:collapse; }
  td { padding:5px 8px; border-bottom:1px solid #161b22; font-size:12px; }
  .k { width:70px; font-size:9.5px; font-weight:600; text-transform:uppercase; }
  .k.created { color:#3fb950; } .k.modified { color:#d29922; }
  .p { font-family:ui-monospace,monospace; font-size:11.5px; word-break:break-all; }
  .n { width:90px; text-align:right; font-family:ui-monospace,monospace; font-size:11px; }
  .n em { color:#3fb950; font-style:normal; } .n del { color:#f85149; text-decoration:none; }
  .sum { color:#7d8590; font-style:italic; }
  footer { margin-top:48px; font-size:10px; color:#3d444d; }
  footer a { color:#58a6ff; text-decoration:none; }
</style>
</head>
<body>
<main>
  <header>
    <h1>${esc(s.label)} <small>· ${esc(date)} · ${s.stopped ? 'finished' : 'live at export'}</small></h1>
  </header>
  <div class="stats">${statHtml}</div>
  ${sections}
  ${changesHtml}
  ${summaryHtml}
  <footer>Recorded with <a href="https://marketplace.visualstudio.com/items?itemName=madiyarzhunussov.traceback-ai">TraceBack</a> — live observability for AI coding agents.</footer>
</main>
</body>
</html>`;
}
