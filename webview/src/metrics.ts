import { TimelineNode } from './components/TimelineCard';
import { isExpectedStumble } from './payloadParser';

/**
 * Aggregate session metrics, derived purely from the node list the extension
 * already streams — no protocol changes needed.
 *
 * Cost/tokens are HEURISTIC: Claude Code hook payloads carry no token usage,
 * so we approximate from tool I/O volume (chars / 4 ≈ tokens, the standard
 * rule of thumb for English/code) at a blended $/MTok rate. `isEstimated`
 * stays true until a V2 data source provides real usage, at which point the
 * UI needs zero changes — only this module does.
 */
export interface SessionMetrics {
  totalDurationMs: number;
  toolCount:       number;
  errorCount:      number;
  estTokens:       number;
  estCostUsd:      number;
  isEstimated:     true;
}

const CHARS_PER_TOKEN   = 4;
const EST_USD_PER_MTOK  = 6; // blended in/out rate, order-of-magnitude only

export function computeMetrics(nodes: TimelineNode[]): SessionMetrics {
  const real = nodes.filter((n) => !n.toolName.startsWith('__'));

  const first = real[0];
  const last  = real[real.length - 1];
  const totalDurationMs = first && last
    ? Math.max(0, last.timestamp + (last.durationMs ?? 0) - first.timestamp)
    : 0;

  let chars = 0;
  for (const n of real) {
    if (n.toolInput) chars += JSON.stringify(n.toolInput).length;
    if (n.detail)    chars += n.detail.length;
  }

  const estTokens = Math.round(chars / CHARS_PER_TOKEN);

  return {
    totalDurationMs,
    toolCount:  real.reduce((sum, n) => sum + n.count, 0),
    errorCount: real.filter((n) => n.status === 'error' && !isExpectedStumble(n)).length,
    estTokens,
    estCostUsd: (estTokens / 1_000_000) * EST_USD_PER_MTOK,
    isEstimated: true,
  };
}

export function formatTokens(n: number): string {
  if (n < 1000)      return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

export function formatCost(usd: number): string {
  if (usd < 0.005) return '<1¢';
  if (usd < 1)     return `${Math.round(usd * 100)}¢`;
  return `$${usd.toFixed(2)}`;
}
