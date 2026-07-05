import TimelineCard, { TimelineNode } from './TimelineCard';
import { AnomalyStateUI } from './SessionOdometer';
import { AnomalyRecordUI } from '../useSessionFeed';

interface Props {
  nodes:       TimelineNode[];
  anomaly?:    AnomalyStateUI;
  history?:    AnomalyRecordUI[];
  expandedId:  string | null;
  onToggle:    (id: string) => void;
  /** Detail zoom: every card open — diffs, commands, raw output inline. */
  expandAll?:  boolean;
  /** Live sessions get the phosphor-green rail. */
  isLive?:     boolean;
}

/**
 * The vertical card timeline with its left rail, plus the live-anomaly
 * flagging and the permanent evidence-trail tags. Shared by both layouts.
 */
export default function Timeline({ nodes, anomaly, history = [], expandedId, onToggle, expandAll = false, isLive = false }: Props) {
  const flaggedIds = new Set(anomaly?.flaggedEventIds ?? []);

  // Permanent evidence trail: eventId → reason of the anomaly it was part of.
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

  return (
    <div style={{ position: 'relative', padding: '10px 12px 24px 14px' }}>
      <div
        className={isLive ? 'tb-rail-live' : undefined}
        style={{
          position: 'absolute',
          left: 18, top: 10, bottom: 24,
          width: 1,
          background: 'var(--tb-border)',
        }}
      />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {nodes.map((node) => (
          <TimelineCard
            key={node.id}
            node={node}
            expanded={expandAll || expandedId === node.id}
            flagged={
              !!anomaly?.isAnomalous &&
              (node.eventIds?.some((id) => flaggedIds.has(id)) ?? false)
            }
            historyReason={historyReasonFor(node)}
            onToggle={onToggle}
          />
        ))}
      </div>
    </div>
  );
}
