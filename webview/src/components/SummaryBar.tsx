import { useMemo } from 'react';

interface TraceNodeData {
  toolName: string;
  status: 'pending' | 'success' | 'error' | 'thinking';
  count?: number;
}

interface Props {
  nodes: TraceNodeData[];
  aiSummary?: string;
  isLive: boolean;
}

function generateSummary(nodes: TraceNodeData[]): string {
  const counts: Record<string, number> = {};
  for (const n of nodes) {
    counts[n.toolName] = (counts[n.toolName] || 0) + (n.count ?? 1);
  }

  const parts = Object.entries(counts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3)
    .map(([tool, count]) => {
      if (tool === 'Bash')      return count > 1 ? `Running ${count} commands` : 'Running a command';
      if (tool === 'Read')      return count > 1 ? `Reading ${count} files`    : 'Reading a file';
      if (tool === 'Edit')      return count > 1 ? `Editing ${count} files`    : 'Editing a file';
      if (tool === 'Write')     return count > 1 ? `Writing ${count} files`    : 'Writing a file';
      if (tool === 'WebSearch' || tool === 'WebFetch')
                                return count > 1 ? `Fetching ${count} pages`   : 'Browsing the web';
      if (tool === 'Agent')     return count > 1 ? `Running ${count} agents`   : 'Spawning a subagent';
      if (tool === 'TodoRead' || tool === 'TodoWrite') return 'Managing tasks';
      return count > 1 ? `${tool} ×${count}` : tool;
    });

  if (parts.length === 0) return 'Starting…';
  return parts.join(' · ');
}

export default function SummaryBar({ nodes, aiSummary, isLive }: Props) {
  const realNodes  = nodes.filter(n => n.toolName !== '__thinking__');
  const errorCount = realNodes.filter(n => n.status === 'error').length;
  const summary    = useMemo(
    () => aiSummary ?? generateSummary(realNodes),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [nodes, aiSummary]
  );

  if (realNodes.length === 0) return null;

  return (
    <div style={{
      display:       'flex',
      alignItems:    'center',
      gap:           8,
      padding:       '4px 12px',
      borderBottom:  '1px solid var(--tb-border)',
      background:    'var(--tb-surface)',
      fontSize:      11,
      color:         'var(--tb-text-muted)',
      fontFamily:    'var(--tb-ui-font)',
      flexShrink:    0,
      minHeight:     26,
    }}>
      {isLive && (
        <span
          className="live-dot"
          style={{
            width:        6,
            height:       6,
            borderRadius: '50%',
            background:   'var(--tb-green)',
            flexShrink:   0,
            display:      'inline-block',
          }}
        />
      )}

      <span style={{
        overflow:      'hidden',
        textOverflow:  'ellipsis',
        whiteSpace:    'nowrap',
        flex:          1,
        color:         aiSummary ? 'var(--tb-text)' : 'var(--tb-text-muted)',
      }}>
        {summary}
      </span>

      {errorCount > 0 && (
        <span style={{
          fontSize:   10,
          fontWeight: 600,
          color:      'var(--tb-red)',
          background: 'rgba(248,81,73,0.12)',
          border:     '1px solid rgba(248,81,73,0.25)',
          borderRadius: 3,
          padding:    '1px 5px',
          flexShrink: 0,
        }}>
          {errorCount} error{errorCount > 1 ? 's' : ''}
        </span>
      )}
    </div>
  );
}
