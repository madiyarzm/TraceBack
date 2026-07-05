export type ZoomLevel = 'chapters' | 'map' | 'steps' | 'detail';

interface Props {
  zoom:     ZoomLevel;
  onChange: (z: ZoomLevel) => void;
  compact?: boolean;
  /** Panel-only "Prompts" level (the prompt-chapter view). */
  withChapters?: boolean;
}

const LEVELS: { id: ZoomLevel; label: string; title: string }[] = [
  { id: 'chapters', label: 'Prompts', title: 'Session grouped by your prompts — tasks, actions, intents' },
  { id: 'map',    label: 'Map',    title: 'Whole session at a glance — plan, phases, errors' },
  { id: 'steps',  label: 'Steps',  title: 'Compact action timeline' },
  { id: 'detail', label: 'Detail', title: 'Everything expanded — diffs, commands, raw output' },
];

/**
 * The zoom axis is TraceBack's core difference from the Claude Code
 * transcript (which has exactly one scale): pull back to see the session's
 * shape, push in to see more nuance than the transcript shows.
 */
export default function ZoomControl({ zoom, onChange, compact = false, withChapters = false }: Props) {
  const levels = withChapters ? LEVELS : LEVELS.filter((l) => l.id !== 'chapters');
  return (
    <div style={{
      display: 'inline-flex',
      border: '1px solid var(--tb-border)',
      borderRadius: 4,
      overflow: 'hidden',
      flexShrink: 0,
    }}>
      {levels.map((level, i) => {
        const active = zoom === level.id;
        return (
          <button
            key={level.id}
            title={level.title}
            onClick={() => onChange(level.id)}
            style={{
              fontSize: compact ? 9 : 9.5,
              fontFamily: 'var(--tb-ui-font)',
              fontWeight: active ? 700 : 400,
              letterSpacing: '0.03em',
              padding: compact ? '2px 6px' : '3px 9px',
              cursor: 'pointer',
              background: active ? 'rgba(88,166,255,0.12)' : 'transparent',
              color: active ? 'var(--tb-blue)' : 'var(--tb-text-muted)',
              border: 'none',
              borderLeft: i > 0 ? '1px solid var(--tb-border)' : 'none',
              transition: 'background 0.1s, color 0.1s',
            }}
          >
            {level.label}
          </button>
        );
      })}
    </div>
  );
}
