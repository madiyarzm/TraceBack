import { useState } from 'react';
import { chapterBadge, PromptChapter, timeAgo, ChapterBadge } from '../chapters';
import { AlertIcon, CheckIcon } from './Icons';

interface Props {
  chapters:       PromptChapter[];
  selectedIndex:  number;             // chapter.index of the focused chapter
  isLive:         boolean;
  /** chapter.index → anomaly count (for the badge). */
  anomalyCounts:  Map<number, number>;
  onSelect:       (index: number) => void;
}

const BADGE_STYLE: Record<ChapterBadge, { color: string; bg: string; border: string }> = {
  done:      { color: '#3fb950', bg: 'rgba(63,185,80,0.1)',  border: 'rgba(63,185,80,0.35)' },
  active:    { color: '#58a6ff', bg: 'rgba(88,166,255,0.1)', border: 'rgba(88,166,255,0.35)' },
  errors:    { color: '#d29922', bg: 'rgba(210,153,34,0.1)', border: 'rgba(210,153,34,0.35)' },
  anomalies: { color: '#f85149', bg: 'rgba(248,81,73,0.1)',  border: 'rgba(248,81,73,0.35)' },
  queued:    { color: 'var(--tb-text-dim)', bg: 'transparent', border: 'var(--tb-border-2)' },
};

/**
 * Left-rail chapter index: one row per user prompt, newest last, with a
 * status badge and action count. Clicking focuses the main panel on that
 * prompt's slice of the session.
 */
export default function PromptList({ chapters, selectedIndex, isLive, anomalyCounts, onSelect }: Props) {
  if (chapters.length === 0) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: '2px 10px 6px' }}>
      {chapters.map((c) => (
        <PromptRow
          key={c.index}
          chapter={c}
          selected={c.index === selectedIndex}
          badge={chapterBadge(c, {
            isLast: c.index === chapters.length,
            isLive,
            anomalyCount: anomalyCounts.get(c.index) ?? 0,
          })}
          onClick={() => onSelect(c.index)}
        />
      ))}
    </div>
  );
}

function PromptRow({ chapter, selected, badge, onClick }: {
  chapter:  PromptChapter;
  selected: boolean;
  badge:    { badge: ChapterBadge; label: string };
  onClick:  () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const s = BADGE_STYLE[badge.badge];
  const preview = chapter.text.replace(/\s+/g, ' ').slice(0, 64)
    + (chapter.text.length > 64 ? '…' : '');
  const badgeIcon =
    badge.badge === 'done'      ? <CheckIcon size={10} />
    : badge.badge === 'errors' || badge.badge === 'anomalies' ? <AlertIcon size={10} />
    : null;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        padding: '11px 13px',
        borderRadius: 9,
        border: `1px solid ${selected ? 'var(--tb-border-2)' : 'transparent'}`,
        background: selected ? 'var(--tb-surface-2)' : hovered ? 'rgba(22,27,34,0.55)' : 'transparent',
        cursor: 'pointer',
        transition: 'background 0.1s, border-color 0.1s',
        fontFamily: 'var(--tb-ui-font)',
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: 7,
        fontSize: 11, fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
        color: 'var(--tb-text-muted)',
      }}>
        <span style={{ fontWeight: 700, color: badge.badge === 'queued' ? 'var(--tb-text-dim)' : 'var(--tb-text-muted)' }}>
          P{chapter.index}
        </span>
        <span style={{ opacity: 0.6 }}>·</span>
        <span>{badge.badge === 'active' ? 'now' : timeAgo(chapter.timestamp)}</span>
      </div>

      <div style={{
        fontSize: 13.5, fontWeight: 600, lineHeight: 1.35, marginTop: 5,
        color: badge.badge === 'queued' ? 'var(--tb-text-muted)' : 'var(--tb-text)',
        overflow: 'hidden',
        display: '-webkit-box',
        WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
      }}>
        {preview}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginTop: 8 }}>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 4,
          fontSize: 10, fontWeight: 600, letterSpacing: '0.02em',
          color: s.color, background: s.bg,
          border: `1px solid ${s.border}`,
          borderRadius: 5, padding: '2px 8px',
        }}>
          {badgeIcon}
          {badge.label}
        </span>
        {chapter.actionCount > 0 && (
          <span style={{ fontSize: 11, color: 'var(--tb-text-muted)' }}>
            {chapter.actionCount} action{chapter.actionCount === 1 ? '' : 's'}
          </span>
        )}
      </div>
    </div>
  );
}
