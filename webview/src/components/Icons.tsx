/**
 * Minimal inline SVG icon set (lucide-style: 24px viewBox, stroke-based,
 * currentColor). Replaces emoji so the UI reads as a designed tool, renders
 * identically across platforms, and inherits text color for free.
 */

interface IconProps {
  size?: number;
  strokeWidth?: number;
  style?: React.CSSProperties;
}

function base(size: number, style?: React.CSSProperties) {
  return {
    width: size, height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    style: { flexShrink: 0, display: 'block', ...style },
  };
}

export function PauseIcon({ size = 12, strokeWidth = 2, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <line x1="9" y1="5" x2="9" y2="19" />
      <line x1="15" y1="5" x2="15" y2="19" />
    </svg>
  );
}

export function PlayIcon({ size = 12, strokeWidth = 2, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <polygon points="7 4 20 12 7 20 7 4" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function AlertIcon({ size = 12, strokeWidth = 2, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

export function SparkleIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="M12 3 13.9 9.1 20 11 13.9 12.9 12 19 10.1 12.9 4 11 10.1 9.1 12 3Z" />
      <path d="M19 17.5 19.6 19.4 21.5 20 19.6 20.6 19 22.5 18.4 20.6 16.5 20 18.4 19.4 19 17.5Z" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function FolderIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.7-.9L9.2 3.9A2 2 0 0 0 7.5 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
    </svg>
  );
}

export function FileIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
    </svg>
  );
}

export function PencilIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
      <path d="m15 5 4 4" />
    </svg>
  );
}

export function CopyIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <rect width="14" height="14" x="8" y="8" rx="2" />
      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
    </svg>
  );
}

export function CheckIcon({ size = 12, strokeWidth = 2.2, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

export function ChevronIcon({ size = 12, strokeWidth = 2, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

export function DotsIcon({ size = 14, strokeWidth = 2, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <circle cx="5"  cy="12" r="0.8" fill="currentColor" />
      <circle cx="12" cy="12" r="0.8" fill="currentColor" />
      <circle cx="19" cy="12" r="0.8" fill="currentColor" />
    </svg>
  );
}

export function PanelIcon({ size = 13, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <rect width="18" height="18" x="3" y="3" rx="2" />
      <path d="M9 3v18" />
    </svg>
  );
}

export function ClockIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </svg>
  );
}

export function TargetIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="5" />
      <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function ListChecksIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="m3 7 2 2 4-4" />
      <path d="m3 17 2 2 4-4" />
      <path d="M13 6h8" />
      <path d="M13 18h8" />
    </svg>
  );
}

export function SendIcon({ size = 12, strokeWidth = 1.8, style }: IconProps) {
  return (
    <svg {...base(size, style)} strokeWidth={strokeWidth}>
      <path d="M22 2 11 13" />
      <path d="M22 2 15 22l-4-9-9-4Z" />
    </svg>
  );
}
