import { useEffect, useRef, useState } from 'react';

/**
 * Empty state: matrix-rain canvas in the TraceBack palette, with the status
 * copy floating on top. Canvas auto-sizes to the container, so it fills the
 * narrow sidebar and the wide panel equally well.
 */

const GLYPHS =
  'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ' +
  '0123456789$#*+=<>[]{}';

const FONT_SIZE  = 13;   // px, cell size
const FRAME_MS   = 72;   // ~14fps — the classic matrix cadence, easy on CPU
const HEAD_COLOR = 'rgba(240, 255, 245, 0.95)';  // bright head char
const TRAIL_HUE  = '63, 185, 80';                 // --tb-green

/** The opening scene, on loop. */
const WAKE_LINES = [
  'Wake up, Neo...',
  'The Matrix has you...',
  'Follow the white rabbit.',
  'Knock, knock, Neo.',
];
const WAKE_TYPE_MS  = 85;    // per character
const WAKE_HOLD_MS  = 2200;  // fully-typed line stays on screen
const WAKE_GAP_MS   = 500;   // blank pause between lines

/** Types each line char-by-char, holds, wipes, moves to the next. Loops. */
function WakeUpLine() {
  const [text, setText] = useState('');
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    function schedule(fn: () => void, ms: number) {
      timer.current = setTimeout(() => { if (!cancelled) fn(); }, ms);
    }

    function playLine(index: number) {
      const line = WAKE_LINES[index % WAKE_LINES.length];
      let chars = 0;
      const typeNext = () => {
        chars++;
        setText(line.slice(0, chars));
        if (chars < line.length) schedule(typeNext, WAKE_TYPE_MS);
        else schedule(() => {
          setText('');
          schedule(() => playLine(index + 1), WAKE_GAP_MS);
        }, WAKE_HOLD_MS);
      };
      typeNext();
    }

    playLine(0);
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  return (
    <p style={{
      fontSize: 14,
      fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
      color: '#7ee787',
      margin: 0,
      minHeight: 22,
      textShadow: '0 0 14px rgba(63,185,80,0.55)',
      whiteSpace: 'nowrap',
    }}>
      {text}
      <span className="cursor-blink">▋</span>
    </p>
  );
}

export default function EmptyState() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef   = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap   = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let drops: number[] = [];
    let cols = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const { width, height } = wrap!.getBoundingClientRect();
      canvas!.width  = Math.max(1, Math.floor(width * dpr));
      canvas!.height = Math.max(1, Math.floor(height * dpr));
      canvas!.style.width  = `${width}px`;
      canvas!.style.height = `${height}px`;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx!.font = `${FONT_SIZE}px ui-monospace, monospace`;

      const nextCols = Math.ceil(width / FONT_SIZE);
      if (nextCols !== cols) {
        cols = nextCols;
        // Stagger the starting rows so the rain doesn't fall as one curtain
        drops = Array.from({ length: cols }, () => Math.floor(Math.random() * -40));
      }
      // Opaque base so the first frames aren't transparent
      ctx!.fillStyle = '#07090d';
      ctx!.fillRect(0, 0, width, height);
    }

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(wrap);

    const timer = setInterval(() => {
      const { width, height } = wrap.getBoundingClientRect();

      // Fade the previous frame toward the background — this paints the trails
      ctx.fillStyle = 'rgba(7, 9, 13, 0.16)';
      ctx.fillRect(0, 0, width, height);

      for (let i = 0; i < cols; i++) {
        const y = drops[i] * FONT_SIZE;
        if (y > 0) {
          const glyph = GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
          const x     = i * FONT_SIZE;
          // Head char bright, body drawn dim by the fade above
          ctx.fillStyle = Math.random() < 0.16 ? HEAD_COLOR : `rgba(${TRAIL_HUE}, 0.8)`;
          ctx.fillText(glyph, x, y);
        }
        // Reset a column shortly after it exits, at a random delay
        if (y > height && Math.random() > 0.965) {
          drops[i] = Math.floor(Math.random() * -20);
        }
        drops[i]++;
      }
    }, FRAME_MS);

    return () => {
      clearInterval(timer);
      observer.disconnect();
    };
  }, []);

  return (
    <div
      ref={wrapRef}
      style={{
        position: 'relative',
        height: '100%',
        overflow: 'hidden',
        fontFamily: 'var(--tb-ui-font)',
        userSelect: 'none',
      }}
    >
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, opacity: 0.8 }} />

      {/* Readability vignette behind the copy */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse 70% 45% at 50% 52%, rgba(7,9,13,0.92) 0%, rgba(7,9,13,0.55) 55%, transparent 100%)',
        pointerEvents: 'none',
      }} />

      {/* ── Floating status copy ── */}
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        gap: 14,
        padding: '0 16px',
        textAlign: 'center',
      }}>
        <WakeUpLine />

        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 7,
          background: 'rgba(13,17,23,0.85)',
          border: '1px solid var(--tb-border-2)',
          borderLeft: '3px solid var(--tb-green)',
          borderRadius: 4,
          padding: '7px 14px',
          backdropFilter: 'blur(2px)',
        }}>
          <span style={{
            color: 'var(--tb-green)', fontSize: 11,
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
          }}>$</span>
          <span style={{
            fontSize: 12, color: 'var(--tb-text)',
            fontFamily: 'var(--tb-mono-font, ui-monospace, monospace)',
          }}>claude</span>
          <span className="cursor-blink" style={{
            fontSize: 12, color: 'var(--tb-green)', lineHeight: 1,
          }}>▋</span>
        </div>

        <p style={{
          fontSize: 10.5, color: 'var(--tb-text-muted)',
          margin: 0, lineHeight: 1.55, maxWidth: 240,
        }}>
          Start a Claude Code session in any terminal — every tool call streams in here live.
        </p>
      </div>

      {/* Footer */}
      <p style={{
        position: 'absolute', bottom: 12, left: 0, right: 0,
        textAlign: 'center',
        fontSize: 9, letterSpacing: '0.12em', textTransform: 'uppercase',
        color: 'var(--tb-text-dim)',
        margin: 0,
      }}>
        Listening on :7777
      </p>
    </div>
  );
}
