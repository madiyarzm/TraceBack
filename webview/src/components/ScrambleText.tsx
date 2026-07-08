import { memo, useEffect, useRef, useState } from 'react';

/**
 * Matrix digital-resolve: text mounts as random katakana/symbols and decodes
 * left-to-right into the real string over ~400ms. Runs once per mount —
 * timeline cards are keyed by node id, so each action decodes exactly once
 * as it appears, and re-renders (status flips, etc.) don't replay it.
 */

const NOISE = 'アイウエオカキクケコサシスセソタチツ0123456789<>*+=';

const STEP_MS  = 28;  // per animation frame
const PER_STEP = 2;   // characters locked in per frame

function scrambled(target: string, locked: number): string {
  let out = '';
  for (let i = 0; i < target.length; i++) {
    if (i < locked || target[i] === ' ') out += target[i];
    else out += NOISE[Math.floor(Math.random() * NOISE.length)];
  }
  return out;
}

function ScrambleText({ text }: { text: string }) {
  const [display, setDisplay] = useState(() => scrambled(text, 0));
  const done = useRef(false);

  useEffect(() => {
    if (done.current) { setDisplay(text); return; }
    let locked = 0;
    const timer = setInterval(() => {
      locked += PER_STEP;
      if (locked >= text.length) {
        done.current = true;
        setDisplay(text);
        clearInterval(timer);
      } else {
        setDisplay(scrambled(text, locked));
      }
    }, STEP_MS);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  return <>{display}</>;
}

export default memo(ScrambleText);
