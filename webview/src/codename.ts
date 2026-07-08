/**
 * Matrix-style session identities: every session id hashes deterministically
 * to an Agent name + accent color, so the same session always resurfaces as
 * the same agent (including after reloads — the id is stable in the archive).
 *
 * Name and color hash INDEPENDENTLY: there are only 8 names but 10 colors,
 * so two sessions that collide on "Agent Smith" almost always differ in
 * color — and always differ in tag (a 4-char slice of the session id itself,
 * rendered next to the name so identical names are never ambiguous).
 *
 * Identity colors are for IDENTITY ONLY (name tints, rail accents) — never
 * for status. Green/red stay reserved for success/error semantics, so the
 * palette deliberately avoids both.
 */

export interface AgentIdentity {
  name:  string;
  color: string;
  /** Short disambiguator from the session id, e.g. "4F2A". */
  tag:   string;
}

const NAMES = [
  'Agent Smith', 'Agent Brown', 'Agent Jones', 'Agent Johnson',
  'Agent Thompson', 'Agent White', 'Agent Gray', 'Agent Jackson',
];

const COLORS = [
  '#39c5cf', // cyan
  '#f0883e', // orange
  '#a371f7', // violet
  '#58a6ff', // blue
  '#f778ba', // pink
  '#e3b341', // gold
  '#2dd4bf', // teal
  '#79c0ff', // sky
  '#d2a8ff', // lavender
  '#e6edf3', // silver
];

function hash(s: string, seed: number): number {
  let h = seed;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function agentIdentity(sessionId: string): AgentIdentity {
  return {
    name:  NAMES[hash(sessionId, 0) % NAMES.length],
    color: COLORS[hash(sessionId, 7) % COLORS.length],
    tag:   agentTag(sessionId),
  };
}

/** Stable 4-char tag from the id's own characters (uuid tail), uppercased. */
export function agentTag(sessionId: string): string {
  const clean = sessionId.replace(/[^a-z0-9]/gi, '');
  return (clean.slice(-4) || sessionId.slice(0, 4)).toUpperCase();
}

export function codename(sessionId: string): string {
  return agentIdentity(sessionId).name;
}
