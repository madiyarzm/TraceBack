/**
 * Matrix-style session identities: every session id hashes deterministically
 * to an Agent name + accent color, so the same session always resurfaces as
 * the same agent (including after reloads — the id is stable in the archive).
 *
 * Identity colors are for IDENTITY ONLY (name tints, rail accents) — never
 * for status. Green/red stay reserved for success/error semantics, so the
 * palette deliberately avoids both.
 */

export interface AgentIdentity {
  name:  string;
  color: string;
}

const AGENTS: AgentIdentity[] = [
  { name: 'Agent Smith',    color: '#39c5cf' }, // cyan
  { name: 'Agent Brown',    color: '#f0883e' }, // orange
  { name: 'Agent Jones',    color: '#a371f7' }, // purple
  { name: 'Agent Johnson',  color: '#58a6ff' }, // blue
  { name: 'Agent Thompson', color: '#f778ba' }, // pink
  { name: 'Agent White',    color: '#e6edf3' }, // silver
  { name: 'Agent Gray',     color: '#8b949e' }, // gray
  { name: 'Agent Jackson',  color: '#d2a8ff' }, // lavender
];

export function agentIdentity(sessionId: string): AgentIdentity {
  let h = 0;
  for (let i = 0; i < sessionId.length; i++) {
    h = (h * 31 + sessionId.charCodeAt(i)) | 0;
  }
  return AGENTS[Math.abs(h) % AGENTS.length];
}

export function codename(sessionId: string): string {
  return agentIdentity(sessionId).name;
}
