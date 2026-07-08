import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  listSessions,
  loadSession,
  pruneOldest,
  saveSession,
  serializeSession,
} from '../src/sessionArchive';
import type { TraceSession } from '../src/traceStore';

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tb-archive-'));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

function makeSession(id: string, startedAt: number, nodeCount = 2): TraceSession {
  return {
    id,
    label: `sess-${id}`,
    startedAt,
    stopped: true,
    events: [],
    nodes: Array.from({ length: nodeCount }, (_, i) => ({
      id: `n${i}`,
      toolName: i === 0 ? 'Read' : 'Edit',
      status: i === 1 ? 'error' : 'success',
      label: `node ${i}`,
      count: 1,
      eventIds: [`e${i}`],
      timestamp: startedAt + i,
    })),
    anomalyHistory: [
      { type: 'stall', reason: 'test stall', flaggedEventIds: ['e0'], detectedAt: startedAt },
    ],
    contextTokens: 1234,
    cwd: '/tmp/project',
  };
}

describe('serializeSession', () => {
  it('drops raw events and counts only real nodes', () => {
    const s = makeSession('a', 1000);
    s.nodes.push({
      id: 'think', toolName: '__thinking__', status: 'thinking',
      label: 'Thinking…', count: 1, eventIds: [], timestamp: 1002,
    });
    const out = serializeSession(s);
    expect(out.nodeCount).toBe(2);
    expect(out.errorCount).toBe(1);
    expect(out.anomalyCount).toBe(1);
    expect(out.tokens).toBe(1234);
    expect('events' in out).toBe(false);
  });
});

describe('save → list → load roundtrip', () => {
  it('persists and reads back a session', () => {
    saveSession(dir, makeSession('abc-123', 5000));

    const metas = listSessions(dir);
    expect(metas).toHaveLength(1);
    expect(metas[0].id).toBe('abc-123');
    expect(metas[0].label).toBe('sess-abc-123');

    const full = loadSession(dir, 'abc-123');
    expect(full).not.toBeNull();
    expect(full!.nodes).toHaveLength(2);
    expect(full!.anomalyHistory).toHaveLength(1);
  });

  it('never archives empty sessions', () => {
    saveSession(dir, makeSession('empty', 1000, 0));
    expect(listSessions(dir)).toHaveLength(0);
  });

  it('lists newest first', () => {
    saveSession(dir, makeSession('old', 1000));
    saveSession(dir, makeSession('new', 2000));
    expect(listSessions(dir).map((m) => m.id)).toEqual(['new', 'old']);
  });

  it('returns null / [] for missing data', () => {
    expect(loadSession(dir, 'nope')).toBeNull();
    expect(listSessions(path.join(dir, 'missing'))).toEqual([]);
  });

  it('skips corrupt files without throwing', () => {
    saveSession(dir, makeSession('good', 1000));
    fs.writeFileSync(path.join(dir, 'bad.json'), '{not json');
    const metas = listSessions(dir);
    expect(metas).toHaveLength(1);
    expect(metas[0].id).toBe('good');
  });
});

describe('pruneOldest', () => {
  it('keeps only the newest N sessions', () => {
    for (let i = 0; i < 6; i++) saveSession(dir, makeSession(`s${i}`, 1000 + i));
    pruneOldest(dir, 3);
    const ids = listSessions(dir).map((m) => m.id);
    expect(ids).toEqual(['s5', 's4', 's3']);
  });
});
