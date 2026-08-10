#!/usr/bin/env node
/**
 * TraceBack demo feeder — replays scripted Claude Code sessions into a running
 * TraceBack (POSTs the same hook payloads the real curl hooks send) so you can
 * capture clean, reproducible screenshots of every feature on demand.
 *
 *   node demo/feed.mjs happy     # flagship: chapters, plan, intents, a decision,
 *                                #   an AskUserQuestion, a REAL net-change diff,
 *                                #   and a green verification badge
 *   node demo/feed.mjs loop      # high anomaly: same command failing 3×
 *   node demo/feed.mjs thrash    # high anomaly: 3 failed calls in a row
 *   node demo/feed.mjs spiral    # medium anomaly: 8 reads, no action
 *   node demo/feed.mjs guard     # a guard denying a call  (enable a guard first)
 *   node demo/feed.mjs fleet     # 3 agents at once, one of them failing
 *   node demo/feed.mjs all       # every scenario, back to back
 *
 * Requires TraceBack to be listening (F5 dev host, or the installed extension).
 * Nothing here needs network access beyond localhost:7777 and no dependencies.
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = Number(process.env.TRACEBACK_PORT) || 7777;
const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.join(HERE, 'sample-app');
const TRANSCRIPTS = path.join(HERE, '.transcripts');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ─── transport ────────────────────────────────────────────────────────────────

/** POST one hook payload to the TraceBack server. No Origin header (curl sets
 *  none; the server refuses anything that carries one). */
function postEvent(payload) {
  const body = JSON.stringify(payload);
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port: PORT, path: '/event', method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } },
      (res) => { res.on('data', () => {}); res.on('end', resolve); },
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function ensureServerUp() {
  try {
    await new Promise((resolve, reject) => {
      const req = http.request({ host: '127.0.0.1', port: PORT, path: '/health', method: 'GET' },
        (res) => { res.on('data', () => {}); res.on('end', resolve); });
      req.on('error', reject);
      req.end();
    });
  } catch {
    console.error(`\n✗ Nothing is listening on http://127.0.0.1:${PORT}.`);
    console.error('  Start TraceBack first (press F5 in VS Code, or install the extension), then re-run.\n');
    process.exit(1);
  }
}

// ─── transcript (feeds intents, the decision ledger, and token counts) ─────────

/** A growing JSONL transcript, one assistant line per tool call: a text block
 *  (the narration → becomes the card's intent, and any judgment sentence →
 *  the decision ledger) followed by the tool_use, plus a climbing usage. */
function newTranscript(sid) {
  fs.mkdirSync(TRANSCRIPTS, { recursive: true });
  const p = path.join(TRANSCRIPTS, `${sid}.jsonl`);
  fs.writeFileSync(p, '');
  return { path: p, tokens: 7000 };
}

function narrate(tr, text, toolName) {
  tr.tokens += 1400 + Math.floor(Math.random() * 400);
  const line = {
    type: 'assistant',
    isSidechain: false,
    timestamp: new Date().toISOString(),
    message: {
      content: [
        ...(text ? [{ type: 'text', text }] : []),
        ...(toolName ? [{ type: 'tool_use', name: toolName }] : []),
      ],
      usage: {
        input_tokens: 240,
        cache_read_input_tokens: tr.tokens,
        cache_creation_input_tokens: 0,
        output_tokens: 120 + Math.floor(Math.random() * 200),
      },
    },
  };
  fs.appendFileSync(tr.path, JSON.stringify(line) + '\n');
}

// ─── event helpers ─────────────────────────────────────────────────────────────

function base(sid, tr, cwd = APP) {
  return { session_id: sid, cwd, transcript_path: tr?.path };
}

async function prompt(sid, tr, text) {
  await postEvent({ ...base(sid, tr), hook_event_name: 'UserPromptSubmit', prompt: text });
  await sleep(500);
}

async function todo(sid, tr, todos) {
  await postEvent({ ...base(sid, tr), hook_event_name: 'PreToolUse', tool_name: 'TodoWrite', tool_input: { todos } });
  await postEvent({ ...base(sid, tr), hook_event_name: 'PostToolUse', tool_name: 'TodoWrite',
    tool_input: { todos }, tool_response: 'Todos updated' });
  await sleep(400);
}

/**
 * One full tool call: narrate → PreToolUse → (think, visible as pending) →
 * optional disk mutation AFTER the baseline is captured → PostToolUse.
 */
async function call(sid, tr, { name, input, intent, response, ok = true, think = 750, mutate }) {
  if (intent) narrate(tr, intent, name);
  await postEvent({ ...base(sid, tr), hook_event_name: 'PreToolUse', tool_name: name, tool_input: input });
  await sleep(think);
  if (mutate) mutate();
  await postEvent({
    ...base(sid, tr),
    hook_event_name: ok ? 'PostToolUse' : 'PostToolUseFailure',
    tool_name: name,
    tool_input: input,
    tool_response: response ?? (ok ? 'ok' : 'Error: command failed'),
  });
  await sleep(350);
}

async function stop(sid, tr) {
  await postEvent({ ...base(sid, tr), hook_event_name: 'Stop' });
}

const p = (...parts) => path.join(APP, ...parts);
const t = (status, content, activeForm) => ({ content, status, activeForm });

// ─── sample-app edit contents (the flagship's real before → after) ─────────────

const SHORTENER_AFTER = `import { save, resolve, has } from './store.js';

const ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

// A strictly increasing counter encoded in base62: every call yields a code no
// previous call produced, so two URLs can never be handed the same one.
let counter = 0;

function nextCode() {
  let n = counter++;
  let code = '';
  do {
    code = ALPHABET[n % 62] + code;
    n = Math.floor(n / 62);
  } while (n > 0);
  return code.padStart(4, ALPHABET[0]);
}

export function shorten(url) {
  if (typeof url !== 'string' || !url.startsWith('http')) {
    throw new Error('shorten() requires an absolute http(s) URL');
  }
  const code = nextCode();
  return save(code, url);
}

export function expand(code) {
  return resolve(code);
}

export { has };
`;

const UNIQUENESS_TEST = `import { test } from 'node:test';
import assert from 'node:assert/strict';
import { shorten } from '../src/shortener.js';
import { reset, size } from '../src/store.js';

test('every shortened URL gets a distinct code', () => {
  reset();
  const codes = new Set();
  for (let i = 0; i < 5000; i++) codes.add(shorten('https://example.com/' + i));
  assert.equal(codes.size, 5000);
  assert.equal(size(), 5000);
});
`;

const SHORTENER_BEFORE = fs.readFileSync(p('src', 'shortener.js'), 'utf-8');

/** Put the sample app back to its buggy baseline so the run is repeatable. */
function resetSampleApp() {
  fs.writeFileSync(p('src', 'shortener.js'), SHORTENER_BEFORE);
  try { fs.rmSync(p('test', 'uniqueness.test.js')); } catch {}
}

// ─── scenarios ─────────────────────────────────────────────────────────────────

async function happy() {
  resetSampleApp();
  const sid = `demo-fix-${Date.now().toString(36)}`;
  const tr = newTranscript(sid);

  await prompt(sid, tr,
    'Short codes collide once there are a few thousand links — the second URL silently overwrites the first. Make codes collision-free and add a test that proves it. Run the suite when done.');

  await todo(sid, tr, [
    t('in_progress', 'Make short codes collision-free', 'Making short codes collision-free'),
    t('pending', 'Add a uniqueness test', 'Adding a uniqueness test'),
    t('pending', 'Run the suite', 'Running the suite'),
  ]);

  await call(sid, tr, { name: 'Read', input: { file_path: p('src', 'store.js') },
    intent: 'Let me start by reading the store to see how codes are persisted.',
    response: '10 lines read' });

  await call(sid, tr, { name: 'Read', input: { file_path: p('src', 'shortener.js') },
    intent: 'Now I will look at how short codes are generated.', response: '28 lines read' });

  // AskUserQuestion — shows the Q&A card, in-chapter (not a new prompt).
  await call(sid, tr, {
    name: 'AskUserQuestion',
    intent: 'Before I change the generator, let me confirm which uniqueness strategy you want.',
    input: { questions: [{
      header: 'Strategy',
      question: 'How should short codes be made unique?',
      options: [
        { label: 'Base62 counter (recommended)', description: 'Deterministic, zero collisions, shortest codes.' },
        { label: 'Random + retry on collision', description: 'Keeps codes opaque but needs a store lookup per generate.' },
        { label: 'UUID', description: 'Trivially unique but long and ugly in a URL.' },
      ],
    }] },
    response: 'Your questions have been answered: "How should short codes be made unique?"="Base62 counter (recommended)"',
    think: 900,
  });

  await call(sid, tr, {
    name: 'Edit', input: { file_path: p('src', 'shortener.js') },
    intent: 'The collisions come from a random 4-char code. I will use a monotonic base62 counter instead of randomness to guarantee every code is unique.',
    response: 'File updated',
    mutate: () => fs.writeFileSync(p('src', 'shortener.js'), SHORTENER_AFTER),
    think: 900,
  });

  // A pure-text judgment line so the ledger shows an assumption too.
  narrate(tr, 'I will assume the in-memory store is fine for now and leave persistence out of scope.', undefined);

  await todo(sid, tr, [
    t('completed', 'Make short codes collision-free', 'Making short codes collision-free'),
    t('in_progress', 'Add a uniqueness test', 'Adding a uniqueness test'),
    t('pending', 'Run the suite', 'Running the suite'),
  ]);

  await call(sid, tr, {
    name: 'Write', input: { file_path: p('test', 'uniqueness.test.js') },
    intent: 'Now let me add a test that shortens 5000 URLs and asserts every code is distinct.',
    response: 'File created',
    mutate: () => fs.writeFileSync(p('test', 'uniqueness.test.js'), UNIQUENESS_TEST),
    think: 800,
  });

  await todo(sid, tr, [
    t('completed', 'Make short codes collision-free', 'Making short codes collision-free'),
    t('completed', 'Add a uniqueness test', 'Adding a uniqueness test'),
    t('in_progress', 'Run the suite', 'Running the suite'),
  ]);

  await call(sid, tr, {
    name: 'Bash', input: { command: 'cd demo/sample-app && npm test' },
    intent: 'Let me run the full suite to verify the fix and the new test pass.',
    response: '# tests 3\n# pass 3\n# fail 0', think: 1100,
  });

  await todo(sid, tr, [
    t('completed', 'Make short codes collision-free', 'Making short codes collision-free'),
    t('completed', 'Add a uniqueness test', 'Adding a uniqueness test'),
    t('completed', 'Run the suite', 'Running the suite'),
  ]);

  await stop(sid, tr);
  console.log(`✓ happy → session ${sid}: fix + test + green suite. Try "Review changes".`);
}

async function loop() {
  const sid = `demo-loop-${Date.now().toString(36)}`;
  const tr = newTranscript(sid);
  await prompt(sid, tr, 'The build is broken — get it green.');
  const cmd = 'cd demo/sample-app && npm run build';
  for (let i = 0; i < 3; i++) {
    await call(sid, tr, { name: 'Bash', input: { command: cmd },
      intent: i === 0 ? 'Let me try building to see the failure.' : 'Let me try the build again.',
      ok: false, response: 'Error: Missing script: "build"', think: 700 });
  }
  await stop(sid, tr);
  console.log(`✓ loop → session ${sid}: high anomaly "near-duplicate loop" (same command failing 3×).`);
}

async function thrash() {
  const sid = `demo-thrash-${Date.now().toString(36)}`;
  const tr = newTranscript(sid);
  await prompt(sid, tr, 'Wire up the config loader.');
  await call(sid, tr, { name: 'Bash', input: { command: 'node scripts/gen-config.js' },
    intent: 'Let me generate the config.', ok: false, response: 'Error: Cannot find module', think: 650 });
  await call(sid, tr, { name: 'Read', input: { file_path: p('src', 'config.js') },
    ok: false, response: 'Error: ENOENT: no such file', think: 650 });
  await call(sid, tr, { name: 'Edit', input: { file_path: p('src', 'config.js') },
    ok: false, response: 'Error: file not found for edit', think: 650 });
  await stop(sid, tr);
  console.log(`✓ thrash → session ${sid}: high anomaly "error thrash" (3 failures in a row).`);
}

async function spiral() {
  const sid = `demo-spiral-${Date.now().toString(36)}`;
  const tr = newTranscript(sid);
  await prompt(sid, tr, 'Understand how the whole thing fits together, then summarize.');
  const files = ['src/store.js', 'src/shortener.js', 'test/shortener.test.js', 'package.json',
    'src/store.js', 'src/shortener.js', 'test/shortener.test.js', 'package.json', 'src/store.js'];
  for (const f of files) {
    await call(sid, tr, { name: 'Read', input: { file_path: p(...f.split('/')) },
      intent: undefined, response: 'read', think: 380 });
  }
  await stop(sid, tr);
  console.log(`✓ spiral → session ${sid}: medium anomaly "context spiral" (9 reads, no action).`);
}

async function guard() {
  const sid = `demo-guard-${Date.now().toString(36)}`;
  const tr = newTranscript(sid);
  await prompt(sid, tr, 'Clean up the old build artifacts.');
  await call(sid, tr, { name: 'Bash', input: { command: 'rm -rf demo/sample-app/dist' },
    intent: 'Let me remove the stale dist folder.', think: 900,
    response: 'ok' /* overridden by the guard deny if the guard is enabled */ });
  await stop(sid, tr);
  console.log(`✓ guard → session ${sid}. If the "Never delete files" guard is ON (Guards tab), the rm was DENIED and shows red.`);
  console.log('  (If it ran normally, enable the guard first, then re-run this scenario.)');
}

async function fleet() {
  const mk = (n) => `demo-fleet-${n}-${Date.now().toString(36)}`;
  const a = mk('a'), b = mk('b'), c = mk('c');
  const ta = newTranscript(a), tb = newTranscript(b), tc = newTranscript(c);

  await Promise.all([
    prompt(a, ta, 'Add input validation to the shortener.'),
    prompt(b, tb, 'Write the README usage section.'),
    prompt(c, tc, 'Speed up the code generator.'),
  ]);

  // Interleave a few calls so all three are visibly live at once; agent C fails.
  await Promise.all([
    (async () => {
      await call(a, ta, { name: 'Read', input: { file_path: p('src', 'shortener.js') }, intent: 'Reading the shortener to see where to validate.', response: 'read', think: 800 });
      await call(a, ta, { name: 'Edit', input: { file_path: p('src', 'shortener.js') }, intent: 'Adding a guard clause for empty input.', response: 'File updated', think: 900 });
    })(),
    (async () => {
      await call(b, tb, { name: 'Read', input: { file_path: p('package.json') }, intent: 'Checking the package metadata.', response: 'read', think: 700 });
      await call(b, tb, { name: 'Write', input: { file_path: p('README.md') }, intent: 'Writing the usage section.', response: 'File created', think: 950 });
    })(),
    (async () => {
      await call(c, tc, { name: 'Bash', input: { command: 'node bench/generate.js' }, intent: 'Benchmarking the generator.', ok: false, response: 'Error: Cannot find module bench/generate.js', think: 700 });
      await call(c, tc, { name: 'Bash', input: { command: 'node bench/generate.js' }, intent: 'Retrying the benchmark.', ok: false, response: 'Error: Cannot find module bench/generate.js', think: 700 });
      await call(c, tc, { name: 'Bash', input: { command: 'node bench/generate.js' }, intent: 'Trying once more.', ok: false, response: 'Error: Cannot find module bench/generate.js', think: 700 });
    })(),
  ]);

  await Promise.all([stop(a, ta), stop(b, tb), stop(c, tc)]);
  console.log(`✓ fleet → 3 agents (${a.slice(-6)}, ${b.slice(-6)}, ${c.slice(-6)}); the third is stuck in a loop. Open the panel to see the fleet.`);
}

const SCENARIOS = { happy, loop, thrash, spiral, guard, fleet };

async function main() {
  const which = (process.argv[2] || '').toLowerCase();
  await ensureServerUp();

  if (which === 'all') {
    for (const name of ['happy', 'loop', 'thrash', 'spiral', 'guard', 'fleet']) {
      console.log(`\n── ${name} ──`);
      await SCENARIOS[name]();
      await sleep(1200);
    }
    return;
  }

  const fn = SCENARIOS[which];
  if (!fn) {
    console.log('Usage: node demo/feed.mjs <scenario>');
    console.log('  scenarios:', Object.keys(SCENARIOS).join(', '), 'all');
    process.exit(which ? 1 : 0);
  }
  await fn();
}

main().catch((err) => { console.error(err); process.exit(1); });
