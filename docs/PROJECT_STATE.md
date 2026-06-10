# TraceBack — Project State & Handoff

> Engineering handoff document. Audience: an AI agent or developer picking this
> project up cold. Describes what it is, how it works end-to-end, the bugs found
> and fixed (and still open), and a strategic feature roadmap.
> Last updated: 2026-06-09.

---

## 1. What it is

**TraceBack** is a VS Code extension that gives **real-time, local-first
observability into AI coding agents** (currently Claude Code). It taps Claude
Code's hook system, streams every tool call into a local HTTP server, and renders
them as a live, interactive "action map" (node graph) in the VS Code sidebar.

- **Status:** working prototype (~1,600 LOC). Compiles clean (extension + webview).
- **Positioning:** *not* a cloud dashboard like Langfuse/Arize/LangSmith. The
  wedge is **in-editor, zero-setup, real-time, anomaly-focused** — catch an agent
  looping/erroring/stuck while it runs, and watch several agents at once.
- **Author goal:** showcase to startups/recruiters, publish on the VS Code
  Marketplace, anchor an "agent observability" portfolio narrative. The strongest
  product differentiator to pursue is **OpenTelemetry GenAI semantic conventions**
  (see §8).

---

## 2. High-level architecture

Two processes communicating over VS Code's `postMessage` bridge:

```
┌─────────────────────── Extension Host (Node.js) ───────────────────────┐
│                                                                         │
│  Claude Code CLI                                                        │
│    │ PreToolUse / PostToolUse / Stop hook fires                         │
│    │ curl POST → 127.0.0.1:7777/event   (stdin payload piped with -d @-)│
│    ▼                                                                    │
│  server.ts        HTTP server, parses hook payload → TraceEvent         │
│    ▼                                                                    │
│  traceStore.ts    in-memory session state; builds nodes; loop/timeout   │
│    │ onDidUpdate(session)                                               │
│    ▼                                                                    │
│  extension.ts     wiring; debounced LLM summary; timeout interval       │
│    ▼                                                                    │
│  webviewProvider.ts  serializes session + allSessions → postMessage     │
│                                                                         │
│  hookManager.ts   installs/removes curl hooks in ~/.claude/settings.json│
│  llmClient.ts     Groq / Ollama HTTP clients (summary + chat)           │
└─────────────────────────────────────────────────────────────────────────┘
                                │ postMessage
                                ▼
┌──────────────────────── Webview (React + Vite) ────────────────────────┐
│  App.tsx          message handling, dagre layout, ReactFlow canvas      │
│  nodes/           ToolNode, BatchNode, ThinkingNode, SwimLaneNode        │
│  components/       DetailDrawer, SummaryBar, StumbleAlert, Toolbar,      │
│                    SessionPicker, EmptyState                             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key design choice:** no sockets/shared memory between extension and webview —
only VS Code API primitives (`postMessage` / `onDidReceiveMessage`). The HTTP
server exists *only* to receive Claude Code hook payloads, not for webview comms.

---

## 3. File-by-file reference

### Extension host (`src/`)

| File | Lines | Responsibility |
|---|---|---|
| `extension.ts` | 131 | `activate()`: starts server, installs hooks, registers webview provider + 4 commands, wires `traceStore.onDidUpdate → postSessionUpdate`, debounced LLM auto-summary (2.5s), timeout check interval (5s). `deactivate()`: stops server, disposes store. **Note: does not call `removeHooks` on deactivate** (see open issues). |
| `server.ts` | 122 | `http` server on `127.0.0.1:<port>`. `POST /event` parses Claude hook JSON → `TraceEvent` → `traceStore.addEvent`. `GET /health` returns `{status:'ok'}`. Maps `hook_event_name` (`PreToolUse`/`PostToolUse`/`Stop`/`Notification`) → internal `EventKind`. Reads `session_id`, `tool_name`, `tool_input`, `tool_response`, `tool_response_is_error` from payload. |
| `traceStore.ts` | ~395 | Core state. `Map<sessionId, TraceSession>`. `addEvent` routes by `event.sessionId`. Two-pass node builder: `buildNodes` (flat list, pending→success/error, thinking nodes, collapse consecutive identical calls) then `applyBatchGrouping` (runs of ≥3 same-tool → one batch node). `detectLoop` (3 identical tool+label in a row), `checkTimeouts` (node pending >45s). Singleton `traceStore`. |
| `hookManager.ts` | 137 | Reads/writes `~/.claude/settings.json`. Injects a `curl` command tagged with `# traceback-managed` marker into `PreToolUse`/`PostToolUse`/`Stop` arrays. `removeHooks` filters managed entries out. Surgical — preserves unrelated config. |
| `webviewProvider.ts` | ~172 | `WebviewViewProvider`. Builds HTML (CSP + nonce, cache-busted asset URIs). `postSessionUpdate` serializes the active session + `allSessions` (filtered to non-empty). Handles webview messages: `ready`, `open_file`, `switch_session`, `clear_session`, `llm_query`. Also opens a full editor panel (`openFullPanel`). |
| `llmClient.ts` | 113 | `callLLM` dispatches to `callGroq` (api.groq.com, OpenAI-compatible, max_tokens 200, temp 0.35) or `callOllama` (localhost:11434 `/api/chat`, stream:false). Raw `https`/`http`, no SDK. |

### Webview (`webview/src/`)

| File | Responsibility |
|---|---|
| `App.tsx` (~500) | Receives `session_update` / `llm_response`. Decides swimlane vs single-session (`allSessions.length > 1`). Layout via **dagre**: `layoutSession(nodes, 'TB'|'LR')` → `applyDagreLayout`. Single-session = vertical (`TB`); swimlane = horizontal lanes (`LR`). `pruneOrphanEdges` strips edges whose endpoints are gone. Handles node click → DetailDrawer, PNG/JSON export, chat, clear. |
| `nodes/ToolNode.tsx` | Single tool-call card (icon, label, status color, count badge, looping halo). |
| `nodes/BatchNode.tsx` | Collapsed run of ≥3 same-tool calls; shows first 3 items + "+N more". |
| `nodes/ThinkingNode.tsx` | Dashed "Thinking…" placeholder between tool calls. |
| `nodes/SwimLaneNode.tsx` | Lane background + agent "personality" badge (Agent Smith/Brown/Jones). |
| `components/DetailDrawer.tsx` | Slide-over showing full tool input/output; hosts the chat box. |
| `components/SummaryBar.tsx` | One-line status ("Reading 25 files · Running 5 commands…") + AI summary. |
| `components/StumbleAlert.tsx` | Red loop/timeout banner + canvas red-tint overlay. |
| `components/Toolbar.tsx` | LIVE/DONE indicator, action count, PNG/JSON/CLR buttons. |
| `components/SessionPicker.tsx` | Dropdown to switch sessions (single-session mode only). |

---

## 4. Data model & lifecycle

```ts
TraceEvent  { id, sessionId, kind, toolName?, toolInput?, toolResponse?, isError?, timestamp }
TraceNode   { id, toolName, status, label, count, eventIds[], detail?, toolInput?,
              isLooping?, timestamp, isBatch?, batchItems? }
TraceSession{ id, label, startedAt, stopped, events[], nodes[], stumbleAlert?, aiSummary? }
```

**Per event flow:**
1. `server.handleHookPayload` → `TraceEvent` (sessionId from Claude `session_id`).
2. `traceStore.addEvent` → `getOrCreateSession(event.sessionId)` (sets it active).
3. `buildNodes(events)` rebuilds the flat node list from scratch every event:
   - `pre_tool_use` → pending node (or `count++` if identical to prev success).
   - `post_tool_use` → flips matching pending node to success/error; pushes a
     `__thinking__` node (unless error/stopped).
   - `stop` → marks session stopped, flips remaining pending → success.
4. `applyBatchGrouping` collapses runs of ≥3 same-tool nodes into a batch node.
5. Loop detection on `pre_tool_use`; timeout detection on a 5s interval.
6. `onDidUpdate` → `postSessionUpdate` → webview re-lays-out and renders.

**Label extraction** (`buildLabel`): prefers `tool_input.description`, falls back
to file path (last 2 segments) / command / query / url, truncated.

---

## 5. How to build / run / test

```bash
# Install
npm install
cd webview && npm install && cd ..

# Build
npm run compile        # tsc → out/
npm run build:webview  # vite → webview/dist/

# Dev
npm run watch          # tsc -w
npm run dev:webview    # vite HMR (note: webview is loaded from dist, not dev server)

# Package
npm run package        # vsce package → traceback-0.1.0.vsix
```

**Run the extension:** open repo in VS Code, press **F5** → Extension Development
Host. Click the TraceBack (pulse) icon. Start a Claude Code session in a terminal.

**Reload rules (important for iteration):**
- Webview-only change → rebuild webview, then reload the Dev Host window (`Cmd+R`).
- Extension-host change → rebuild, then **full restart** (`Shift+F5` → `F5`).
- Asset URIs are cache-busted with `?v=<timestamp>`, so reloads don't serve stale JS.

**Safe testing:** the agent edits files in whatever folder its session runs in.
Use a throwaway repo (e.g. `~/tb-sandbox`) for edit-flow tests, not a real codebase.

---

## 6. Bugs found & FIXED in the 2026-06-09 session

All four were architectural, not cosmetic. Fixes are **uncommitted** in the working
tree as of this writing.

1. **Sessions never separated (root cause of most weirdness).**
   `addEvent` called `getOrCreateActiveSession()` and ignored `event.sessionId`,
   so every agent's events merged into one session and swimlane mode could never
   trigger. → Added `getOrCreateSession(sessionId)`; `addEvent` now routes by it.
   *(traceStore.ts)*

2. **Orphaned edges.** When individual nodes collapsed into a `batch-*` node, the
   old edges remained, rendering as lines connecting nothing. → Added
   `pruneOrphanEdges(nodes, edges)` applied on both render paths. *(App.tsx)*

3. **Wrong layout orientation for a sidebar.** Single-session used a hand-rolled
   vertical stacker that overlapped cards; first fix reused the swimlane's
   horizontal (`LR`) dagre layout, which spread nodes off-screen in the narrow
   sidebar. → Generalized `applyDagreLayout`/`layoutSession` with a `rankdir`
   param; single-session now uses **`TB` (top-to-bottom)**, swimlanes keep `LR`.
   Edges switched from bezier to `smoothstep`. *(App.tsx)*

4. **Ghost sessions / stuck in swimlane + stale bundle.**
   - `postSessionUpdate` sent *all* sessions ever created (incl. empty ones and a
     fresh one made by every CLR), so `allSessions.length > 1` locked the view in
     swimlane mode with empty lanes. → Filter `allSessions` to sessions with real
     activity.
   - `clearActive` *added* a session instead of clearing. → Now wipes the whole
     `_sessions` map.
   - Webview bundle filename is stable (`index.js`, no content hash) so Electron
     served a cached old copy across reloads. → Cache-bust asset URIs with `?v=`.
   *(webviewProvider.ts, traceStore.ts)*

---

## 7. OPEN problems / tech debt / risks

**Correctness / robustness**
- **No `removeHooks` on `deactivate`.** Hooks persist in `~/.claude/settings.json`
  after the extension is disabled/uninstalled unless the user runs Stop Listening.
  Leaves a dangling `curl` that fails silently if the server is down (adds latency
  to every tool call). Should clean up on deactivate.
- **Hook failure adds latency to the agent.** Every tool call shells out to `curl`.
  If the server is slow/down, Claude Code waits on it. No timeout on the curl
  (`curl -s` with no `--max-time`). Add `--max-time 1`.
- **`buildNodes` rebuilds the entire node list on every event** — O(n) per event,
  O(n²) per session. Fine for short sessions, will lag on long ones. Consider
  incremental updates.
- **PostToolUse → pending node matching is by toolName + "last pending"**, which
  can mis-attribute when the same tool runs concurrently (parallel tool calls in
  one turn). Claude Code can emit parallel tool calls; matching should use a
  tool-call id if the payload provides one.
- **Single hook entry with empty `matcher`** catches all tools; fine, but the
  `Notification` hook type is declared in types yet never installed.

**Product / UX**
- **Value on the happy path is thin.** A linear list of successful reads ≈ a
  progress bar; the transcript already shows it. Value concentrates in
  failure/loop/timeout/multi-session/audit scenarios (see §8). Owner has flagged
  this ("50/50 on usefulness") — it is the central product question.
- **No persistence.** Everything is in-memory; sessions vanish on reload. No
  history, no post-hoc audit, no export of a *past* session.
- **No token/cost/latency metrics** — the data observability buyers actually want.
- **Swimlane "personalities"** are cute but only 3 hardcoded; 4th+ session wraps.

**Project hygiene (blocks Marketplace publish)**
- **No `LICENSE` file** though README + `package.json` claim MIT.
- **`publisher: "traceback"`** is a placeholder — needs a real owned publisher id.
- **No icon / galleryBanner**, `categories: ["Other"]` only.
- **`out/` is committed** (build artifacts in git).
- **No tests, no CI.** `traceStore` detection logic is pure and highly testable.
- **No bundling** — ships raw `tsc` output; consider esbuild for size/startup.

---

## 8. Strategic feature roadmap

Three tracks. Track A makes it shippable; Track C makes it *interesting* to the
agent-observability market.

### Track A — Ship it (≈1 day)
- Add `LICENSE` (MIT). Claim a Marketplace publisher id, fix `package.json`.
- Add 128px icon + `galleryBanner`; set real `categories`/`keywords`.
- esbuild bundle; `.gitignore` `out/`; `vsce publish`.
- **Why:** a live Marketplace listing with install count is the single most
  credible artifact for recruiters/startups.

### Track B — Credibility (≈1 day)
- Unit tests for `traceStore`: loop detection, timeout, batch grouping, session
  routing (all pure functions, trivial to test). GitHub Actions CI badge.
- Persist sessions to disk (workspaceState or a JSON log) → history + replay.

### Track C — The observability story (the differentiator)
This is what reframes TraceBack from "a Claude Code toy" to "local-first
observability for any agent," and aligns with where the industry is going
(OpenTelemetry GenAI semantic conventions are becoming the standard).

1. **Emit OpenTelemetry GenAI spans.** Model each session as a trace; each tool
   call as an `execute_tool` span; use `gen_ai.*` attributes. Lets TraceBack
   *forward* to Langfuse/Phoenix/Honeycomb while staying the local live viewer.
   (`keywords` already lists `opentelemetry` but nothing implements it.)
2. **Generalize ingest beyond Claude hooks.** Accept any agent emitting OTel/JSON
   (LangGraph, OpenAI Agents SDK, MCP tool calls). The `/event` endpoint already
   exists — add an OTLP-shaped adapter.
3. **Session metrics panel:** token usage, cost, duration, tool-frequency
   histogram, error rate. This is the data observability buyers pay for.
4. **Smarter stumble detection:** beyond exact-match loops — near-duplicate edits,
   thrashing on the same file, repeated failed commands, runaway cost.
5. **Post-hoc audit view:** load a saved session, scrub the timeline, diff what
   files changed. Turns it from "live toy" into "incident review tool."
6. **MCP-tool awareness:** label and group MCP tool calls distinctly; agents are
   moving toward MCP and few tools visualize it well.

**Recommended order:** Track A (ship a real artifact) → Track C item 1 (OTel spans,
the headline) → Track B (tests/persistence) → remaining Track C.

---

## 9. Glossary of gotchas for the next agent
- The webview is served from `webview/dist`, **not** the Vite dev server — always
  rebuild the webview to see UI changes.
- `code` on this machine points to **Cursor**, not VS Code. Don't use `code .`.
- Extension-host changes need a full F5 restart; webview-only changes need a
  window reload. Asset URIs are cache-busted, so reloads load fresh JS.
- `traceStore` is a singleton; tests must reset it between cases.
- Loop detection only fires once per session (`if (!session.stumbleAlert)`).
</content>
