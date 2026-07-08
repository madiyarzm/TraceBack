<div align="center">

# TraceBack

**Real-time observability for AI coding agents — inside VS Code.**

[![CI](https://github.com/madiyarzm/TraceBack/actions/workflows/ci.yml/badge.svg)](https://github.com/madiyarzm/TraceBack/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![VS Code](https://img.shields.io/badge/VS%20Code-%5E1.85.0-007ACC?logo=visual-studio-code)](https://code.visualstudio.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)

</div>

---

You started a Claude Code session and walked away. The agent read fourteen files, ran some commands, edited four. Five minutes later you come back to "All done!" — and you have no idea what actually changed, why, or whether any of it was checked.

**TraceBack answers that.** It hooks into Claude Code's hook system and turns a session into something you can *review*: your prompts become chapters, the agent's work groups under the tasks it declared, and when the run ends you get a net-change diff per file — the true before-and-after, annotated with the agent's own reasoning and a badge for whether anything verified it.

Claude Code's transcript shows you the agent working. TraceBack shows you **what changed, why, and whether it was checked** — so you stay the engineer instead of a spectator.

> _"What did my agent actually change? Why this edit? Was it tested? What did it assume on my behalf?"_ — answered from evidence, in your editor.

---

## Demo

<div align="center">
  <img src="assets/traceback_demo.gif" alt="TraceBack — live agent timeline with anomaly detection, pause and redirect" width="720" />
</div>

---

## Why TraceBack?

Most "agent observability" tools (Langfuse, Arize, LangSmith) ship cloud dashboards that answer *"how much"* — tokens, latency, spend — for production traffic. TraceBack answers a different question, *"what changed and can I trust it"*, for the single developer in the loop: **local-first, in-editor, zero-setup, evidence-based**.

| | Cloud dashboards | Terminal output | **TraceBack** |
|---|---|---|---|
| Setup | API keys, SDKs, project ids | none | **zero** — auto-installs hooks |
| Answers | how much (stats) | what, right now | **what changed, why, verified?** |
| Net diff per file | no | no | **baseline → now, with reasoning** |
| Verification | no | no | **which changes a test actually ran** |
| Loop / drift detection | none | none | **built-in, tuned for low noise** |
| Multi-agent visibility | one project per dashboard | one terminal each | **fleet view in one panel** |
| Cost | $$$ tier | free | **free, local** |

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/madiyarzm/TraceBack
cd TraceBack
npm install
cd webview && npm install && cd ..

# 2. Build
npm run compile && npm run build:webview

# 3. Open in VS Code and press F5
#    (launches the Extension Development Host with TraceBack loaded)
```

Click the TraceBack icon (`$(pulse)`) in the activity bar, then run any Claude Code session in your terminal. Tool calls will start appearing in the sidebar as they happen.

---

## How it works

TraceBack injects lightweight `curl` hooks into `~/.claude/settings.json` on activation. Every time Claude Code fires `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, or `Stop`, the hook payload is `POST`ed to a local HTTP server on `localhost:7777`. The extension parses those events, builds a session timeline, and streams updates into a React webview rendered in the VS Code sidebar.

```
Claude Code CLI
    │  UserPromptSubmit / PreToolUse / PostToolUse / Stop hook fires
    │  curl → POST 127.0.0.1:7777/event
    ▼
TraceBack server  (Node.js, in-process)
    │  parses payload → TraceEvent
    ▼
TraceStore  (in-memory session state) ──► AnomalyDetector  (pure, tail-only, O(1))
    │  onDidUpdate event                    baseline snapshots on first edit
    ▼
Webview  (React + Vite)
    └─ prompt chapters, task blocks, net-change review — sidebar or full panel
```

The server binds `127.0.0.1` only and refuses any request carrying an `Origin` header, so a browser can't reach it. Extension ↔ webview talk over VS Code's `postMessage` — no sockets, no polling.

---

## Features

### Prompt chapters
A session is a book: each prompt you send opens a chapter, and everything the agent did until your next prompt belongs to it. Inside a chapter, actions group under the tasks the agent declared (via its todo tools) — so you read *"Fix stream close → ran the test, edited the file"* instead of a flat scroll. When the agent doesn't plan, actions still group into tidy **Reading / Editing / Running** phase blocks; the view is never a raw list.

### Net-change review
When a run ends, hit **Review changes**. Instead of replaying every edit, TraceBack shows the *net* diff per file — the true baseline-to-now, captured by snapshotting each file the instant before the agent's first edit. Each file carries the agent's own reasoning for the change, the failing command that triggered it (if any), and a verification badge.

### Verification badges
For every changed file: **verified** (a test/build/lint command ran after the last edit and passed), **failing** (it ran and errored), or **unverified** (nothing exercised it). "2 of 5 changed files never checked" is the sentence that decides whether you commit — the agent says "done"; TraceBack says what was actually run.

### Decision & assumption ledger
The judgment calls the agent makes in prose — *"I'll assume the config stays JSON,"* *"went with a regex instead of a dependency"* — mined from the transcript and surfaced as a list. Catch a wrong assumption live and redirect before three files calcify around it; for a learner, it makes visible that coding is choices, not typing.

### Replay
Step through any finished session like a debugger. The cursor slices the event list and *every* view — chapters, files, decisions — recomputes from the slice, so the whole session time-travels together. Read the intent, predict the next action, advance to check: the study loop that keeps the tool from atrophying yours.

### Anomaly engine — tuned for low noise
A pure, tail-only detector that re-evaluates on every event in O(1). It is deliberately quiet, because a tool that cries wolf gets muted:

- **Near-duplicate loop** *(high)* — the exact same command failing 3+ times, or the same file read 5+ times with no edit between. Iterative read→edit→re-read is normal and never flagged.
- **Error thrash** *(high)* — 3 consecutive failed tool calls.
- **Context spiral** *(medium)* — 8+ consecutive reads with no edit, write, or command.
- **Scope creep** *(medium)* — edits landing outside the working directory (Claude's own config excepted).

A stall (a call pending with no result) is treated as **"waiting on you"** — a quiet notice, not a red alarm — because it usually means Claude is at a permission prompt. Real anomalies fire a native VS Code notification and stay as a permanent evidence trail; live alerts self-clear when the condition stops holding.

### Breakpoints for running agents
Hit **⏸ pause** and the agent freezes at its next tool call — TraceBack holds the hook's HTTP response open, exactly like a breakpoint in a debugger. Inspect the timeline, then **▶ resume**… or type into the redirect box:

> _"Stop trying to install that package — use the native library instead."_

Your message is delivered into the agent's context as the reason its call was denied. The agent reads it and changes course, mid-run. Human-in-the-loop steering for black-box agents.

### Guards
Policy rules that protect **every session at once**, no human watching. Four built-in guards toggle on/off from the Guards tab in the panel:

- **Never delete files** — blocks any `rm` command or DeleteFile call
- **Stay in project folder** — blocks edits/writes outside the session's working directory
- **Protect .env and secrets** — blocks reads/edits of `.env`, `.secret`, and `secrets/` paths
- **No git push to main** — intercepts push commands targeting `main`

Custom guards are plain regexes matched against the full tool call (tool name + arguments):

```jsonc
"traceback.customGuards": ["rm -rf|sudo ", "curl.*prod"]
```

A matching call is denied *before it executes* via Claude Code's hook decision protocol; the guard's name is fed back into the agent's context so it knows why and can change course. The CLI asks per session, interactively; guards are fleet-wide policy.

### Files touched, not just files changed
The Files tab toggles between **Changes** (what was created/modified, with +/− lines and verification badges) and **Touched** — a tree of everything the agent *read* as well. "It read fourteen files to make this two-line change" is a coupling insight no chronological view surfaces.

### Multi-agent fleet view
Run multiple Claude Code sessions in parallel? Each gets a distinct identity (color + short tag, so two runs in the same folder never blur together) and a live status. See a failure in agent #3 while watching agent #1.

### Real token & cost metrics
Pulls actual token usage from the Claude Code transcript (`input + cache_read + cache_creation + output`) instead of estimating. Falls back to a character-based heuristic when the transcript isn't reachable.

### Narrative Engine *(optional)*
Connect a Groq or local Ollama instance and TraceBack generates a 1–2 sentence plain-English summary of the session after each tool call. _"It's been reading config files and is about to make its first edit."_

### Chat assistant *(optional)*
Ask questions about the current session directly in the webview. _"Why did the agent fail?"_, _"What files were touched?"_, _"Is this loop intentional?"_ — answered with the timeline already loaded as context.

### Curated payloads & copy-everything
Expanded cards show purpose-built views instead of raw dumps: Bash commands with exit pills, file ops with line/byte metrics, plus a deterministic one-line outcome that explains known failures in plain English (*"path is a directory — Read only works on files"*). Raw input/output stays one click away, and everything is copyable.

### Export
One **Export session** menu, four formats: JSON (the raw session), Markdown (a conversation-shaped post-mortem for a GitHub issue or a handoff to a fresh agent), a self-contained shareable HTML page, or a PNG snapshot.

### Auto hook management
TraceBack surgically adds and removes only its own entries in `~/.claude/settings.json` — never touches unrelated config.

---

## Requirements

| Requirement | Version |
|---|---|
| VS Code | `^1.85.0` |
| [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) | latest |
| `curl` | on `PATH` |
| Node.js *(dev only)* | `^20` |
| Groq API key or Ollama *(optional)* | for Narrative Engine |

---

## Installation

### From source (recommended while pre-release)

```bash
git clone https://github.com/madiyarzm/TraceBack
cd TraceBack
npm install && cd webview && npm install && cd ..
npm run compile && npm run build:webview
```

Open the repo in VS Code and press **F5** to launch the Extension Development Host. To install it permanently:

```bash
npm run package                           # produces traceback-<version>.vsix
code --install-extension traceback-*.vsix
```

---

## Commands

| Command | Description |
|---|---|
| `TraceBack: Open Action Map` | Open the action map in a full editor panel |
| `TraceBack: Clear Current Session` | Reset the current session timeline |
| `TraceBack: Start Listening` | Start the local server and install hooks |
| `TraceBack: Stop Listening` | Stop the local server and remove hooks |

---

## Configuration

All settings live under `traceback.*` in VS Code settings.

| Setting | Default | Description |
|---|---|---|
| `traceback.port` | `7777` | Port the hook server listens on |
| `traceback.autoInstallHooks` | `true` | Auto-install hooks on activation |
| `traceback.llmProvider` | `"disabled"` | Narrative Engine: `"disabled"`, `"groq"`, or `"ollama"` |
| `traceback.groqApiKey` | `""` | Groq API key |
| `traceback.groqModel` | `"llama-3.1-8b-instant"` | Groq model for summaries and chat |
| `traceback.ollamaModel` | `"llama3.2"` | Ollama model (must be pulled locally) |
| `traceback.ollamaPort` | `11434` | Ollama port |
| `traceback.guards` | `{}` | Built-in guard toggles (`never_delete`, `stay_in_project`, `protect_secrets`, `no_push_main`) |
| `traceback.customGuards` | `[]` | Custom guard regexes that block matching tool calls before execution |

### Enabling the Narrative Engine

The AI helper (plain-English summary + sidebar chat) is **opt-in** and supports two backends. You can configure it via VS Code settings *or* a local `.env` file at the repo root — whichever is easier.

#### Option A — Groq (cloud, free tier, recommended)

1. Sign up at [**console.groq.com**](https://console.groq.com) — it's free, no credit card required.
2. Open the menu in the top-right corner → **API Keys** → **Create API Key**.
3. Copy the generated key (starts with `gsk_…`).
4. Drop it into a `.env` file at the repo root:

   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```

   ```dotenv
   # .env
   GROQ_API_KEY=gsk_your_key_here
   # GROQ_MODEL=llama-3.1-8b-instant   # optional override
   ```

5. Reload the VS Code window (`Cmd+R` in the Extension Development Host). TraceBack will auto-detect the key, log `Loaded .env keys: GROQ_API_KEY` to its output channel, and start producing live narrative summaries.

> `.env` is git-ignored. The only env file ever committed is `.env.example`.

If you'd rather use VS Code settings (e.g. for a team-shared workspace), the equivalent is:

```jsonc
{
  "traceback.llmProvider": "groq",
  "traceback.groqApiKey":  "gsk_..."
}
```

VS Code settings always override `.env` when both are set.

#### Option B — Ollama (fully local, no API key)

Install Ollama from [ollama.com](https://ollama.com), pull a model, then:

```jsonc
{
  "traceback.llmProvider": "ollama",
  "traceback.ollamaModel": "llama3.2"
}
```

or in `.env`:

```dotenv
OLLAMA_MODEL=llama3.2
# OLLAMA_PORT=11434
```

---

## Architecture

```
src/  (extension host)
├── extension.ts        # activation, command registration, LLM wiring
├── server.ts           # HTTP server that receives Claude Code hook payloads
├── traceStore.ts       # session state, node building, baseline snapshots
├── anomalyDetector.ts  # pure tail-only detector (loops / thrash / spiral / scope)
├── tokenReader.ts      # tails the transcript for tokens, intents, decisions
├── promptHeuristics.ts # calibrates the todo nudge to prompt substance
├── hookManager.ts      # reads/writes ~/.claude/settings.json
├── guardsManager.ts    # policy rules that deny tool calls before execution
├── sessionArchive.ts   # per-session JSON persistence (history + replay)
└── webviewProvider.ts  # bridges the extension ↔ React webview

webview/src/  (React + Vite)
├── App.tsx             # one bundle, two surfaces (sidebar / panel)
├── chapters.ts         # pure: events → prompt chapters + task groups
├── review.ts           # pure: per-file review annotations (the "why")
├── fileChanges.ts      # pure: changed/touched files + verification
└── components/
    ├── PromptChapterView.tsx   # the main view: chapters, task & phase blocks
    ├── ReviewPanel.tsx         # net-change review with diffs + reasoning
    ├── RightPanel.tsx          # Anomalies / Files / Decisions / Guards tabs
    ├── DiffViewer.tsx          # LCS line diff
    └── TimelineCard.tsx        # one tool call → one card
```

The derivation modules (`chapters.ts`, `review.ts`, `fileChanges.ts`) are pure functions over the event list — which is why replay is nearly free and why they carry the bulk of the test suite.

---

## Development

```bash
npm run watch          # tsc -w  (extension host, incremental)
npm run dev:webview    # vite dev server (webview HMR)

npm run compile        # full extension build
npm run build:webview  # full webview build
npm test               # vitest unit tests
npm run lint           # eslint
```

Press **F5** in VS Code to launch the Extension Development Host with the extension loaded.

### Testing

The pure derivation modules carry the suite — anomaly detector, trace store, chapters, review, file changes, token/decision mining, prompt heuristics — 130+ tests running on every push via GitHub Actions on Node 20 and 22.

```bash
npm test           # one-shot
npm run test:watch # watch mode
```

---

## Roadmap

- **Beyond Claude Code.** A generic OTLP-shaped adapter so any agent (LangGraph, OpenAI Agents SDK, MCP servers) can stream into the same chapter/review views.
- **OpenTelemetry GenAI spans.** Emit `gen_ai.*`-tagged spans per session/tool call to forward to Langfuse / Phoenix / Honeycomb while staying the live local viewer.
- **Review for archived sessions.** Persist baseline snapshots so the net-change review works on history, not just live runs.
- **LLM-assisted ledger.** An optional pass to catch judgment calls the regex miner misses, still offline-first.

---

## License

[MIT](LICENSE) — built by [Madiyar Zhunussov](https://github.com/madiyarzhunussov).
