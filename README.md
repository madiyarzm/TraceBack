<div align="center">

# TraceBack

**Real-time observability for AI coding agents — inside VS Code.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![VS Code](https://img.shields.io/badge/VS%20Code-%5E1.85.0-007ACC?logo=visual-studio-code)](https://code.visualstudio.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)

</div>

---

You opened a Claude Code session. The agent starts running. Files are being read. Commands are executing. Edits are landing somewhere.

**You have no idea what's happening — until it's done (or broken).**

TraceBack fixes that. It hooks directly into Claude Code's hook system and streams every tool call into a live, interactive action map in your VS Code sidebar as it happens.

> _"What did my agent just do? Why is it stuck? Did it loop? What file did it touch?"_ — answered in real time, without leaving your editor.

---

## Demo

> ⚡ Add a GIF here showing the live action map in the sidebar while a Claude Code session runs.

---

## How it works

TraceBack injects lightweight `curl` hooks into `~/.claude/settings.json` on activation. Every time Claude Code fires `PreToolUse`, `PostToolUse`, or `Stop`, the hook payload is `POST`ed to a local HTTP server on `localhost:7777`. The extension parses those events, builds a session timeline, and streams updates into a React + ReactFlow webview in the VS Code sidebar.

```
Claude Code CLI
    │  PreToolUse / PostToolUse / Stop hook fires
    │  curl → POST localhost:7777/event
    ▼
TraceBack server  (Node.js, in-process)
    │  parses payload → TraceEvent
    ▼
TraceStore  (in-memory session state)
    │  onDidUpdate event
    ▼
Webview  (React + ReactFlow + Tailwind)
    └─ rendered in VS Code sidebar, live
```

No build step in the hot path. No polling. Zero latency between a tool call firing and the node appearing in your action map.

---

## Features

### Live action map
Every tool call renders as a node — `pending → success / error` — in real time. Edges animate while a step is in-flight and settle once it resolves. Click any node to inspect the full tool input and output in a slide-over detail drawer.

### Stumble detection
TraceBack watches for two failure modes automatically:
- **Loop detection** — three identical tool+label pairs in a row trigger a red alert with a halo animation on the looping nodes.
- **Timeout detection** — a node pending for more than 45 seconds raises a timeout alert.

Both alerts surface immediately in the UI without requiring any configuration.

### Multi-session swimlane view
Running multiple agents in parallel? When more than one session is active, TraceBack switches to a swimlane layout — one horizontal lane per session, each with its own personality badge and color accent. All lanes update live on the same canvas.

### Narrative Engine _(optional)_
Connect a Groq or local Ollama instance and TraceBack generates a 1–2 sentence plain-English summary of the session after each tool call. No jargon, no log-scrolling — just "It's been reading config files and is about to make its first edit."

### Chat assistant _(optional)_
Ask questions about the current session timeline directly in the webview. The LLM answers with the node context already loaded: _"Why did the agent fail?"_, _"What files were touched?"_, _"Is this loop intentional?"_

### Export
Snapshot the current action map as a PNG or dump the raw session JSON — useful for bug reports, post-mortems, or sharing with teammates.

### Auto hook management
TraceBack writes its hooks into `~/.claude/settings.json` on activation and removes them cleanly on deactivation. It never overwrites unrelated config — it surgically adds and removes its own entries only.

---

## Requirements

| Requirement | Version |
|---|---|
| VS Code | `^1.85.0` |
| [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) | latest |
| `curl` | on `PATH` |
| Node.js _(dev only)_ | `^20` |
| Groq API key or Ollama _(optional)_ | for Narrative Engine |

---

## Installation

### From source

```bash
git clone https://github.com/madiyarzhunussov/TraceBack
cd TraceBack
npm install
npm run compile
npm run build:webview
```

Then open the repo in VS Code and press **F5** to launch the Extension Development Host, or package and install it:

```bash
npm run package                           # produces traceback-0.1.0.vsix
code --install-extension traceback-0.1.0.vsix
```

Once installed, the extension activates automatically on VS Code startup — no manual setup required.

---

## Usage

1. Install the extension.
2. Open the **TraceBack** panel in the activity bar (look for the `$(pulse)` icon).
3. Start a Claude Code session in your terminal.
4. Watch tool calls appear in the panel as they happen.

### Commands

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

### Enabling the Narrative Engine

**Groq** (free tier available at [console.groq.com](https://console.groq.com)):
```jsonc
{
  "traceback.llmProvider": "groq",
  "traceback.groqApiKey": "gsk_...",
  "traceback.groqModel": "llama-3.1-8b-instant"
}
```

**Ollama** (fully local, no API key needed):
```jsonc
{
  "traceback.llmProvider": "ollama",
  "traceback.ollamaModel": "llama3.2"
}
```

---

## Architecture

```
src/
├── extension.ts        # activation, command registration, LLM wiring
├── server.ts           # HTTP server that receives hook payloads
├── traceStore.ts       # session state, node building, loop/timeout detection
├── hookManager.ts      # reads/writes ~/.claude/settings.json
├── llmClient.ts        # Groq + Ollama abstraction
└── webviewProvider.ts  # bridges the extension ↔ React webview

webview/src/
├── App.tsx             # ReactFlow canvas, swimlane layout, message handling
├── nodes/              # ToolNode, ThinkingNode, BatchNode, SwimLaneNode
└── components/         # DetailDrawer, SummaryBar, StumbleAlert, Toolbar, SessionPicker
```

The extension and webview communicate over VS Code's `postMessage` / `onDidReceiveMessage` channel. No sockets, no shared memory — just clean VS Code API primitives.

---

## Development

```bash
# Extension (TypeScript, incremental)
npm run watch

# Webview (React + Vite HMR)
npm run dev:webview

# Lint
npm run lint

# Full production build
npm run compile && npm run build:webview
```

Press **F5** in VS Code to launch the Extension Development Host with the extension loaded.

---

## License

[MIT](LICENSE) — built by [Madiyar Zhunussov](https://github.com/madiyarzhunussov)
