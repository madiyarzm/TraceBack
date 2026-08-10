# TraceBack demo kit

Everything needed to capture clean, **reproducible** screenshots of TraceBack —
without waiting on a live agent to happen to do the right thing.

- **`sample-app/`** — a tiny URL shortener (`linkstash`) with a real
  birthday-paradox collision bug. It's the codebase the demo "agent" works on.
- **`feed.mjs`** — a deterministic feeder. It POSTs the exact hook payloads
  Claude Code's real `curl` hooks send, writes a matching transcript (so
  intents, the decision ledger, and token counts populate), and mutates the
  sample app's real files between the Pre/Post events — so the net-change diff
  and verification badge show **genuine** data, not mockups.

## Setup (once)

1. Launch TraceBack: open the repo in VS Code and press **F5** (Extension
   Development Host), or install the extension. Confirm it's listening — the
   output channel says `Listening on http://127.0.0.1:7777`.
2. That's it. The feeder needs no dependencies and only talks to `localhost:7777`.

> If you run more than one TraceBack (e.g. the Marketplace build in another
> window), only one owns port 7777 — events flow to that one. Close the others.

## Capture a session

```bash
node demo/feed.mjs happy      # the flagship — capture most shots here
```

Watch the sidebar/panel fill in live, then screenshot. Re-run any time; each run
is a fresh session and the sample app resets itself first.

| Scenario | Command | What it shows |
|---|---|---|
| **happy** | `node demo/feed.mjs happy` | Prompt chapter → declared plan → task blocks → per-call **intents** → an **AskUserQuestion** card → an **Edit** with a real **net-change diff** → a created test → a passing `npm test` → **verified** badges → the **decision & assumption ledger**. This one session covers most of the product. |
| **loop** | `node demo/feed.mjs loop` | High anomaly: the same command failing 3× → red "near-duplicate loop" banner + permanent record. |
| **thrash** | `node demo/feed.mjs thrash` | High anomaly: 3 failed tool calls in a row → "error thrash". |
| **spiral** | `node demo/feed.mjs spiral` | Medium anomaly: 9 reads with no action → "context spiral". |
| **guard** | `node demo/feed.mjs guard` | A guard **denying** a call (red, in-context reason). **Enable the "Never delete files" guard in the Guards tab first**, or the `rm` just runs. |
| **fleet** | `node demo/feed.mjs fleet` | Three agents live at once with distinct identities; the third is stuck in a loop — the "watch #1 while #3 fails" shot. |
| **all** | `node demo/feed.mjs all` | Every scenario back to back (also fills the session history rail). |

## Suggested screenshot order

1. **`happy`**, mid-run — live timeline with the working glow, an in-progress
   task block, and climbing tokens.
2. **`happy`**, after it finishes — hit **Review changes** for the net-change
   diff + verification badges; open the **Decisions** tab for the ledger; expand
   the **AskUserQuestion** card and the **Bash** card (now split, readable).
3. **`loop`** — the red anomaly banner.
4. **`fleet`** — the multi-agent view.
5. **`guard`** (guard enabled) — the denied call.

## How it stays honest

The feeder never fakes a diff or a green check. For `happy` it actually rewrites
`sample-app/src/shortener.js` to a collision-free base62 generator and adds a
real test — `cd demo/sample-app && npm test` passes with all three tests — so
the "verified" badge reflects a command that truly ran green. The only synthetic
part is the timing and the transcript prose (the narration and judgment
sentences), which stand in for what a real agent would have written.

## Reset

Transient files (`.transcripts/`, the generated `uniqueness.test.js`) are
git-ignored and safe to delete. The feeder restores `shortener.js` to its buggy
baseline at the start of every `happy` run.
