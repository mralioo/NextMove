# Operator desktop (React) — a minimal control-room screen with Toby, the assistant

> **Migrated.** The UI now lives in `frontend/` with the new design (tabs: Operator Desk, Chat Copilot, Network & Heatmap, Flow Analytics, Cascade Simulator, Energy Efficiency) — see [`frontend_migration.md`](frontend_migration.md) and [`modules/frontend.md`](modules/frontend.md). This page still describes the **behaviour** of the desk, Toby, the operations column and the feedback controls, which were kept. File names below (`MapView.jsx`, `Avatar.jsx`, `ChatPanel.jsx`, `OpsLog.jsx`) are those of the archived first version in `frontend_old/`; in `frontend/` they are `components/desk/MapView.jsx`, `components/Toby.jsx`, `components/chat/ChatCore.jsx`, `components/chat/OpsLog.jsx`.

**Status:** built and used end to end (map, replay, chat with operations log, feedback, pending badge); checked with a headless Chrome script (screenshots and clicks), not with a human operator.
**Open it:** `make up` → **http://127.0.0.1:8770/app/** (served by the operator API). Development with hot reload: `make ui` → http://localhost:3000/app/ (proxies `/api`). Build: `make ui-build` (`make up` rebuilds it automatically when the sources are newer than `frontend/dist`).

## 1. What it looks like

```
┌─ NextMove ── Passengers/15 min · Closures · Alerts · Weather ───────────────── clock 08:45 · 2026-09-27 ─┐
│ LINES         │                                             │ ALERTS (closure, unusual load, big event)           │
│ U1 ▮▮▮ 2 614  │   city topology: 167 stations at their      │ CLOSURES NOW  [What should we do?] → asks Toby     │
│ U2 ▮▮▮▮ 5 584 │   coordinates, edges in line colours,       │ EVENTS AROUND                                       │
│ … (click =    │   dots coloured by load vs typical,         │ BUSIEST STATIONS                                    │
│  filter)      │   closures red-dashed, closed stations ×    │ selected station card [Ask Toby about this station] │
│               ├─────────────────────────────────────────────┤                                                     │
│               │ ⏮ ▶ ⏭ 1× date time [jump to a closure ▾]   │                              Toby ◕‿◕ (floats, draggable)│
│               │ ───●──────── slider with closure ticks      │                                                     │
│               │ network sparkline of the day vs typical     │                                                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

There is **no live feed**: the desktop replays the recorded data (training + test split, 2026-06-10 → 2026-10-01) with a time cursor; the clock, map, panels and alerts follow it. It opens on the last recorded closure so something is happening.

* **Map** — `frontend/src/components/MapView.jsx`. Stations from `stations_with_ubahn.csv` at their lon/lat, edges from the connections file in the colour of the line that serves both ends, station dot = passengers vs the typical value for that weekday and 15-minute slot (blue < 0.7 · green normal · yellow 1.25× · orange 1.8× · red 2.5×+), interchanges ringed, labels for the busiest / unusual stations. Wheel = zoom, drag = pan, click a station = card + "Ask Toby about this station". Active closures (from `closures*.csv`, both files merged) draw the blocked edges dashed red and mark unserved stations ×.
* **Lines** — load per line vs typical; click a line to isolate it on the map.
* **Alerts / closures / events / busiest** — computed for the cursor time (see §3). "What should we do?" on a closure sends a ready-made question to Toby.
* **Time bar** — slider over the whole window with a red tick per recorded closure, date and time pickers, play (1× = one 15-minute slot per second, 3×, 8×), ±1 h, "jump to a closure…", and the network total of the day against typical.

## 2. Toby (the assistant)

* **Hovering avatar** (`Avatar.jsx`): floats in the bottom-right corner, blinks, shows a hint on hover ("Ask me" / "3 to close") and a red badge with the number of **situations waiting for your action report**. **Drag** it anywhere; drop it near a corner and it snaps there (position remembered in `localStorage`). **Click** (without dragging) and it flies to the **middle of the page** and opens the conversation; **✕**, a click on the backdrop or the corner buttons (↘ ↖) send it back.
* **Conversation** (`ChatPanel.jsx`): plain-language questions → the agent's **brief** (verdict, up to 3 actions, one watch-out, confidence line; precedent / caution line when operators handled a similar situation). Under every answer: **Rate** (5 stars → `POST /turns/{id}/score`), **Why? / Evidence / Which tools?** (send the follow-up in the same session → the full report from the stored artifact), and for answers that ask for an action **"I did something"** → a one-field form (what did you do, as recommended / modified / something else / did nothing, outcome) → `POST /turns/{id}/action`. Start suggestions include the closure that is active at the cursor time.
* **Operations column** (`OpsLog.jsx`), always next to the conversation — for the last answer: **number of tool calls**, **inference time** (sum of the LLM calls), **tokens** (provider usage; flagged *est.* when only character counts exist), total time, the split Dispatcher / Analyst + Inspector / Writer, the token line (in · out · LLM calls · distinct tools) and the **step-by-step log** (dispatcher route, every MCP tool with its arguments and seconds, inspector verdict, each LLM call with model and tokens, writer). Below it: totals for the session.

* **Conversations** — one conversation = one situation. **History** in the chat header lists earlier conversations (title = first question, category, size, time); opening one brings its messages back and continues it connected to its last answer. A **Topic** pill above the composer names the current topic (**Start a new topic** clears it). When a message does not look connected to the topic, an amber card asks once — **New conversation** / **Continue this topic** — instead of mixing topics. Details: [`conversation_threads_and_graph.md`](conversation_threads_and_graph.md).

## 3. Backend it uses (all in the operator API, port 8770; interactive docs at `/docs`)

Existing (feedback loop, see `docs/operator_feedback_api.md`): `POST /turns/{id}/score`, `POST /turns/{id}/action`, `GET /feedback/pending`, `PATCH /feedback/{id}`, `GET /precedents`, `GET /feedback/stats` …

**Added for the desktop:**

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/chat` `{message, operator_id, session_id?}` | One call for the whole chat turn. Creates the ADK session when new, runs the agent, and returns `answer` (Markdown), `turn_id` / `artifact_turn_id` / `requires_action` / `precedent` (for feedback), `steps` (operations log), `tools` (tool, server, args, seconds, bytes), `llm` (role, model, seconds, tokens), `tokens` {in, out, total, estimated}, `timing` {supervisor_s, worker_evaluator_s, mcp_s, writer_s, total_s}, `counts`, `wall_s`, `session_id` (send it back to continue: "why?", "what about 22:30?"). `503` with a readable message when the agent server is down. Implemented in `backend/chat_bridge.py` (parses the ADK events; ADK URL from `ADK_URL`, `.run/services.json` or `:8000`). |
| `GET /api/v1/conversations`, `GET /api/v1/conversations/{session_id}` | chat history (one entry per conversation) and the messages of one, with `resume_turn_id`; `POST /chat` takes `context_mode` (auto / continue / new) and `link_turn_id` and may answer `needs_choice` |
| `GET /api/v1/ops/topology` | stations (id, short name, lon, lat, lines), edges (with serving lines), line colours, bounding box |
| `GET /api/v1/ops/timeline` | replay window (start, end, 15-min step), default time (an active closure), all closures for the ticks |
| `GET /api/v1/ops/snapshot?at=ISO` | the network at one slot: per station `{v, base, ratio}`, line loads, network total vs typical, busiest stations, **active closures with blocked edges and unserved stations** (from `ml/disruption.py`), events within ± 12 h with a phase (ongoing / starting soon / ended recently), weather, alerts |
| `GET /api/v1/ops/series?date=` | network passengers per 15 min for a day, with typical |

Data logic: `backend/ops_data.py` (uses the same loaders as the agent and the Streamlit dashboard; the typical value = mean of the same station, weekday and slot over the whole window).

## 4. Code map

`frontend/` (Vite + React 18, `react-markdown`, `remark-breaks`; ≈ 290 kB JS) · `src/App.jsx` (state, replay, layout) · `api.js` (client) · `components/MapView.jsx`, `Panels.jsx`, `TimeBar.jsx`, `Avatar.jsx` (Toby + drag/snap), `ChatPanel.jsx` (conversation, stars, action form), `OpsLog.jsx` · `styles.css` (dark control-room theme). Backend: `backend/operator_api.py` (routes, serves `/app`), `backend/chat_bridge.py`, `backend/ops_data.py`.

## 5. Things found and fixed while building it

* A brief whose sections were glued together by the model rendered as one paragraph → `writer.tidy()` puts a blank line before every `**Section:**`.
* A **precedent from another line** (a U2 closure action shown for a U7 closure) — the graph's similarity was word overlap dominated by the boilerplate of closure questions → a precedent now needs a shared line or station (`require_entity`).
* "Why?" after an answer that had not passed the sanity check found no context (only *accepted* answers were kept as follow-up context) → what the operator was shown is now the follow-up context; only accepted answers are reused from history or added to the graph.
* A follow-up reply overwrote the situation id (`artifact_turn_id`) with its own → fixed; feedback always attaches to the original situation.
* Reused ("from history") answers carried no ids → they now carry the id of the answer they came from, so they can be rated too.

## 6. Limits

* **Replay, not live**; there is no notion of "now" (the agent has none either). The cursor time is not sent to the agent: questions carry their own dates.
* **Events have no coordinates** in the data: they are listed, not drawn. Line geometry is straight segments between stations (no track shapes).
* The alert rules are simple thresholds (station ≥ 2.5× typical and ≥ 300 passengers, active closures, big events starting/ending) — not a validated early-warning model.
* The feedback loop needs the agent's **knowledge-base memory** (`memory` = `cognee` in `TMT_CONFIG`, the default when `COGNEE_ENABLED` is set). To use it **without uploading to Cognee**, run with `COGNEE_ENABLED=0` and `TMT_CONFIG='{"memory":"cognee"}'` (that is how the UI test ran, against scratch databases); with plain `memory=session` no turn ids or artifacts exist and the rating / action controls are hidden.
* No authentication (the operator id is fixed to `operator-1` in `App.jsx`); one open chat session; no streaming — the UI shows the phases (Dispatcher → Analyst → Inspector → Writer) while it waits.
* Tested with headless Chrome only (1500 × 900); a narrow-screen layout exists but was not checked; no unit tests for the React code (backend routes and the event parser are tested).
