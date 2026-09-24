# Frontend migration — from the operator desktop v1/v2 to the new UI

**Status:** done and verified with a headless Chrome run (every tab opened and screenshotted, one real question asked through Toby, no page errors) and `make check` (137 tests, guardrail suite 46/46). Not tested by a human operator.
**Open it:** `make up` → **http://127.0.0.1:8770/app/** · development with hot reload: `make ui` → http://localhost:3000/app/ (needs `make up` for the API).

## 1. What was migrated, and from where

| | Path | What it is |
| --- | --- | --- |
| **Old UI** (archived, still in git) | `frontend_old/` | Our first React desktop: one screen (city map with replay + panels) and Toby the floating assistant. Plain CSS, hand-drawn SVG map. |
| **New UI (design source)** | teammate's sibling checkout `innotrans/frontend/` | Dark teal / gold design with a tab bar: Chat Copilot, Network & Heatmap, Flow Analytics, Cascade Simulator, Energy Efficiency; Leaflet map, Recharts charts, voice input/output. Talked to an unversioned `/api/*` backend of its own. |
| **Result** | `frontend/` (this repo) | The new design **plus** everything the old desktop could do, wired to the one operator API (`/api/v1/*`, `backend/operator_api.py`). |

> Assumption made when the migration was requested: "the new frontend" is the teammate's design in `innotrans/frontend/`. `frontend/` in this repo was identical to `frontend_old/` at that moment (the old UI had been moved there). If a different source was meant, only `frontend/src/pages/*` and `index.css` would need to change — the chat, API client and backend are independent of the design.

## 2. What the user sees now

| Tab | Source | What it shows |
| --- | --- | --- |
| **Operator Desk** *(kept from the old UI)* | `pages/Desk.jsx`, `components/desk/*` | KPI cards (passengers per 15 min vs typical, closures, alerts, weather, replay clock), lines panel, city map (SVG, no third party) with load colouring and blocked edges, alerts / closures / events / busiest stations, time bar with replay, closure markers and day sparkline. "What should we do?" and "Ask Toby about this station" open the assistant with a ready-made question. |
| **Chat Copilot** *(new page, same assistant as Toby)* | `pages/Chat.jsx` → `components/chat/ChatCore.jsx` | Full-page assistant: history column, conversation, live team progress while it works, operations column afterwards. |
| **Network & Heatmap** | `pages/Network.jsx` | Leaflet map of the 167 stations and the track edges by line (filter by line, double-click = only this line), hourly heatmap (all stations of one line, else the 20 busiest). OSM background tiles can be switched off (they are the only third-party request the UI makes). |
| **Flow Analytics** | `pages/Analytics.jsx` | Network passengers per day + 7-day mean, top-15 stations, weekday vs weekend commute curves. |
| **Cascade Simulator** | `pages/Disruptions.jsx` | Recorded closures (count, line suspensions vs station closures, list) and a what-if simulator: pick closed stations, time, share diverted (25/50/75 %) and reach (1–3 hops) → load vs typical at the open stations around, `at_risk` from 1.3×. |
| **Energy Efficiency** | `pages/Energy.jsx` | Wh per passenger by line (worst → best), with the evidence per line. |
| **Toby** (on every tab except Chat Copilot) | `components/Toby.jsx`, `components/ChatOverlay.jsx` | Floating avatar: drag anywhere, snaps to a corner, badge = situations waiting for an action report, click = opens centred over the page. Same conversation as the Chat Copilot tab (one shared state). |

Header: pill tabs, API status (`/api/v1/status`: API up/down, agent reachable, writer model in the tooltip). Voice: microphone in the composer (speech-to-text) and "Speak" under each answer (`hooks/useSpeech.js`, browser Web Speech API — nothing is sent by us).

## 3. Chat: what was kept, what is new

Everything the old chatbot offered is kept, in the same behaviour:

| Function | Where in the new code |
| --- | --- |
| Brief answer by default, full report on request | "Why? / Evidence / Which tools?" chips in `chat/Feedback.jsx` send the follow-up in the same session |
| Rating 1–5 → knowledge base, graph, memory | `Feedback.jsx` → `POST /turns/{id}/score` |
| "I did something" action report (as recommended / modified / other / nothing, outcome) → precedents | `Feedback.jsx` → `POST /turns/{id}/action`; the badge on Toby counts `GET /feedback/pending` |
| Operations column: tool calls, inference time, tokens, Dispatcher / Analyst+Inspector / Writer split, steps | `chat/OpsLog.jsx` (unchanged data contract of `POST /chat`) |
| Conversation history, resume, topic pill, **topic-switch choice** (new conversation / continue) | `chat/History.jsx`, `TopicChoice` in `ChatCore.jsx`, `state/ChatContext.jsx` |
| Ask from the map / closures | `useChat().ask(q)` opens Toby and sends |

New in this migration:

* **Streaming.** The UI calls `POST /api/v1/chat/stream` (Server-Sent Events) and shows the four roles working live (Dispatcher → Analyst → Inspector → Writer, each with its tool calls and seconds) instead of an empty spinner. If streaming fails before any answer, the UI falls back to `POST /api/v1/chat` transparently.
* **One shared conversation** (`ChatProvider`) for the floating assistant and the full page.
* Pitch names in the UI (Dispatcher / Analyst / Inspector / Writer with their one-line descriptions).

## 4. Backend changes made for it

| Change | File |
| --- | --- |
| Real data for the new pages: stations, edges, daily flow, hourly profile, heatmap, per-station profile, closures, centrality, energy ranking, cascade | `backend/analytics_data.py` (new; reads the same merged files as the agent via `ops_data.world()`) |
| Endpoints for those pages under `/api/v1/*` | `backend/operator_api.py` |
| `POST /api/v1/chat/stream` (SSE), incremental ADK-event parser | `backend/chat_bridge.py` (`EventParser`, `ask_stream`), `backend/operator_api.py` |
| `/status` now also reports `agent_up`, `models`, `quality_db`, `knowledge_base` | `backend/operator_api.py` |
| `make ui`, `make ui-build`, `make docs`; `make up` rebuilds the UI when its sources are newer than `frontend/dist` | `Makefile`, `scripts/services.py`, `scripts/tasks.py` |
| Tests for all of it (10) | `tests/test_operator_ui_api.py` |

## 5. Old API → new API

The teammate's UI spoke to its own unversioned `/api/*`. Now there is one API, `/api/v1/*` (full list: [`api_reference.md`](api_reference.md)).

| New UI call (`frontend/src/api/client.js`) | Was (teammate's `/api/*`) | Notes |
| --- | --- | --- |
| `GET /api/v1/status` | `/api/status` | + `agent_up`, `models`, `quality_db`, `knowledge_base` |
| `GET /api/v1/stations`, `/network/edges` | `/api/stations`, `/api/network/edges` | same shape |
| `GET /api/v1/flows/daily`, `/flows/heatmap?lines=`, `/flows/station/{name}` | `/api/flows/*` | daily flow drops days that are only partly in the data window |
| `GET /api/v1/flows/hourly-profile` | *(hard-coded array in `Analytics.jsx`)* | now computed from the data |
| `GET /api/v1/closures`, `/centrality?n=` | same | |
| `POST /api/v1/cascade` | `/api/cascade` | now `share` and `hops` are parameters and the response states its method and assumption |
| `GET /api/v1/energy` | `/api/energy` | now `{ranking:[{line, wh_per_pax, …}], unit, method, limits}` (was an array in MWh / 1k pax) |
| `POST /api/v1/chat`, `/chat/stream`, `GET /conversations…`, `POST /turns/{id}/score`, `/turns/{id}/action`, `GET /feedback/pending` | `POST /api/feedback {response_id, rating, …}` | the new UI's one-call thumbs feedback is replaced by our feedback loop (score + action → knowledge base, graph, precedents) |
| `GET /api/v1/ops/topology`, `/timeline`, `/snapshot`, `/series` | — | the Operator Desk (kept from the old UI) |

## 6. Fabricated numbers removed

The design source contained placeholder data that would have been presented as facts. It was removed or replaced:

| Page | Was | Now |
| --- | --- | --- |
| Flow Analytics | 8 hard-coded points for the weekday / weekend curve | 24 hourly points computed from the flows (`/flows/hourly-profile`) |
| Flow Analytics | "Jun 10 – Sep 21", "8,320 timestamps", "Alexanderplatz" as fallbacks | data window and row count from `/status`; no fallbacks |
| Energy | a hard-coded fallback table in MWh / 1k pax and an "HCADE recommended intervention: short-turn U5, ~22 % savings" | the real ranking in **Wh per passenger** (U5 545.5 worst, U9 252.6 best — equal to the evaluation ground truth); explanations are computed from ridership and load-following evidence; **no intervention is invented** (there is no rolling-stock or timetable data; the page prints that limit) |
| Cascade | "85 % rerouting rate", fallback counts 26 / 15 / 11 | share is a visible parameter (25 / 50 / 75 %), counts come from the data (30 closures in training + test) |

## 7. Limits (honest list)

* There is no live feed: the Operator Desk **replays** the recorded window (2026-06-10 → 2026-10-01).
* The cascade simulator is a **what-if with an assumed diversion share**, not a measurement; the data has no diversion behaviour. The agent's own closure answers (Category C) use TabPFN and are the better source for a real decision.
* The energy per passenger is a **proxy**: interchange stations count for every line they serve.
* OSM tiles (Network tab) and, if the operator uses it, the browser's Web Speech service come from third parties; the fonts fall back to system fonts (no CDN links).
* The bundle is ~900 kB (one chunk); no code-splitting yet.
* Not covered by tests: the React components themselves (checked by headless Chrome only).

## 8. Where things are in `frontend/`

```
frontend/
  index.html  vite.config.js  package.json      base "/app/", dev proxy /api → OPERATOR_API (default 127.0.0.1:8770)
  src/main.jsx                                  imports index.css (tokens, shell), chat.css (dk-free, chat-*), desk.css (dk-*)
  src/App.jsx                                   header, tabs, ChatProvider, ChatOverlay
  src/api/client.js                             every API call, the SSE reader (chatStream)
  src/state/ChatContext.jsx                     conversation state shared by Toby and the Chat tab
  src/pages/                                    Desk, Chat, Network, Analytics, Disruptions (cascade), Energy
  src/components/Toby.jsx, ChatOverlay.jsx      floating avatar + centred overlay
  src/components/chat/                          ChatCore, Feedback, OpsLog, ThinkingPanel, History
  src/components/desk/                          MapView (SVG), TimeBar, Panels
  src/hooks/useSpeech.js                        speech in / out
```

More: [`modules/frontend.md`](modules/frontend.md) (module reference), [`operator_desktop.md`](operator_desktop.md) (behaviour of the desk, chat and feedback — still valid for the functions it describes).
