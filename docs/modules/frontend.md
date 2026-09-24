# `frontend/` — operator UI (React)

**Purpose.** The screen the control-room operator uses: the replayed network (Operator Desk), the assistant (Toby / Chat Copilot) and the analytics tabs. Vite 5 + React 18; react-leaflet (map), recharts (charts), lucide-react (icons), react-markdown + remark-breaks (answers). Migration story, old→new mapping and removed placeholder data: [`../frontend_migration.md`](../frontend_migration.md). Behaviour of desk / chat / feedback: [`../operator_desktop.md`](../operator_desktop.md). API used: [`../api_reference.md`](../api_reference.md).

## Run / build

| Command | Result |
| --- | --- |
| `make up` | builds the UI when its sources are newer than `frontend/dist`, serves it at **http://127.0.0.1:8770/app/** |
| `make ui` | dev server with hot reload, http://localhost:3000/app/ (proxies `/api` to `OPERATOR_API`, default `http://127.0.0.1:8770`; the API must be running) |
| `make ui-build` | `npm install && npm run build` → `frontend/dist` |

Needs Node.js 18+. `frontend/dist` and `node_modules` are not committed.

## Structure

```
src/main.jsx                    entry: index.css (tokens, shell), chat.css (chat-*, Toby), desk.css (dk-*)
src/App.jsx                     header + pill tabs, status badges (/status), ChatProvider, ChatOverlay (Toby) on every tab but Chat Copilot
src/api/client.js               every call to /api/v1 (JSON) and the SSE reader chatStream()
src/state/ChatContext.jsx       ChatProvider / useChat(): messages, session, thread, streaming stages, history, pending badge, send / choose / ask
src/pages/Desk.jsx              Operator Desk: KPIs, lines, map, time bar, alerts / closures / events / busiest
src/pages/Chat.jsx              Chat Copilot: ChatCore variant="page"
src/pages/Network.jsx           Leaflet map + hourly heatmap
src/pages/Analytics.jsx         daily flow, top stations, weekday / weekend curves
src/pages/Disruptions.jsx       recorded closures + cascade simulator
src/pages/Energy.jsx            Wh per passenger by line + evidence
src/components/Toby.jsx         floating draggable avatar (snaps to corners, badge, hint)
src/components/ChatOverlay.jsx  Toby + centred overlay frame
src/components/chat/            ChatCore (messages, composer, topic choice), Feedback (stars, follow-up chips, action form),
                                OpsLog (tool calls, inference time, tokens, split, steps), ThinkingPanel (live team progress), History
src/components/desk/            MapView (SVG, no third party), TimeBar (replay), Panels
src/hooks/useSpeech.js          Web Speech API: speech-to-text and speak-the-answer
```

Pages other than Desk and Chat are mounted only while visible (the map and the charts need a visible container to measure); Desk and Chat stay mounted so the replay and the conversation keep their state.

## State and data flow

* One `ChatProvider` at the root: Toby and the Chat Copilot tab show the same conversation. `send()` streams (`POST /chat/stream`), updates the four role stages from `step` events, appends the `answer`, refreshes history and the pending-action badge; on a stream failure before any answer it falls back to `POST /chat`.
* `choose(index, "new" | "continue")` answers the topic-switch card and re-sends with the chosen `context_mode`.
* `openConversation(c)` reloads a conversation and resumes it connected to its last answer (`link_turn_id`).
* `ask(q)` (used by "What should we do?" and "Ask Toby about this station") opens the overlay and sends.

## Conventions

CSS classes: `chat-*` (assistant), `dk-*` (Operator Desk), shared tokens (`--teal`, `--gold`, `--danger`, …) and components (`.card`, `.pill-tab`, `.pill-badge`, `.metric-card`) in `index.css`. No CDN fonts or stylesheets (Inter / Space Grotesk fall back to system fonts). Third-party requests: OpenStreetMap tiles on the Network tab (switchable off) and the browser's own speech service if the microphone is used.

## Tests / verification

No component tests. Verified by a headless-Chrome script (every tab screenshotted, one question asked through Toby, page errors collected) and by the API tests in `tests/test_operator_ui_api.py`. `frontend_old/` keeps the first desktop for reference (not built by anything).

## Limits

Replay only (no live feed); one JS chunk of ~900 kB; the cascade simulator is an assumption-driven what-if; accessibility (keyboard navigation of the map, screen-reader labels) is minimal.
