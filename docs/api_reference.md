# API reference

Two HTTP APIs exist. The **operator API** is the only one the UI (and any operator tool) should use; the **ADK server** is what the operator API talks to.

| API | Default address | Code | Interactive docs |
| --- | --- | --- | --- |
| **Operator API** (FastAPI) | http://127.0.0.1:8770 · UI at `/app/` | `backend/operator_api.py` | http://127.0.0.1:8770/docs (Swagger) · `/openapi.json` |
| **ADK server** (`adk web`) | http://127.0.0.1:8000 | `agent/agent.py` (app `agent`) | http://127.0.0.1:8000/docs · dev UI `/dev-ui/?app=agent` |

* The complete, always-current endpoint table is **generated** from the code: [`generated/operator_api.md`](generated/operator_api.md) (+ [`generated/openapi.json`](generated/openapi.json)); regenerate with `make docs`. This page adds what a table cannot: groups, examples, the streaming format, errors.
* All operator-API paths are under `/api/v1`. JSON in, JSON out, UTF-8. No authentication: it is meant for `127.0.0.1` (CORS origins: `OPERATOR_API_CORS`, default `*` — restrict it for anything but a demo). Bind address and port: `OPERATOR_API_HOST`, `OPERATOR_API_PORT` (default 8770).
* Errors: `422` invalid body / query (FastAPI validation, or unknown station in `/cascade`), `404` unknown turn / feedback id, `503` the agent server is not reachable (`/chat`), `400` a feedback rule is violated (`{"detail": "<readable text>"}`).
* Time: every timestamp is **Berlin local time**, naive ISO (`2026-09-25T20:45:00`) unless it is an epoch (`ts`, `started`, `last` = seconds).

## 1. Groups

| Group (`tags`) | Endpoints | Used by |
| --- | --- | --- |
| **chat** | `POST /chat` · `POST /chat/stream` · `GET /conversations` · `GET /conversations/{session_id}` | Toby, Chat Copilot |
| **feedback** | `POST /turns/{id}/score` · `POST /turns/{id}/action` · `POST /feedback` · `GET /turns/{id}/feedback` · `PATCH /feedback/{fid}` · `DELETE /feedback/{fid}` · `GET /feedback/pending` · `GET /feedback/stats` · `GET /enums` | chat controls, badge on Toby |
| **knowledge** | `GET /turns` · `GET /turns/{id}` · `GET /turns/{id}/artifact` · `GET /turns/{id}/report` · `GET /artifacts/search` · `GET /precedents` · `GET /operator-actions` · `GET /health` | reports, precedents |
| **graph** | `GET /graph/stats` · `GET /graph/taxonomy` · `GET /graph/audit` · `POST /graph/repair` | admin, dashboard |
| **ops** (replay) | `GET /ops/topology` · `/ops/timeline` · `/ops/snapshot?at=` · `/ops/series?date=` | Operator Desk |
| **analytics** | `GET /status` · `/stations` · `/network/edges` · `/flows/daily` · `/flows/hourly-profile` · `/flows/heatmap?lines=` · `/flows/station/{name}` · `/closures` · `/centrality?n=` · `/energy` · `POST /cascade` | Network, Flow Analytics, Cascade, Energy tabs |

Static: the built UI is served at `/app/` (from `frontend/dist`); `/` redirects there.

## 2. Chat

### `POST /api/v1/chat`

Request (`ChatBody`):

```json
{ "message": "Which metro line has the worst energy per passenger?",
  "operator_id": "operator-1",
  "session_id": null,
  "context_mode": "auto",
  "link_turn_id": null }
```

| Field | Meaning |
| --- | --- |
| `message` | 1–1500 characters |
| `operator_id` | becomes the ADK `user_id` |
| `session_id` | omit to start a conversation; send back the returned one to continue (follow-ups such as "why?", "what about 22:30?") |
| `context_mode` | `auto` = ask when the message does not look connected to the conversation's situation · `continue` = connect it anyway · `new` = fresh session, no inherited context |
| `link_turn_id` | only without `session_id`: start the new session **connected** to an earlier answer (resume a conversation from the history) |

Response — an answer:

```json
{ "answer": "**Verdict:** U5 is worst at 545.5 Wh per passenger; …",
  "session_id": "ui-1790274617992", "operator_id": "operator-1", "wall_s": 3.54,
  "turn_id": 348, "artifact_turn_id": 348, "requires_action": false, "precedent": null,
  "source": "worker", "answer_mode": "brief", "guard": "pass",
  "steps":  [ {"kind":"route","stage":"dispatcher","label":"Dispatcher: proceed · category E","seconds":0.01,"detail":"energy · engine none · tools energy_efficiency"}, … ],
  "tools":  [ {"tool":"energy_efficiency","server":"ubahn-flow-data","args":{},"seconds":0.27,"bytes":1234,"ok":true,"round":1,"result_preview":"…"} ],
  "llm":    [ {"role":"writer","model":"azure/gpt-5.6-luna","seconds":3.23,"tok_in":2439,"tok_out":158} ],
  "tokens": { "in":2439, "out":158, "total":2597, "estimated":false },
  "timing": { "supervisor_s":0.01, "worker_evaluator_s":0.28, "mcp_s":0.27, "writer_s":3.23, "writer_llm_s":3.23, "total_s":3.5 },
  "counts": { "tool_calls":1, "distinct_tools":1, "llm_calls":1, "events":9 },
  "context": { "mode":"new", "relation":"related", "reason":"first question of the conversation", "title":"Which metro line …", "linked_turn_id":null } }
```

`timing.supervisor_s` = Dispatcher, `worker_evaluator_s` = Analyst + Inspector (pitch names ↔ code names: [`agent_architecture_v3.md`](agent_architecture_v3.md)). `tokens.estimated` is true when the provider returned no usage and characters/4 was used.

Response — the operator must choose (only with `context_mode: "auto"`; nothing was sent to the agent):

```json
{ "needs_choice": true, "relation": "unrelated", "reason": "another kind of question", "message": "<the message>", "session_id": "ui-…",
  "anchor": { "question": "<first question of the conversation>", "current": "<latest>", "category": "C", "turn_id": 12 },
  "choices": [ {"id":"new","label":"New conversation","hint":"…"}, {"id":"continue","label":"Continue this topic","hint":"…"} ],
  "recommended": "new" }
```

The UI shows the two buttons and repeats the call with `context_mode` = `"new"` or `"continue"`.

### `POST /api/v1/chat/stream` (Server-Sent Events)

Same request body. Response `Content-Type: text/event-stream`; each frame is `event: <name>\ndata: <json>\n\n`:

| `event` | `data` | When |
| --- | --- | --- |
| `choice` | the `needs_choice` object above | the topic-switch question; the stream ends after it |
| `step` | `{kind: route\|tool\|worker\|evaluator\|llm\|writer, stage: dispatcher\|analyst\|inspector\|writer, label, seconds, detail, round?}` | every time the agent finishes an operation — while it is still working |
| `answer` | the complete response, identical to `POST /chat` | once, last |
| `error` | `{"message": "…"}` | the agent server is not reachable, or the run failed (then no `answer`) |

`stage` says which of the four roles the step belongs to (tool calls → `analyst`; the evaluator's rounds and its LLM call → `inspector`; the writer's LLM call and the final answer → `writer`; route decision and the router's LLM fallback → `dispatcher`). Example from a real run:

```
event: step
data: {"kind":"route","stage":"dispatcher","label":"Dispatcher: proceed · category E","seconds":0.01,"detail":"energy · engine none · tools energy_efficiency"}

event: step
data: {"kind":"tool","stage":"analyst","label":"energy_efficiency","seconds":0.27,"detail":"{}","round":1}

event: step
data: {"kind":"evaluator","stage":"inspector","label":"Inspector round 1: accept","seconds":0.0,"detail":"score 1.0 · deterministic"}

event: step
data: {"kind":"writer","stage":"writer","label":"Writer","seconds":3.23,"detail":"48 words · brief"}

event: answer
data: {"answer":"**Verdict:** …", "session_id":"ui-…", …}
```

Browser example (the UI's own reader is `frontend/src/api/client.js → chatStream`); with curl: `curl -N -X POST localhost:8770/api/v1/chat/stream -H 'content-type: application/json' -d '{"message":"…"}'`.

### Conversations

* `GET /conversations?operator_id=&limit=30` → `[{conversation_id, title (first question), category, n_turns, started, last, mean_score, resumed, session_ids[], last_session_id, last_turn_id, first_turn_id}]`, newest first. A conversation resumed from the history is still one entry.
* `GET /conversations/{session_id}` → `{title, resume_turn_id, turns:[{turn_id, question, answer, has_artifact, requires_action, operator_score, action_reported, …}]}`. Reopen a conversation by calling `POST /chat` **without** `session_id` and with `link_turn_id = resume_turn_id`.

## 3. Feedback loop

Full description and rules: [`operator_feedback_api.md`](operator_feedback_api.md). Short form:

* `POST /turns/{turn_id}/score` `{operator_id, score 1-5, comment?}` — a 1–2 rating stops that answer being reused from the history.
* `POST /turns/{turn_id}/action` `{operator_id, action_text (3–1500 chars), followed?: as_recommended|modified|different|none, outcome?: worked|partly|did_not_work|unknown, occurred_at?, score?, comment?}` — stations and lines named in the text are linked in the knowledge graph.
* `POST /feedback` — both in one call (body has `turn_id`). `PATCH /feedback/{id}` adds the outcome later; `DELETE` withdraws a record.
* `GET /feedback/pending?operator_id=&limit=` — answers that asked for an action and have no action report (the badge on Toby).
* `GET /feedback/stats`, `GET /enums` (allowed values), `GET /precedents?question=&category=` (what operators did in similar situations, plus `precedent_line`).

Every submission is fed to the knowledge base (`operator_feedback`), the knowledge graph (Feedback / OperatorAction nodes) and — if configured — the memory agent (Cognee). The response lists what it was fed to (`fed`).

## 4. Knowledge and artifacts

* `GET /turns?operator_id=&session_id=&limit=&needs_action_report=` — recent answers with `operator_score`, `n_feedback`, `requires_action`, `action_reported`, `has_report`, `tools`.
* `GET /turns/{id}` one answer + its feedback · `GET /turns/{id}/artifact` the whole stored bundle ([`data_schema.md`](data_schema.md) §5) · `GET /turns/{id}/report` the full report **if it was generated before** (never generated here: the UI sends "why?" through the chat).
* `GET /artifacts/search?q=&category=&k=` past accepted answers that resemble `q`.
* `GET /operator-actions?category=&station=&min_score=` every action operators reported.
* `GET /health` API and store counters (entries, turns, artifacts, feedback).
* Graph: `GET /graph/stats` (nodes/edges by type), `/graph/taxonomy` (by category and domain), `/graph/audit`, `POST /graph/repair` (nothing is deleted).

## 5. Operator Desk (replay)

There is no live feed; the desk replays the recorded window with a time cursor.

* `GET /ops/topology` → stations `{id, name, lon, lat, lines[], has_flow}`, `edges [{a, b, lines[]}]`, `lines [{line, color}]`.
* `GET /ops/timeline` → `{start, end, step_min: 15, default_at, closures:[{id, when, end, label}]}`.
* `GET /ops/snapshot?at=<ISO>` → `{at, network:{total, typical, ratio}, stations:{<name>:{v, base, ratio}}, lines:[{line, color, load, typical, ratio, n_stations}], top:[{station, id, v, ratio, lines[]}], closures:[{id, kind: line_section\|station, line, from, to, station, start, end, reason, description, path[], blocked_edges, unserved}], events:[{name, venue, phase, attendance}], weather:{temp, prcp, wspd, rhum}, alerts:[{level, text, station?}]}`.
* `GET /ops/series?date=YYYY-MM-DD` → `[{t, total, typical}]` per 15 minutes (sparkline).

## 6. Analytics (Network, Flow Analytics, Cascade, Energy tabs)

Everything is computed from the merged training + test files (`backend/analytics_data.py`); nothing is typed by hand.

| Endpoint | Returns |
| --- | --- |
| `GET /status` | `{status, stations_count, flows_rows, data_window:{start,end}, lines[], agent_up, models:{supervisor,router,worker,evaluator,writer}, quality_db, knowledge_base}` |
| `GET /stations` | 167 × `{name (exact), short_name, lat, lon, lines[], daily_flow, betweenness, degree}` |
| `GET /network/edges` | one row per edge × serving line `{line, from, to, from_lat, from_lon, to_lat, to_lon}` |
| `GET /flows/daily` | `{dates[], values[], rolling_mean[], unit}` — days that are only partly inside the data window are left out |
| `GET /flows/hourly-profile` | 24 × `{hour:"HH:00", weekday, weekend}` (network passengers per hour, mean over weekdays / weekends) |
| `GET /flows/heatmap?lines=U1,U2` | one line → all its stations, else the 20 busiest of the selected lines: `{stations[], station_ids[], values[][24], max_value, count, lines[], unit}` |
| `GET /flows/station/{name}` | `{station, short_name, lines[], daily_mean, profile:[{hour, weekday, weekend}], …}` — mean passengers per hour of one station |
| `GET /closures` | `[{id, when, end, duration_hours, description, closure_type: line_suspension\|station_closure, line, segment, reason}]` (training + test) |
| `GET /centrality?n=15` | busiest stations with betweenness `[{station, short_name, daily_flow, betweenness}]` |
| `GET /energy` | `{ranking:[{line, wh_per_pax, mwh_day, pax_day, n_stations, pax_per_station_day, corr_energy_pax, weekend_vs_weekday_energy_pct, weekend_vs_weekday_pax_pct, efficiency_rank, explanation}], unit, method, limits[], median_pax_per_station_day}` |
| `POST /cascade` | body `{closed_stations[] (exact or loose names), timestamp, share 0.05–1 (default 0.5), hops 1–4 (default 2)}` → `{closed[], unknown[], timestamp, share, hops, moved_passengers_per_15min{}, all_stations:[{station, short_name, is_closed, typical, extra_passengers, overflow_ratio, at_risk}], max_overflow_ratio, method, assumption}`. `overflow_ratio = (typical + extra) / max(typical, 10)`; `at_risk` from 1.3. The share is an **assumption**, stated in the response. Unknown station → 422. |

## 7. ADK server endpoints used

The operator API calls these (`backend/chat_bridge.py`; URL from `ADK_URL`, else `.run/services.json`, else `:8000`); any other client can too.

| Call | Purpose |
| --- | --- |
| `GET /list-apps` | liveness (`/status.agent_up`) |
| `POST /apps/agent/users/{user}/sessions/{session}` (body `{"state": {"link_turn_id": N}}` optional) | create a session; the state links a resumed conversation to its earlier answer |
| `POST /run` `{app_name, user_id, session_id, new_message:{role:"user", parts:[{text}]}}` | run one turn, returns the list of events |
| `POST /run_sse` (same body, `streaming:false`) | same, as an event stream — the source of `/chat/stream` |

The events (and what `customMetadata.kind` values they carry) are catalogued in [`events_and_data_flow.md`](events_and_data_flow.md).

## 8. MCP endpoints (not HTTP JSON APIs)

| Server | Transport / address | Tools |
| --- | --- | --- |
| data + analytics + TabPFN | stdio (spawned by the agent); HTTP `:8765/mcp` with `WITH_DATA_MCP=1` | 17 |
| knowledge | HTTP `:8766/mcp` | 18 |
| quality (+ golden data) | HTTP `:8768/mcp`; in-process for the Inspector | 22 |

Catalogue: [`generated/mcp_tools.md`](generated/mcp_tools.md); how tools are called: [`mcp_and_tools.md`](mcp_and_tools.md).

## 9. Try it

```bash
curl -s localhost:8770/api/v1/status | jq
curl -s 'localhost:8770/api/v1/flows/heatmap?lines=U8' | jq '.stations[:3], .max_value'
curl -s -X POST localhost:8770/api/v1/cascade -H 'content-type: application/json' \
     -d '{"closed_stations":["Hermannplatz"],"timestamp":"2026-09-25T20:45:00","share":0.5,"hops":2}' | jq '.max_overflow_ratio, .all_stations[:3]'
curl -s -X POST localhost:8770/api/v1/chat -H 'content-type: application/json' \
     -d '{"message":"Which metro line has the worst energy per passenger?","operator_id":"me"}' | jq '.answer, .timing'
```
