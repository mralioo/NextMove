# Operator feedback loop — what it does, and the endpoints for the UI

**Status:** backend done and tested (`tests/test_operator_feedback.py`, 108 tests pass); UI not built — this document is the contract for it.
**Code:** `agent/feedback.py` (logic) · `agent/knowledge.py` (`operator_feedback` table) · `agent/kgraph.py` (graph nodes, precedents) · `backend/operator_api.py` (HTTP API) · `mcp_server/knowledge_server.py` (3 new read tools).

## 1. The loop

```
 operator asks ──▶ agent answers with a BRIEF ──▶ UI shows it + "Rate this answer" (1-5) + "What did you do about it?" field
        ▲                                                   │  (only when the answer asked for an action: requires_action = true)
        │                                                   ▼
 next similar situation: the brief carries one         operator API  ──▶ operator knowledge base (SQLite)   score / action / outcome
 line  "Precedent: in a similar case operators …       (POST …/score,  ──▶ knowledge graph (+ Neo4j)        Feedback, OperatorAction nodes
 (rated 5/5, worked)"  or  "Caution: … did not work"     …/action)     ──▶ memory agent (Cognee, if it is the configured memory)
```

Two kinds of feedback:

| Kind | What the operator gives | When |
| --- | --- | --- |
| **score** | 1–5 (thumbs up/down = 5 / 1) + optional comment | any answer, any time |
| **action** | free text of what they actually did about the situation, whether they followed the advice (`as_recommended` · `modified` · `different` · `none`), optionally how it went (`worked` · `partly` · `did_not_work` · `unknown`) and when | answers with `requires_action = true` (closures, events, pressure rankings, or any question that asked what to do); the outcome can be added later with `PATCH` |

**What the system does with it (all implemented):**

1. **Operator knowledge base** — one row per submission in `operator_feedback`; the answer's turn row carries `operator_score` (mean) and `n_feedback`. **An answer rated ≤ 2 is never reused from history again** ("answer from history" skips it), so a bad answer does not get served twice.
2. **Knowledge graph** — `Problem –RECEIVED_FEEDBACK→ Feedback{kind, score, comment, followed, outcome, action, operator, occurred_at}`; `Artifact –RECEIVED_FEEDBACK→ Feedback`; an action report also creates `Feedback –DESCRIBES→ OperatorAction{text}`, `Problem –OPERATOR_TOOK→ OperatorAction`, `OperatorAction –AT→ Station`, `–ON_LINE→ Line` (extracted from the operator's own words with the router's station matcher), and `OperatorAction –MATCHES→ Action` when it carries out something the answer recommended; the `Answer` node keeps `n_ratings` / `mean_score`. Mirrored to Neo4j like the rest of the graph.
3. **Memory agent** — when the configured memory is Cognee, one entry per submission (question, score, what the operator did, followed, outcome). *This is an upload to a third party, like the answers themselves; it is skipped when memory is `session`.*
4. **Later decisions** — for every new answer the agent looks up **precedents** (similar past problems that have operator reports) and adds one deterministic line to the brief: `_Precedent: in a similar case operators sent two staff to Neukölln (rated 5/5, worked)._` or, if the past outcome was bad (outcome `did_not_work` or score ≤ 2), `_Caution: in a similar case operators … — it did not work out (rated 2/5)._` (a caution wins over a precedent). The full precedent list is in the answer's artifact and in the full report ("Operator precedents"). It never changes the numbers of an answer and is skipped in evaluation runs (`TMT_HISTORY=off`). The same data is available to agents/evaluators through the knowledge MCP tools `operator_precedents`, `operator_actions`, `operator_feedback_stats`.

Verified live with scratch databases: answer → `POST …/action` (score 5, `worked`) → graph had `Feedback 1, OperatorAction 1` → a similar question from another operator returned the brief with the precedent line.

## 2. Services and addresses

| Service | Address | Started by |
| --- | --- | --- |
| **Operator API** (this document) | `http://127.0.0.1:8770` (env `OPERATOR_API_PORT`; next free port if taken) · interactive docs `/docs` · schema `/openapi.json` | `make up` (service `operator-api`) or `./.venv/bin/python scripts/tasks.py operator-api` |
| **ADK agent server** (the chat) | `http://localhost:8000` (`ADK_PORT`) · `/docs` | `make up` |
| Streamlit dashboard | `http://localhost:8502` — *Agent Workflow → 5c* shows artifacts and the feedback table | `make up` |

CORS is open (`OPERATOR_API_CORS`, comma-separated origins, default `*`). **There is no authentication** in this prototype: put the API behind the operator's gateway. `operator_id` is a plain string = the ADK `user_id` of the session.

## 3. Getting the ids from the chat (ADK endpoints the UI also needs)

```
POST http://localhost:8000/apps/agent/users/{operator_id}/sessions/{session_id}     body {}                      → creates the session
POST http://localhost:8000/run     {"app_name":"agent","user_id":"op1","session_id":"…","new_message":{"role":"user","parts":[{"text":"…"}]}}
                                   → JSON list of events; (`/run_sse` streams them)
```
The **answer** is the last event with `author == "writer"` and `customMetadata.kind == "answer"`; its text is in `content.parts[0].text` (Markdown). Its `customMetadata`:

| Field | Meaning for the UI |
| --- | --- |
| `turn_id` | id of **this reply** in the knowledge base (exists for every stored reply, also declines / "need more detail") |
| `artifact_turn_id` | id of the **situation** the reply belongs to: the answer that produced the analysis. For a normal answer = `turn_id`; for a follow-up reply ("why?") it points at the original answer. **Use this id for feedback** (fall back to `turn_id` when null) |
| `requires_action` | `true` → show the **"What did you do about it?"** field under this answer |
| `precedent` | the precedent / caution line already inside the text (also given separately, e.g. to style it) |
| `answer_mode` | `brief` (default) or `detail`; `source` = `worker` / `follow_up` / `history` / `decline` / `bounce` …; `answer_words`, `guard`, and the timing fields |

Show the **full report** by sending "why?" (or "show me the evidence", "which tools did you call?") to the *same session*: the reply is the stored report (no recomputation). `GET /api/v1/turns/{id}/report` tells whether it already exists.

## 4. Endpoint reference (`/api/v1`, JSON)

Errors: `404` unknown turn / feedback · `422` invalid value (message in `detail`, e.g. `score must be an integer 1-5`) — the UI can show `detail` verbatim.

### Feedback (write)

| Method · path | Body | Returns |
| --- | --- | --- |
| **`POST /turns/{turn_id}/score`** `201` | `{"operator_id":"op1","score":4,"comment":"clear"}` (`score` 1–5 required) | the stored record + `fed` |
| **`POST /turns/{turn_id}/action`** `201` | `{"operator_id":"op1","action_text":"Sent two staff to Neukölln and ordered the bus","followed":"as_recommended","outcome":"worked","occurred_at":"2026-09-25T21:05","score":5,"comment":"…"}` — only `action_text` (3–1500 chars) is required; `score` optionally rates the answer in the same submission | the stored record + `fed` |
| `POST /feedback` `201` | same fields plus `turn_id`; score and/or action in one call | the stored record + `fed` |
| `PATCH /feedback/{feedback_id}` | any of `score, comment, action_text, followed, outcome, occurred_at` — typically `{"outcome":"did_not_work"}` when the result is known | the updated record (graph refreshed) |
| `DELETE /feedback/{feedback_id}` | — | `{"deleted": id}`; the record no longer counts in scores, precedents or the pending list |

Record returned by the writes:
```json
{"feedback_id": 12, "turn_id": 41, "session_id": "s1", "operator_id": "op1", "kind": "both", "score": 5, "comment": null,
 "action_text": "Sent two staff to Neukölln and ordered the bus", "followed": "as_recommended", "outcome": "worked", "occurred_at": null,
 "stations": ["S+U Neukölln (Berlin)"], "lines": ["U7"], "category": "C", "problem_key": "bb82c5ea0dc8", "ts": 1790260566.09,
 "fed": {"knowledge_base": true, "knowledge_graph": true, "memory_agent": false}}
```
`kind` is `score`, `action` or `both`. `fed.memory_agent` is `true` only when the memory agent (Cognee) accepted the entry; `fed.knowledge_graph_error` appears if the graph write failed (the record is safe in the knowledge base; `tasks.py feedback-sync` re-feeds the graph).

### Turns and artifacts (read)

| Method · path | Returns |
| --- | --- |
| **`GET /turns?operator_id=&session_id=&limit=30&needs_action_report=false`** | recent replies, newest first: `turn_id, operator_id, session_id, ts, question, category, answer, confidence, accepted, operator_score, n_feedback, has_artifact, has_report, requires_action, recommended_actions[], problem_key, tools[], precedent, action_reported`. Use it for the history panel and for "rate this" / "did you act?" state |
| **`GET /feedback/pending?operator_id=&limit=20`** | the same rows, only accepted answers that `requires_action` and have **no action report yet** — the list behind a "3 situations to close" badge; ask the operator later what they did and how it went |
| `GET /turns/{turn_id}` | one reply + `feedback[]` (all its records) |
| `GET /turns/{turn_id}/feedback` | the records only |
| `GET /turns/{turn_id}/artifact` | the whole stored bundle: plan, every tool call with arguments / time / result preview, facts, confidence + reasons, evaluator checks, LLM calls (model, seconds, tokens), brief, report, precedents, recommended actions. `404` for replies without one (declines, need-more-detail, answers stored before artifacts) |
| `GET /turns/{turn_id}/report` | `{has_report, report, method, brief}` — `method` = the deterministic "how this was worked out" section, always available; `report` is filled once "why?" was asked in chat |

### Knowledge and statistics (read)

| Method · path | Returns |
| --- | --- |
| `GET /feedback/stats?operator_id=` | `{n_scores, mean_score, score_histogram{"1":n…}, n_action_reports, followed{…}, outcome{…}, by_category{C:{n,mean_score}}, turns_requiring_action, action_report_rate}` |
| `GET /precedents?question=&category=&k=2` | `{precedents:[{problem, similarity, category, n_reports, mean_score, actions:[{action, followed, outcome, score, operator, occurred_at}], recommended[]}], precedent_line}` — e.g. to preview "what did operators do last time" while the operator types |
| `GET /operator-actions?category=&station=&min_score=&limit=30` | every reported action, newest first, with the situation text |
| `GET /artifacts/search?q=&category=&k=5` | past accepted answers that resemble `q` (brief, confidence, tools, `has_report`) |
| `GET /graph/stats` | nodes / relationships by type (Feedback, OperatorAction, Artifact, Tool …) |
| `GET /health` · `GET /enums` | liveness + store sizes · allowed values (`score` range, `followed`, `outcome`, max text length) for building the form |

## 5. What the UI has to build (suggested)

1. **Under every answer:** a 5-star (or thumbs) control → `POST /turns/{artifact_turn_id ?? turn_id}/score`; optional comment box.
2. **Under answers with `requires_action`:** a text field "What did you do about it?" + three-way choice *as recommended / modified / something else / did nothing* (`followed`) + optional "when" → `POST …/action`. Keep it one field long: the operator is busy; everything but the text is optional.
3. **Badge / list "situations to close":** `GET /feedback/pending?operator_id=…`; tapping one opens a small form to add the **outcome** (`worked` / `partly` / `did_not_work`) with `PATCH /feedback/{id}` (or a first action report if none exists).
4. **"Show full report" button:** send "why?" to the same ADK session; render the Markdown. Optionally a link to `GET /turns/{id}/artifact` for an engineer view.
5. **Show the precedent / caution line** the brief already contains (it is the italic line above the confidence line); `GET /precedents` can pre-warm a side panel.
6. **Confirmation:** the write returns `fed`; show "saved" (and, if `knowledge_graph_error`, a discreet warning).

## 6. Data model (SQLite, `observability/memory.db`)

`operator_feedback(id, ts, turn_id, session_id, user_id, kind, score, comment, action_text, followed, outcome, occurred_at, stations_json, lines_json, category, problem_key, deleted)`; `turns` gained `operator_score`, `n_feedback`; the artifact JSON of a turn gained `requires_action`, `recommended_actions`, `precedents`, `precedent`. Delete is soft (`deleted = 1`). The graph is `observability/kgraph.db` (+ Neo4j mirror). Override paths with `TMT_MEMORY_DB`, `TMT_KG_DB`.

## 7. Try it

```
make up
curl -s localhost:8770/api/v1/health
curl -s "localhost:8770/api/v1/feedback/pending?operator_id=op1"
curl -s -X POST localhost:8770/api/v1/turns/41/action -H 'content-type: application/json' \
     -d '{"operator_id":"op1","action_text":"Sent two staff to Neukölln","followed":"as_recommended","outcome":"worked","score":5}'
curl -s "localhost:8770/api/v1/precedents?question=U7%20suspended%20Hermannplatz%20staff&category=C"
```

## 8. Limits and open points

* No authentication / authorisation and no per-operator permissions; `operator_id` is trusted. Add it at the gateway before any real use.
* Matching a situation to precedents is word overlap + category + shared stations/lines (the same similarity the graph already used); it can miss paraphrases and it can match a situation that differs in an important detail — that is why it is shown as a *precedent line*, never as a recommendation, and the operator sees the operator's own words.
* One score per submission; scoring the same answer twice averages the scores. There is no per-operator weighting, no de-duplication of identical action reports, no moderation of free text (it is shown back to other operators as part of a precedent line: review before a real deployment).
* An action report is free text: stations and lines are extracted with the router's matcher (station names and `U1`–`U9`); durations, quantities ("two staff") stay in the text.
* The precedent line needs the production graph (`TMT_HISTORY` on) and at least one report on a similar problem; evaluation runs never show it.
* Feedback about a follow-up reply (the full report) should be sent with the `artifact_turn_id` of the original situation — sending the reply's own `turn_id` also works but links to a row without an artifact.
* Cognee mirroring uploads feedback text to a third party when it is the configured memory (a scratch test of the live flow above did upload one entry).
