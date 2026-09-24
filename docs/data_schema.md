# Data schema — every data shape in the system

One page that says where each kind of data lives, its columns / fields and who reads and writes it. Detailed schemas that already have their own document are linked, not repeated.

| # | Layer | Where | Detail |
| --- | --- | --- | --- |
| 1 | Raw data (organisers) | `data/training dataset/`, `data/testing dataset/` | [`../data/dataset_schema.md`](../data/dataset_schema.md) (§1 below is the summary) |
| 2 | Golden (pre-processed) data | `data/normalized/`, `data/processed/`, `data/quality/` | [`../data/data_schema_high_quality.md`](../data/data_schema_high_quality.md) (§2) |
| 3 | Knowledge base | `knowledge/knowledge.json`, `.md` | §3 |
| 4 | SQLite stores | `observability/*.db`, `data/quality/quality.db` | §4 |
| 5 | Artifact bundle (one per answer) | `memory.db.turns.artifact_json` | §5 |
| 6 | Agent message schemas | `agent/schemas.py` (21 pydantic models), `docs/schemas/*.json` | §6 |
| 7 | Knowledge graph | `observability/kgraph.db` (+ Neo4j mirror) | §7 |
| 8 | API payloads | operator API | [`api_reference.md`](api_reference.md), [`generated/openapi.json`](generated/openapi.json) |
| 9 | UI state | browser | §8 |

Conventions: timestamps are **Berlin local time**, naive, 15-minute grain (`2026-09-25 20:45:00`); station names are the exact data names (`U Hermannplatz (Berlin)`, `S+U Warschauer Str. (Berlin)`), the UI shows short names; lines are `U1 … U9` (no U4 data exists — 8 lines, 167 stations).

## 1. Raw data (organisers' files)

Training split = 2026-06-10 → 2026-09-21 (file suffix `_pre_innotrans`); testing split = 2026-09-22 → 2026-10-01 (`_rest`). The system merges both for answers (window 2026-06-10 05:00 → 2026-10-01 00:45); the normal-flow model is **fitted on the training split only**.

| File | Grain / rows | Columns |
| --- | --- | --- |
| `flows_*.csv` | 15 min × 167 stations (168 columns: one duplicate interchange column) — simulated | `timestamp`, one column per station (passengers) |
| `weather_data_*.csv` | 15 min (real, Berlin) | first column = timestamp, `temp`, `rhum`, `prcp`, `wdir`, `wspd`, `pres`, `cldc`, `coco` |
| `stations_with_ubahn.csv` | 167 stations | `station_id`, `station_name`, `longitude`, `latitude`, `u_bahn_lines` (comma-separated) |
| `berlin_ubahn_connections.csv` | edges | `station_id_1`, `station_id_2` (undirected) |
| `berlin_ubahn_lines_used.csv` | 9 lines | `line_id`, `line_name`, `operator`, `mode`, `product`, `n_variants` |
| `berlin_events_summer_2026_*.csv` | events | `event_name`, `began_local`, `estimated_end_local`, `venue_name`, `address`, `city`, `country`, `segment`, `genre`, `description`, `estimated_attendance`, `event_url` |
| `closures_*.csv` | 30 closures in total | `when`, `duration` (`3h30min`), `description` (line section or station, reason) |
| `energy_consumption_*.csv` | daily, simulated | `timestamp` (date, MM/DD/YYYY), one column per line (U1…U9 minus U4) = MWh that day |

What the data cannot support (capacity, delays, costs, dates outside the window, line U4) is kept as **boundary** entries in the knowledge base (§3) so the agent declines honestly.

## 2. Golden data (pre-processed, read-only)

Produced by the normalization pipeline (`ml/nextmove_pipeline`, [`ml_preprocessing_pipeline.md`](ml_preprocessing_pipeline.md)); read through `ml/golden_data.py` and the `golden_*` MCP tools.

| Path | Content |
| --- | --- |
| `data/normalized/normal_flow_passengers[_test].csv` | wide table `timestamp × station`: passengers expected in a **normal** slot (time of day × day type, typical weather) |
| `data/normalized/normalized_flows[_test].csv` | `log1p(actual) − log(normal)` — 0 = normal, 0.1 ≈ +10 %, 0.69 ≈ twice |
| `data/normalized/normalized_weather[_test].csv`, `normalized_rest[_test].csv` | weather part and the rest (events, closures, anomalies); `total = weather + rest` exactly |
| `data/normalized/episodes[_test].csv` | events and closures with anchors and measured effect |
| `data/normalized/normal_flow_coefficients.csv`, `normal_flow_model.pkl` | per-station effects (rain, temperature, holiday) and the fitted ridge model |
| `data/processed/geocode_cache_pipeline.json` | venue → coordinates |
| `data/quality/quality.db`, `cells.npz` | derived boundaries and checks (§4.5) |

## 3. Knowledge base (`knowledge/knowledge.json`, 55 entries)

Built by `agent/knowledge_build.py` **from the raw CSVs with plain pandas** (never from the agent's tools), so it can vouch for them. List of entries, each:

```json
{ "id": "B-COV", "kind": "boundary | ground_truth | insight", "cats": ["A","B","C",…], "text": "…human sentence…", "value": { …machine-readable numbers… } }
```

`boundary` (14) what the data cannot support · `ground_truth` (37) recomputed numbers the Inspector checks answers against · `insight` (4) notes. `knowledge.md` is the human / Cognee copy; `validation.json` the pressure-ranking skill check.

## 4. SQLite stores

Locations are overridable (`OBS_DB`, `TMT_MEMORY_DB`, `TMT_KG_DB`, `SUBMISSIONS_DB`).

### 4.1 `observability/memory.db` — operator knowledge base (`agent/knowledge.py`)

| Table | Columns |
| --- | --- |
| `turns` (one per answered question) | `id`, `session_id`, `user_id`, `ts`, `question`, `cat`, `answer`, `facts_json`, `sanity_json`, `verdict`, `confidence`, `plan_key`, `data_end`, `accepted`, `plan_json`, `artifact_json`, `operator_score`, `n_feedback`, `linked_from` |
| `operator_feedback` | `id`, `ts`, `turn_id`, `session_id`, `user_id`, `kind` (score \| action), `score`, `comment`, `action_text`, `followed`, `outcome`, `occurred_at`, `stations_json`, `lines_json`, `category`, `problem_key`, `deleted` |
| `insights` | `id`, `ts`, `text`, `cats`, `source` |

`accepted` = the Inspector accepted and the sanity check passed: only accepted answers that were not rated 1–2 are reused from history (same words or subject, same data window, ≤ 24 h). `linked_from` = the earlier turn a resumed conversation continues.

### 4.2 `observability/agent_obs.db` — observability and evaluation (`agent/observability.py`)

| Table | Columns |
| --- | --- |
| `runs` (one per question) | `run_id`, `session_id`, `trace_id`, `ts`, `mode`, `source`, `question`, `answer`, `status`, `error`, `category`, `conf`, `tier`, `facts_status`, `guard`, `total_s`, `route_ms`, `tools_s`, `write_s`, `model`, `tok_in`, `tok_out`, `n_llm_calls`, `n_tool_calls`, `events_json`, `plan_json`, `facts_json`, `timing_json` |
| `spans` (OpenTelemetry) | `span_id`, `trace_id`, `parent_span_id`, `name`, `start_time_unix_nano`, `end_time_unix_nano`, `session_id`, `invocation_id`, `attributes_json` |
| `eval_runs` / `eval_items` | evaluation suite runs and their per-question answers, metrics, checks and judge output |
| `exp_runs` / `exp_turns` | component experiments (router / memory / MCP transport / engine / writer) |
| `ls_runs` / `ls_feedback` | LangSmith-style judge runs and scores |

### 4.3 `observability/submissions.db` — organiser workbook runs (`evaluation/submission_db.py`)

`sub_runs` (`run_id`, `created_at`, `label`, `note`, `git_commit`, `status`, `config_json`, `metrics_json`, `stages`, `n_questions`, `wall_s`) and `sub_answers` (one row per answered question: answer, source, category, decision, verdict, confidence, latency split, tokens, tool calls, models, plan / facts / checks JSON). Every run is also exported to `evaluation/submissions/<run_id>.json`.

### 4.4 `observability/kgraph.db` — knowledge graph store

`kg_nodes` (`id`, `label`, `key`, `props`, `ts`) and `kg_edges` (`id`, `src`, `rel`, `dst`, `props`, `weight`, `ts`); semantics in §7.

### 4.5 `data/quality/quality.db` — derived boundaries (`ml/quality_db.py`)

| Table | Content |
| --- | --- |
| `meta`, `data_issues` | build info; the data-quality issues found while building |
| `stations` | 168 rows: name, lines, lon / lat, `has_flow` |
| `boundaries` | per station × day type × hour: `n_clean`, `normal_median`, `mean_clean`, `p01 … p99`, `max_clean`, `rest_q05/q95`, `band_low`, `band_high` |
| `station_bounds` | per station: `max_all`, `max_clean`, `p999_clean`, `max_episode`, `mean_daily`, weekday / weekend peak hour and average, `clipped_at_500_share`, `hard_ceiling`, `soft_ceiling` |
| `network_bounds` | per day type × hour: network percentiles |
| `episodes`, `event_effects` | events / closures with measured effect; median / p90 / max effect per group |
| `coefficients` | weather and holiday effects per station in % |
| `anomalies`, `outages`, `daily` | unexplained spikes, missing flow, daily passengers per station |

## 5. Artifact bundle (`agent/artifacts.py`, version 1.0)

One per answered question, stored in `turns.artifact_json` and linked on the graph. Everything "why / evidence / which tools" needs, so nothing is recomputed:

```
artifact_version, turn_id, session_id, created_at, question, category, specialist, decision, source, answer_mode (brief|detail),
objective{kind,statement,success_criteria[]}, entities{lines,stations,dates,times,…}, route{specialist,mcp_servers,datasets,tools,ml_engine},
assumptions[], confidence, confidence_reasons[],
verdict{verdict,score,objective_met,issues[],rationale,model,rounds,checks[{id,ok,detail}],ground_truth_ids[],boundary_ids[],similar_cases[]},
tools[{tool,server,args,seconds,bytes,ok,round,result_preview}], datasets[], ml_engine, llm[{role,model,seconds,tok_in,tok_out,error}],
facts{…compact facts JSON…}, brief, report (null until asked), references[{kind,id,…}], sources_line, timing{supervisor_s,worker_evaluator_s,mcp_s,writer_s,writer_llm_s,total_s},
guard, sanity{ok,fails[]}, data_window,
+ added by the pipeline: kind (situation|variant), situation_key, parent_problem_key, problem_key, requires_action, recommended_actions[], precedents[], precedent
```

## 6. Agent message schemas (`agent/schemas.py`)

Everything that moves between the roles is a pydantic model with `extra="forbid"` and a `schema_version`, validated at every hand-over (a malformed message raises — never guessed). JSON Schemas of the main ones are in [`schemas/`](schemas/).

```
question ─► Dispatcher ─► SupervisorPlan ─► Analyst ─► WorkerResult ─► Inspector ─► EvaluatorVerdict ─┐
              (guardrails, history,         ▲   (revise: Adjustments → new WorkerTask)                │
               follow-up, route)            └──────────────────────────────────────────────────────────┘
              └ bounce / decline / answer-from-history ────────────────► FinalAnswer ◄── Writer ◄── WriterInput
```

`GuardrailResult` · `Entities` · `Objective` · `Route` · `HistoryHit` · `FollowUp` · `PartPlan` · `SupervisorPlan` · `WorkerTask` · `ToolCall` · `WorkerResult` · `Adjustments` (closed list) · `CheckResult` · `KGCase` · `EvaluatorVerdict` · `Reference` · `WriterInput` · `FinalAnswer` · `GraphNode` · `GraphEdge` · `GraphCase`. Field-level table: [`agent_architecture_v3.md`](agent_architecture_v3.md) §3.

**Facts** (`WorkerResult.facts`, the only thing the Writer sees): a small JSON with short keys, rounded numbers, station names without `U ` / `(Berlin)`, assumptions as codes, and `status` ∈ `ok | need | unsupported | oos | follow | error | multi`.

**Run configuration** (`TMT_CONFIG`, `agent/config.py`): `router` rules|llm · `memory` session|none|cognee · `mcp` stdio|inmemory|http · `engine` tabpfn|empirical · `writer` llm|template.

## 7. Knowledge graph (`agent/kgraph.py`)

Local property graph (SQLite + NetworkX), mirrored live to Neo4j when configured. Current content (754 nodes / 1240 edges):

| Node labels | Meaning |
| --- | --- |
| `Problem` (116), `Situation` (115), `Conversation` (6) | a question; the situation it is about (variants such as "what about 22:30?" share the situation); a chat |
| `Answer` (122), `Action` (81), `Option` (40), `ActionType` (4) | what was answered, recommended, offered; the type of an action (staffing, replacement bus, information …) |
| `Category` (11), `Domain` (9) | question category (A–H, P, X, …) grouped into domains (Events, Disruptions, Stations, Network, Energy, Diagnostics, Operations planning, Strategy) |
| `Station` (60), `Line` (18), `Event` (8), `Venue` (3) | entities involved |
| `Artifact` (10), `Tool` (8), `Dataset` (7), `Model` (1), `KBEntry` (17), `Document` (22), `Concept` (92) | provenance of an answer; knowledge-base entries; documents and concepts from the LLM Graph Builder step |
| `Feedback` (4) | operator score / action report |

Relationships: `IN_CATEGORY`, `IN_DOMAIN`, `PART_OF` (Problem → Situation), `ANSWERED_BY`, `RECOMMENDS`, `OFFERS_OPTION`, `OF_TYPE`, `INVOLVES`, `AT`, `RELATED_TO`, `VARIANT_OF`, `DISCUSSED` (Conversation → Problem), `ABOUT_SITUATION`, `HAS_ARTIFACT`, `USED_TOOL`, `USED_DATASET`, `USED_MODEL`, `CITES`, `SUPPORTED_BY`, `RECEIVED_FEEDBACK`, `FROM_DOCUMENT`. Sources of problems are tagged (`seed:closure` 26 recorded closures, `seed:training`, `seed:challenge`, `seed:bank`, `runtime` — every accepted answer). Stats and audit: `GET /graph/stats`, `/graph/taxonomy`, `/graph/audit`; narrative: [`conversation_threads_and_graph.md`](conversation_threads_and_graph.md).

## 8. UI state (browser only)

| Key | Where | Content |
| --- | --- | --- |
| `localStorage["toby.pos"]` | `components/ChatOverlay.jsx` | `{corner}` or `{x, y}` of the floating assistant |
| `localStorage["ttmt.tab"]` | `App.jsx` | last open tab |
| React state (`ChatProvider`) | `state/ChatContext.jsx` | `messages[]` (`role`: user \| agent \| choice \| note; agent messages carry the whole API response), `sessionId`, `thread {title, linkTurn}`, `stages[]` (live progress), `convs[]`, `pending`, `open` |

Nothing about the operator is kept in the browser except these two convenience keys; conversations, feedback and precedents live server-side.
