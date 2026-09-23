# Session Summary — Talk To My Train (InnoTrans 2026 Hackathon)

Written 2026-09-23 as a handoff/continuation record. Read this first if picking the
project back up in a new session or after moving the folder — it points to every
artifact built, why, and what state it's in.

---

## 1. What this project is

**Challenge:** Alstom InnoTrans 2026 hackathon, *"Talk To My Train"* — build a
conversational AI agent that helps Berlin U-Bahn operators understand, anticipate, and
respond to passenger-flow disruptions, grounded in provided data (network topology,
15-min passenger flows, weather, events, closures, energy). Problem statement in
`docs/innotrans2026_hackathon_problem_statement.md`; dataset schema in
`data/dataset_schema.md`; 11 training questions in `docs/hackathon_questions_training.md`.

## 2. What was built, in order

| # | Deliverable | Location | Status |
| --- | --- | --- | --- |
| 1 | Exploratory Streamlit dashboard | `dashboard/` | ✅ Built, Dockerized, captioned |
| 2 | Question-category analysis (A–H) | `docs/agentic_system_design.md` | ✅ Done |
| 3 | Brainstorm slide deck | `docs/Talk_To_My_Train_Agentic_Brainstorm.pptx` | ✅ Done |
| 4 | TabPFN ML pipeline (overcrowding classifier + flow regressor) | `ml/` | ✅ Built, trained once |
| 5 | MCP server (dataset + TabPFN tools) | `mcp_server/server.py` | ✅ Built, live-tested |
| 6 | 4-role ADK agentic system | `agent/agent.py` | ✅ Built, live-tested end to end |
| 7 | Multi-`.env` merge fix | `env_loader.py` | ✅ Built (fixed a real bug) |
| 8 | Full technical reference doc | `docs/system_design.md` | ✅ Done |
| 9 | Agent roster + prompts + loop diagrams | `docs/agents.md` | ✅ Done |
| 10 | This file | `SESSION_SUMMARY.md` | ✅ You're reading it |

Everything marked ✅ was actually run and verified this session — schemas were dumped
from a live server, the agent loop was traced from real runs, not simulated or assumed.

---

## 3. The dashboard (`dashboard/`)

Multi-page Streamlit + Plotly app, one page per dataset facet, all now captioned:

- `app.py` — home/overview (KPIs, network-wide flow, line coverage, data-quality snapshot)
- `views/1_Network_Explorer.py` — station map (by line/ridership/**fragmentation risk**)
  + resilience ranking (articulation points, betweenness centrality, fragmentation score
  = betweenness × ridership — see `docs/agents.md`/chat history for the full explanation
  if needed again)
- `views/2_Passenger_Flow.py` — busiest/quietest stations, per-station time series,
  hourly profile, day×hour heatmap, commute-peak-vs-network benchmark
- `views/3_Events.py` — filterable event browser + daily attendance vs. flow overlay
- `views/4_Weather.py` — correlation heatmap, daily scatter+trend, condition-code
  breakdown, weather-driven-peak candidates
- `views/5_Closures.py` — Gantt timeline, line/reason breakdowns, duration distribution
- `views/6_Energy.py` — per-line consumption, Wh/passenger efficiency ranking (answers
  training question 5)

Shared logic in `dashboard/utils/data_loader.py` — every downstream consumer (MCP
server, ML pipeline) reads through this same module, so nothing can silently disagree
about how a timestamp/closure/station name is parsed.

**Run it:** `make run` (local) or `make docker-run` (Docker, builds `dashboard/Dockerfile`).
Last run manually on port **8502** (8501 was occupied by something outside this session).

---

## 4. ML engine — TabPFN (`ml/`)

- `ml/features.py` — `build_feature_table()` produces ~1.4M rows (station × 15-min
  timestamp) with `FEATURE_COLUMNS` (17 features: temporal, station, weather, event-proxy,
  closure-proxy) and two targets: `overcrowded` (1 if flow ≥ that station's own 90th
  percentile — no real capacity figure exists in the source data) and `passengers` (raw
  count, doubles as the regression target).
  - **Real data quirk found & fixed:** `stations_with_ubahn.csv` models some interchange
    stations (e.g. "U Stadtmitte (Berlin)") as one row per line under the same
    `station_name`; a naive merge against `flows.csv` (one column per physical station)
    silently duplicated rows. Fixed by collapsing station metadata before joining.
- `ml/train_overcrowding_classifier.py` — offline training script: chronological
  80/20 day-level split (not row-level, to avoid autocorrelation leakage), stratified
  8,000/2,000 sample, `TabPFNClassifier`. **Real result:** ROC-AUC 0.86, PR-AUC 0.446 —
  real signal — but the default 0.5 threshold predicts "normal" for 100% of the test set
  (0 recall on the 15%-positive minority class). **Not fixed** — flagged for whoever
  picks this up (needs threshold tuning or `balance_probabilities=True`).
  - Output already generated once: `ml/output/overcrowding_predictions.csv` (gitignored).
- **Requires `TABPFN_API_TOKEN`** — you set this yourself in `.env` this session.

**Run it:** `make install-ml && make train-overcrowding`

---

## 5. MCP server (`mcp_server/server.py`)

FastMCP, stdio transport. 6 tools, verified live via full MCP protocol round-trip (not
just direct function calls):

| Tool | Type | Notes |
| --- | --- | --- |
| `describe_dataset` | dataset | coverage window, station/line counts |
| `list_stations` | dataset | all 167 stations + lines |
| `resolve_station` | dataset | fuzzy name → exact `station_name` |
| `station_profile` | dataset | peak hour, weekday/weekend rhythm, network-mean benchmark — answers training Q4 |
| `predict_overcrowding_risk` | TabPFN classify | needs exact station + exact 15-min timestamp in coverage window |
| `predict_expected_flow` | TabPFN regress | same input contract |

Both predict tools lazily fit ONE classifier + ONE regressor per server process (cached
for process lifetime) and report `seen_in_training_sample` so a prediction is never
presented as held-out when it wasn't. Full JSON schemas (dumped live) in
`docs/system_design.md` §5.

**Run it standalone:** `make install-mcp && make mcp-server`

---

## 6. Agentic system (`agent/agent.py`) — the main deliverable

**4-role architecture**, matching the engineering blueprint's role table:

| Role | Model (env var) | Tools | Job |
| --- | --- | --- | --- |
| **Supervisor** | `SUPERVISOR_LITELLM_MODEL` (currently `azure/gpt-5.6-luna`) | none directly — 3 sub-agents as tools | classify → delegate → verify → bounded repair (≤1) → relay |
| **Analysis specialist** | `WORKER_LITELLM_MODEL` (currently `gpt-4o-mini`) | full MCP toolset | station profiling + predictions — the actual worker for everything built so far |
| **Scenario specialist** | same worker model | full MCP toolset | what-if/rerouting remit (categories C/F/H) — honestly declines, no real tools exist yet |
| **Verifier** | same worker model | none | reviews draft text against a 5-point checklist, returns PASS/REPAIR/PARTIAL/ABSTAIN |

**Wiring:** each specialist + the Verifier is `Agent(..., mode="single_turn")`, attached
via `sub_agents=[...]` on the Supervisor — ADK auto-exposes each as a callable tool that
runs inline in the Supervisor's turn (its currently-recommended pattern over the older
`AgentTool` wrapper). Full prompts for all four agents, plus mermaid diagrams of the
static wiring and the interaction loop, are in **`docs/agents.md`**.

**Two real bugs found and fixed this session, both empirically diagnosed against live
credentials, not guessed:**

1. **Multi-`.env` shadowing.** Two `.env` files existed (repo root + parent folder,
   `/home/alioo/Desktop/hackathon/innotrans/.env`) with different keys split across them.
   `find_dotenv()` only reads the nearest one, so the Supervisor silently fell back to
   Gemini (no `GOOGLE_API_KEY`) and errored. Fixed with `env_loader.py`, which merges
   every `.env` found walking up the tree, nearest-wins on conflicts. Used by
   `agent/agent.py`, `mcp_server/server.py`, and `ml/train_overcrowding_classifier.py`.
2. **Azure "luna" endpoint 404.** `SUPERVISOR_API_BASE` had the full path+query baked in
   (`.../openai/responses?api-version=...`) with an `openai/` model prefix — wrong on
   both counts for that Azure host. Root-caused via isolated `litellm.completion()` /
   `litellm.responses()` calls with debug logging on. Fix (now automatic in
   `_build_model()`): detect `azure.com` in the host, trim `api_base` to scheme+netloc,
   extract `api_version` from the query string, coerce the model prefix to `azure/`.

**Verified live end to end** (real transcript in `docs/agents.md` §4): Supervisor (luna)
→ Analysis specialist (gpt-4o-mini) → MCP tools → Verifier → one bounded repair → final
answer. The bounded-repair-once rule held correctly. One known rough edge: the Verifier
has twice false-flagged a number that was literally the tool's own JSON field as
"lacking a clear calculation" — not fixed, one-sentence prompt tweak would resolve it.

**Run it:**
```
make install-agent
make agent-web              # ADK dev UI, visualize tool calls — http://localhost:8000
make agent-cli               # interactive terminal chat
make agent-query Q="..."     # one-shot plain-text answer
```

---

## 7. Category coverage (answer to "what can this MVP actually answer")

**Solves: Category D (station profiling)** — training question 4 and any rephrasing of
it, for any of the 167 stations — plus **point prediction** (TabPFN classify/regress,
historical-replay only, not forecasting).

**Does not yet solve:** Categories A, B, C, E, F, G, H (event impact, anomaly
root-cause, disruption rerouting, energy efficiency, network resilience *reasoning* [the
dashboard's fragmentation ranking is a deterministic analytics feature, not an agent
tool yet], latent correlation, reroute behavior). The Scenario specialist is wired in
specifically to decline these honestly rather than let the Supervisor guess — see its
prompt in `docs/agents.md`.

---

## 8. Environment / secrets (do not commit — already gitignored)

Two `.env` files currently in use (merged automatically by `env_loader.py`):
- `<repo-root>/.env` — `TABPFN_API_TOKEN`, `OPENAI_API_KEY`
- `<parent-of-repo>/.env` — `SUPERVISOR_LITELLM_MODEL`, `SUPERVISOR_API_BASE`,
  `SUPERVISOR_API_KEY`, `LUNA_API_KEY`, `WORKER_LITELLM_MODEL`

Full env var reference (every variable, what reads it, what it's for):
`docs/system_design.md` §6.5.

**If moving this folder:** the parent-folder `.env` will NOT move with it (it lives
outside this repo). Either consolidate both `.env` files into the repo root before
moving, or recreate the parent one at the new location — `env_loader.py` will keep
working either way since it walks upward from wherever the scripts are run, but the
values themselves won't follow automatically.

---

## 9. Full file map

```
Makefile                          # every make target for every component below
env_loader.py                     # shared multi-.env merge loader

dashboard/                        # Streamlit exploratory dashboard (Dockerized)
  app.py, views/*.py, utils/data_loader.py, utils/ui.py
  requirements.txt, Dockerfile, .dockerignore, .streamlit/config.toml

ml/                                # TabPFN feature engineering + offline training
  features.py, train_overcrowding_classifier.py, requirements.txt
  output/overcrowding_predictions.csv   (gitignored, generated)

mcp_server/                       # FastMCP server: dataset + TabPFN tools
  server.py, requirements.txt

agent/                            # ADK 4-role agentic system
  agent.py, run_query.py, requirements.txt
  .adk/                            (gitignored, runtime session storage)

docs/
  agentic_system_design.md        # question-category analysis (written before build)
  system_design.md                # as-built technical reference (schemas, ML internals)
  agents.md                       # agent roster, full prompts, loop diagrams
  Talk_To_My_Train_Agentic_Brainstorm.pptx   # brainstorm deck
  innotrans2026_hackathon_problem_statement.md, hackathon_questions_training.md,
  ALSTOM_Challenge_Evaluation_Onepager.pdf, Hackathon_Agenda_UPDATE_15_09_26.pdf
  (organizer-provided materials, untouched)

data/                              # organizer-provided dataset (untouched)
evaluation/                        # organizer-provided answer template (untouched)
```

---

## 10. Suggested next steps (not started)

Roughly in priority order, per the roadmap discussion in the brainstorm deck's closing
slide:
1. Fix the Verifier's false-positive pattern (one prompt sentence).
2. Tune/replace the classifier's decision threshold (currently unusable at default 0.5).
3. Build Category C tooling (closure lookup + graph reroute) — unlocks the Scenario
   specialist and reuses the dashboard's existing `network_resilience()` graph code.
4. Consider an evidence-receipt system (the blueprint's `evidence.get` contract) — right
   now tool traces only exist in the ADK session for the current run, nothing durable.
5. Decide the ADK-vs-LangGraph question the brainstorm deck raised, if still open.

---

## 11. Addendum — Category C solver (added 2026-09-23, later session)

Category C (disruption response, training Q3) is now solved end to end:
`resolve_closure → apply_closure → alternate_paths → scenario_flow`, TabPFN regression
(quantiles) as the ML engine, wrapped in the same MCP server and used by the Scenario specialist.
Details, held-out metrics and the 26-closure case study: **`docs/disruption_case_study.md`**.
Key finding: the dataset has no measurable redistribution around closures, so the solver reports
assumption-based scenarios (low/base/high), never a capacity claim.
Files: `ml/disruption.py`, `ml/demand_baseline.py`, `ml/scenario.py`,
`ml/train_disruption_baseline.py`, `mcp_server/disruption_tools.py`; `make train-disruption`.
Also fixed: `agent/agent.py` now passes env + import path to the MCP child process.

Dashboard update (same session): new **ML Engine** page (`dashboard/views/7_ML_Engine.py`, loader
`dashboard/utils/ml_results.py`) plots TabPFN predictions vs ground truth — regression scatter,
interval reliability, error/hour breakdowns, prediction band vs the real series, classifier
ROC/PR/confusion with a live threshold slider, and the 26-closure observed-vs-predicted explorer.
It only reads files in `ml/output/` (written by `make train-disruption` / `make train-overcrowding`).
Dashboard fixes: `app.py` is now an `st.navigation` router (home moved to `home.py`; sidebar no longer
shows "app"); `make run` now `cd`s into `dashboard/` so `.streamlit/config.toml` (theme) is applied;
Docker mounts `ml/output` as `RESULTS_DIR`.

Checkpoints & ML Engine explanations (same session): `make checkpoints` saves the three inference
models to `ml/checkpoints/` (server model id + exact training sample + fingerprint) and the MCP server
restores them instead of refitting (`ml/checkpoints.py`, `ml/inference_models.py`,
`ml/save_checkpoints.py`). The ML Engine page now shows a scoreboard vs naive baselines and a
what/how-to-read/verdict block under every chart. **Key result: against fair baselines TabPFN is a tie**
(regression MAE 74.6 vs 75.4; classifier ROC-AUC 0.860 vs 0.859) — see `docs/disruption_case_study.md` §2.
Dashboard folder `pages/` was renamed `views/` (pages are registered via `st.navigation` in `app.py`).

Latency work (same session): answers were ~59 s (37 s of LLM calls passing long prose between 4 agents,
22 s of cold tool start-up). New default agent = **router (deterministic, ~1 ms, 96 % on the question bank,
LLM fallback) → executor (parallel MCP playbooks, no LLM, compact facts JSON) → writer (one short LLM call) +
deterministic number guard**. Warm mean **3.8 s**, cold single-shot 16.7 s. ML/MCP side: parquet-cached
feature table, one TabPFN call instead of two, prediction cache pre-warmed for all 26 closures, persistent
pre-warmed MCP server. Full details/measurements: `docs/latency_optimization.md`; commands: `make bench`,
`make eval-router`; old loop: `AGENT_MODE=llm`. The "JEV model" was not integrated (no model card/weights;
unverifiable third-party endpoint). Files: `agent/{router,executor,writer,fast_agent,mcp_runtime,llm_config,bench,eval_router}.py`,
`ml/table_cache.py`.
