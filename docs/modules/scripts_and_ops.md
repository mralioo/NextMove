# `scripts/`, `Makefile`, `tests/` — running and checking the system

## `Makefile` — the commands

| Command | What it does |
| --- | --- |
| `make install` | creates `.venv`, installs dashboard / ML / MCP / agent requirements; if `npm` exists also `npm install && npm run build` in `frontend/` |
| `make up` | `scripts/services.py up`: prepares what is missing (knowledge base, quality database, knowledge graph seed, **UI build when the sources are newer than `frontend/dist`**), then starts every service and prints the addresses. Options: `NO_ADK=1`, `NO_NEO4J=1`, `WITH_DATA_MCP=1`, `PORT=`, `ADK_PORT=` |
| `make down` | stops what `make up` started |
| `make check` | unit tests + guardrail suite + router accuracy (no LLM, no network) |
| `make ui` | UI dev server with hot reload, http://localhost:3000/app/ (API must be up) |
| `make ui-build` | build the UI into `frontend/dist` |
| `make docs` | regenerate `docs/generated/` (MCP tool catalogue, operator-API endpoints, OpenAPI) |
| `make` | help |

## `scripts/services.py` — service supervisor

`up`, `down`, `status`, `restart [service]`. Services: `dashboard` (:8501), `adk-web` (:8000), `operator-api` (:8770, UI at `/app/`), `mcp-quality` (:8768), `mcp-knowledge` (:8766), `neo4j` (Docker, :7474 / :7687), optional `mcp-data` (:8765). State in `.run/services.json`, logs in `.run/logs/<service>.log`. A taken default port → the next free one (the UI finds the API through the same host, so nothing else needs changing). `restart operator-api` after a backend change.

## `scripts/tasks.py` — everything else

`./.venv/bin/python scripts/tasks.py` lists every task with its description (services status/logs, agent queries, MCP servers, resources, Cognee / Neo4j, quality database, knowledge base, knowledge graph, schemas, checkpoints, training, evaluation, submission runs, LangSmith, experiments, backups, `ui-build`, `ui-dev`, `operator-api`, `feedback-sync`). `tasks.py <task> [args]` passes extra arguments through.

## `scripts/gen_docs.py` — generated documentation

Reads the live FastMCP objects and the FastAPI app and writes `docs/generated/mcp_tools.md`, `operator_api.md`, `openapi.json`. Run it after adding a tool or an endpoint (`make docs`).

## `scripts/neo4j.py`

Docker helpers behind `neo4j-up | neo4j-sync | neo4j-down` and the observability backups.

## `tests/` — 137 offline tests

| File | Covers |
| --- | --- |
| `test_agent_schema_v3.py` | schemas, guardrails, supervisor decisions, worker confidence, evaluator, loop |
| `test_fast_pipeline.py`, `test_workflow_v2.py` | router, entities, number guard, categories A–H / P, closure matching |
| `test_brief_and_artifacts.py` | brief vs full report, artifact bundle |
| `test_operator_feedback.py` | feedback loop, precedents, history reuse, HTTP round trip, desk endpoints |
| `test_operator_ui_api.py` | analytics endpoints, cascade, event parser, SSE stream, error event |
| `test_quality_pipeline.py`, `test_disruption.py`, `test_checkpoints.py` | normalization pipeline, quality database and MCP, closure graph logic, checkpoints |
| `test_eval_and_observability.py`, `test_llm_judge.py`, `test_experiments.py`, `test_submission_run.py`, `test_trace_payloads.py` | evaluation, judge, experiments, submission runs, trace payloads |

`conftest.py` switches the Neo4j mirror off; tests that write use temporary databases (`tmp_path`). Last full run: **137 passed**, guardrail suite 46/46, router 71/75.

## Environment

`.env` (never committed; loaded by `env_loader.py`, which walks up from the working directory): `TABPFN_API_TOKEN`; LLM settings per role (`SUPERVISOR_*`, `WORKER_*`, `ROUTER_*`, `WRITER_*`, `EVALUATOR_*`: `_LITELLM_MODEL`, `_API_BASE`, `_API_KEY`, `_API_VERSION`); optional `COGNEE_ENABLED`, `COGNEE_API_BASE_URL`, `COGNEE_API_KEY`; `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`; `LANGSMITH_API_KEY`. Without `.env` the dashboard, the data pages, the operator API's analytics and the MCP servers still start; the agent needs the keys.

Third-party traffic is opt-in: Cognee mirroring only with `COGNEE_ENABLED`, LangSmith only with `--upload` and a key; TabPFN is required for forecasts (`SCENARIO_ENGINE=empirical` avoids it).
