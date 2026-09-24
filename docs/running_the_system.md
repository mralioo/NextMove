# Running the whole system

Four `make` commands. Everything else is a task listed by one script.

```bash
make install     # once: .venv + every dependency (dashboard, ML, MCP, agent, evaluation)
make up          # run everything (prepares the knowledge base + graph first)
make down        # stop everything `make up` started
make check       # unit tests + guardrail suite + router accuracy — no LLM, no network
```

All other tasks: `./.venv/bin/python scripts/tasks.py` lists them with descriptions; `scripts/tasks.py <task> [args]` runs one, e.g.
`scripts/tasks.py eval --suite challenge --ids CH1,CH3 --cheap` · `scripts/tasks.py agent-query "Line U9 is suspended ... why?"` · `scripts/tasks.py status`.

## What `make up` starts

`make up` prints the addresses (the next free port is used if a default is taken; `tasks.py status` shows them again). Options: `NO_ADK=1`, `NO_NEO4J=1`, `WITH_DATA_MCP=1`, `PORT=9000`, `ADK_PORT=9001`.

| Service | Default address | What it is for |
| --- | --- | --- |
| **dashboard** | http://localhost:8501 | Streamlit: data pages, ML engine, Observability, Evaluation, Experiments, **Agent Workflow** (diagram, supervisor tester, guardrails, schemas, loop traces, knowledge graph) and **Resources** (every resource, its health and links to its UI) |
| **adk-web** | http://localhost:8000 | ADK UI: **chat with the agent**, events, tool calls, and the **Evals** tab (`eval_set_1`) |
| **operator-api** | http://127.0.0.1:8770/app/ (React operator desktop; API docs `/docs`) | Feedback loop + operator knowledge base API for the UI (score of a response, what the operator did, pending situations, precedents) — `docs/operator_feedback_api.md` |
| **mcp-quality** | http://127.0.0.1:8768/mcp | MCP server: normalized data + boundaries (the Inspector's quality database, `docs/ml_preprocessing_pipeline.md`); built on first `make up` (~30 s) |
| **mcp-knowledge** | http://127.0.0.1:8766/mcp | MCP server: ground truth, boundaries, `sanity_check`, history, Cognee recall, `kg_*` graph tools |
| **neo4j** | bolt://localhost:7687 · browser http://localhost:7474 | The knowledge graph as a queryable copy (Docker `nextmove-neo4j`, user `neo4j`, password in `.env`; created once by `tasks.py neo4j-up`) |
| mcp-data *(optional)* | http://127.0.0.1:8765/mcp | Data + analytics + TabPFN MCP server for external clients. The agent starts its own copy on the first question (~30 s warm-up) |

Before starting, it builds the **knowledge base** (`knowledge/knowledge.json`) and the **knowledge graph** (`observability/kgraph.db`, without the LLM step) if they are missing, and warns if `.env` is absent. Keys the agent needs in `.env`: `TABPFN_API_TOKEN`, the LLM settings (`SUPERVISOR_*`, `WORKER_*`, see `agent/llm_config.py`), for memory `COGNEE_ENABLED`, `COGNEE_API_BASE_URL`, `COGNEE_API_KEY`, for Neo4j `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, optionally `LANGSMITH_API_KEY`. The first question after a cold start waits ~30 s for the TabPFN model; later ones take 2–5 s.

## Where the UIs are

| Resource | UI | Link |
| --- | --- | --- |
| Dashboard | Streamlit pages | http://localhost:8501 |
| ADK agent | Chat, events, traces, Evals (`eval_set_1`) | `http://localhost:<adk port>/dev-ui/?app=agent` · API docs `/docs` |
| Neo4j | Neo4j Browser — explore the graph, Cypher | http://localhost:7474 |
| Cognee Cloud | Web app (sign-in) · API docs `<COGNEE_API_BASE_URL>/docs` · graph visualisation saved by `tasks.py cognee-graph` | https://platform.cognee.ai |
| Knowledge MCP server | no web page — `npx @modelcontextprotocol/inspector` | — |
| LangSmith | only with `LANGSMITH_API_KEY` | https://smith.langchain.com |

## Talking to it

* **Chat:** the ADK UI, app `agent`: *"There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00. What does the flow look like at Hermannplatz, and what should we do at 23:15 when it ends?"* → *"What about if it ends at 22:30?"* (follow-up) → *"Why do you say that?"* (explanation).
* **Terminal:** `scripts/tasks.py agent-query "..."` · `agent-cli` · `demo` (scripted 6 turns, small models).

## Tasks (`./.venv/bin/python scripts/tasks.py <task>`)

| Goal | Task |
| --- | --- |
| Services: status, ports, health · follow logs | `status` · `logs [service]` |
| Every resource, whether it works, links to its UI · Cognee memory graph as a file | `resources` · `cognee-graph` |
| Neo4j: start + load the graph · bring it level · stop | `neo4j-up` · `neo4j-sync [--clear]` · `neo4j-down` |
| Knowledge base from the raw CSVs · push to Cognee | `kb-build` · `kb-sync` |
| Knowledge graph: seed · export Cypher | `kg-seed [--no-llm]` · `kg-export` |
| JSON Schemas of the agent messages | `schemas` |
| Score the agent (small model): one brutal question, or `--suite challenge --ids CH1,CH3`, `--suite training --allow-many`, `--cheap` | `eval` |
| Routing / scope decisions on a labelled set · router accuracy (no LLM) | `guardrail-suite` · `eval-router` |
| ADK eval set (3 approximate-answer cases) | `adk-evalset` |
| LangSmith: check key · create dataset · run + upload the experiment (`--all`, `--offline`) · offline scoring of stored runs | `ls-status` · `ls-dataset` · `ls-run` · `ls-eval` |
| **Answer the organiser workbook** (TRAINING + FINAL_TEST + held-out stress run + TEAM_EVIDENCE) and store the run with its configuration: dashboard page *Submission runs* | `submission-run --label NAME [--cheap] [--env K=V] [--config JSON]` |
| Component study and its report | `experiments [--list]` · `experiments-report` |
| TabPFN: checkpoints · train · pressure-ranking skill | `checkpoints [--force]` · `train-disruption` · `train-overcrowding` · `validate-pressure` |
| Trace database: back up before big runs · delete | `backup-obs` · `clean-obs` |
| Dashboard in Docker (no make target) | `docker build -t ubahn-flow-dashboard dashboard` |

**After the Sept 22–30 dataset arrives** (drop the files under `data/`, they are merged): `tasks.py kb-build`, `kb-sync`, `kg-seed`, `neo4j-sync --clear`.

## Troubleshooting

* *A service shows `starting`* — `tasks.py logs <service>`; the dashboard and MCP servers load pandas / the data on first start.
* *Port already in use* — `make up` picks the next free port; `tasks.py status` shows the real one.
* *Stale state after a crash* — `make down` (safe to run twice), then `make up`.
* *Answers say "outside the data"* — the agent knows 2026-06-10 to 2026-09-22 until the new dataset is added.

Architecture, schemas and guardrails: [`agent_architecture_v3.md`](agent_architecture_v3.md).
