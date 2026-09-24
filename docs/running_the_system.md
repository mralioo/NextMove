# Running the whole system

One command starts everything; one stops it.

```bash
make install-all      # once: .venv + all dependencies
make up               # prepare + start dashboard, ADK chat UI, knowledge MCP server
make status           # what runs, on which port, healthy or not
make logs             # follow the logs (make logs S=dashboard for one service)
make down             # stop everything `make up` started
```

`make up` prints the addresses (the next free port is used if a default is taken). Nothing needs Docker.

## What starts

| Service | Default address | What it is for |
| --- | --- | --- |
| **dashboard** | http://localhost:8501 | Streamlit: data pages, ML engine, Observability, Evaluation, Experiments and the **Agent Workflow** page (workflow diagram, supervisor tester, guardrails, schemas, loop traces, knowledge graph) |
| **adk-web** | http://localhost:8000 | ADK dev UI: **chat with the agent** (select the app `agent`), see every event, tool call and timing |
| **mcp-knowledge** | http://127.0.0.1:8766/mcp | MCP server (streamable HTTP): ground truth, boundaries, `sanity_check`, history, Cognee recall, `kg_*` graph tools — for evaluators and other agents |
| **neo4j** | bolt://localhost:7687 · browser http://localhost:7474 | The knowledge graph as a queryable copy (Docker container `nextmove-neo4j`, created once by `make neo4j-up`; user `neo4j`, password in `.env`). `make up NO_NEO4J=1` skips it |
| mcp-data *(optional)* | http://127.0.0.1:8765/mcp | MCP server: datasets, analytics, TabPFN tools. Start with `make up WITH_DATA_MCP=1`. The agent does **not** need it: it starts its own stdio copy on the first question (~30 s warm-up) |

Options: `make up NO_ADK=1` (skip the chat UI) · `make up WITH_DATA_MCP=1` · `PORT=9000 make up` · `ADK_PORT=9001 make up`.

## What `make up` prepares

Before starting, `scripts/services.py` makes sure two things exist and builds them if not:
1. the **knowledge base** — `knowledge/knowledge.json` (`make kb-build`, ~10 s);
2. the **knowledge graph** — `observability/kgraph.db` (`make kg-seed ARGS=--no-llm`, no LLM calls; run `make kg-seed` once for the LLM-extracted part).

It also warns if `.env` is missing. Keys the agent needs in `.env`: `TABPFN_API_TOKEN`, the LLM settings (`SUPERVISOR_*`, `WORKER_*`, see `agent/llm_config.py`), and for memory `COGNEE_ENABLED`, `COGNEE_API_BASE_URL`, `COGNEE_API_KEY`. The first question after a cold start waits ~30 s for the TabPFN model; later ones take 2–5 s.

## Talking to it

* **Chat:** open the ADK UI, pick `agent`, ask e.g. *"There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00. What does the flow look like at Hermannplatz, and what should we do at 23:15 when it ends?"* — then *"What about if it ends at 22:30?"* (follow-up) and *"Why do you say that?"* (explanation).
* **Terminal:** `make agent-query Q="..."` (one question) · `make agent-cli` (chat) · `make demo` (scripted 6-turn conversation, small models).
* **MCP client:** connect to the knowledge server URL above, e.g. `sanity_check`, `kg_similar`, `kb_boundaries`.

## Everything else, in the order you will need it

| Goal | Command |
| --- | --- |
| Check the code without any LLM or network | `make check` (unit tests + guardrail suite + router accuracy) |
| Score answers (small model) | `make eval ARGS="--suite challenge --ids CH1,CH3 --cheap"` · `make ls-eval` |
| See every resource and whether it works (Cognee, KB, graph, Neo4j, MCP tools, models, keys) | `make resources` · dashboard page **Resources** |
| Load the graph into Neo4j / bring it level | `make neo4j-up` (first time) · `make neo4j-sync` |
| Load the 3 ADK eval cases (approximate answers) | `make adk-evalset` → ADK UI → Evals → `eval_set_1` |
| Rebuild knowledge after the Sept 22–30 data arrives | `make kb-build && make kb-sync && make kg-seed` |
| Back up the trace database before big runs | `make backup-obs` |
| All commands with descriptions | `make help` |

Architecture, schemas and guardrails: [`agent_architecture_v3.md`](agent_architecture_v3.md).

## Troubleshooting

* *A service shows `starting` for a long time* — `make logs S=<name>`; the dashboard and MCP servers load pandas / the data on first start.
* *Port already in use* — `make up` picks the next free port and prints it; `make status` shows the real one.
* *Stale state after a crash* — `make down` (safe to run twice), then `make up`.
* *Answers say "outside the data"* — the agent only knows 2026-06-10 to 2026-09-22 until the new dataset is added under `data/` (files are merged automatically).
