# Tool calling and MCP

How the agents call tools, which MCP servers exist, which tool belongs to which question category, and how to call them from outside. The per-tool list (names, descriptions, parameters) is **generated** from the live servers: [`generated/mcp_tools.md`](generated/mcp_tools.md) (`make docs`).

## 1. Principles

1. **Every fact comes from a tool, never from the model.** The Analyst (worker) is a function that runs a fixed *playbook* of MCP calls per question category; the only generative step on the hot path is the Writer, and a deterministic guard rejects any number in the brief that is not in the facts.
2. **Three MCP servers, three jobs** — data & forecasts, knowledge & memory, quality & golden data. All are FastMCP servers (`fastmcp` 4.x, MCP over stdio / streamable HTTP / in-memory).
3. **Tools are read-only and deterministic** (except the two TabPFN prediction tools, which call the TabPFN API, and `add_insight` / `kg_add_case`, which write to the knowledge stores).
4. **Every call is observable**: it becomes an ADK event (`mcp_call` / `mcp_result`), a span (`mcp.tool <name>`), a row in the run's tool list and a line in the operator's operations log ([`events_and_data_flow.md`](events_and_data_flow.md)).

## 2. The three servers

| Server (FastMCP name) | Code | Transport / address | Tools | Who calls it |
| --- | --- | --- | --- | --- |
| **data** (`ubahn-flow-data`) | `mcp_server/server.py` + `analytics_tools.py` + `disruption_tools.py` | **stdio**, spawned once by the agent process and kept warm (`agent/mcp_runtime.py`); optional HTTP `:8765/mcp` (`WITH_DATA_MCP=1` / `MCP_TRANSPORT=http`) for outside clients | 17: dataset (`describe_dataset`, `list_stations`, `resolve_station`, `station_profile`), disruption (`resolve_closure`, `apply_closure`, `alternate_paths`, `scenario_flow`), analytics (`event_impact`, `rank_pressure`, `find_anomalies`, `energy_efficiency`, `network_resilience_ranking`, `correlated_stations`, `reroute_behaviour`), TabPFN (`predict_overcrowding_risk`, `predict_expected_flow`) | the **Analyst** (every question) |
| **knowledge** (`nextmove-knowledge`) | `mcp_server/knowledge_server.py` | HTTP `:8766/mcp` (`MCP_KNOWLEDGE_PORT`), or stdio | 18: ground truth & boundaries (`kb_search`, `kb_ground_truth`, `kb_boundaries`, `sanity_check`), memory (`session_history`, `cognee_recall`, `memory_status`, `add_insight`), knowledge graph (`kg_similar`, `kg_neighbors`, `kg_top_actions`, `kg_stats`, `kg_add_case`), operator knowledge base (`operator_kb_search`, `operator_kb_get`, `operator_actions`, `operator_precedents`, `operator_feedback_stats`) | **other** agents and evaluators (LangSmith-style judge, Claude Code, MCP Inspector). The agent itself reads the same stores in-process (`agent/knowledge.py`, `kgraph.py`) — one implementation, two doors. |
| **quality** (`nextmove-quality`) | `mcp_server/quality_server.py` (+ `ml/quality_db.py`, `ml/golden_data.py`) | HTTP `:8768/mcp` (`MCP_QUALITY_PORT`); **in-memory** client inside the agent (`agent/quality_mcp.py`, ms latency, no subprocess) | 22: `quality_*` ×12 (boundaries, normal flow, plausibility checks, episodes, anomalies, data issues …) and `golden_*` ×10 (the pre-processed files themselves) | the **Inspector** (every answer, `quality_check_facts`) and outside clients |

Ports are the defaults; `make up` uses the next free port if one is taken and prints the addresses (`scripts/tasks.py status`). Setting `TMT_CONFIG='{"mcp":"inmemory"}'` or `'{"mcp":"http"}'` changes how the agent reaches the data server (experiment knob, default `stdio`).

## 3. Question category → playbook → tools

`agent/specialists.py` (`SPECIALISTS`) is the single source of truth; the executor, the dashboard's *Agent Workflow* page and the docs read it.

| Cat | Specialist | Status | Tools called (in order; independent calls run in parallel) | ML |
| --- | --- | --- | --- | --- |
| **C** | disruption | live | `resolve_closure` → `apply_closure` ∥ `alternate_paths` → `scenario_flow` (TabPFN demand baseline, 25 / 50 / 75 % diversion) | TabPFN |
| **D** | station | live | `resolve_station` → `station_profile` (+ `predict_expected_flow` ∥ `predict_overcrowding_risk` when a time is asked) | TabPFN |
| **A** | events | partial | `event_impact` (venue → station mapping learned from the flows) | — |
| **P** | pressure | partial | `rank_pressure` (stations most likely above their own busiest-5 % level on a day) | TabPFN |
| **B** | anomaly | partial | `find_anomalies` (explained by closure / event / weather / unexplained) | — |
| **E** | energy | live | `energy_efficiency` | — |
| **F** | resilience | live | `network_resilience_ranking` | — |
| **G** | correlation | partial | `correlated_stations` | — |
| **H** | reroute | partial | `reroute_behaviour` | — |
| **X** | strategy | planned | none — declined honestly with what is possible | — |

Not tools but decisions of the Dispatcher: greeting / off-topic / manipulation → **bounce** (no tool, no LLM); related but unsupported (capacity, delays, costs) → **decline**; a question about the previous answer → **follow-up** (`explain` from the stored facts, or `rerun` with the operator's change); an accepted identical answer from ≤ 24 h → **answer from history** (0 s). See [`agent_architecture_v3.md`](agent_architecture_v3.md).

## 4. How a tool is called (Analyst)

```
executor.playbook_c(plan, mcp, trace)            agent/executor.py, specialists.py
   └─ await mcp.call("apply_closure", closure_id=8)          agent/mcp_runtime.py  (one warm subprocess, own asyncio loop, calls from any loop, several in flight)
        └─ FastMCP client ── stdio ──► mcp_server/server.py  ──► tool function ──► dashboard/utils/data_loader.py  (same code as the Streamlit dashboard)
   ◄─ python object (the tool's JSON)
   └─ compress to "facts": short keys, rounded numbers, station names without "U "/"(Berlin)", assumptions as codes
```

* The data server starts in a background thread at import and warms its caches (feature table, disruption model checkpoint, prediction cache); the first question after a cold start can wait ~30 s for TabPFN, later ones 2–5 s.
* Every call records `{tool, args, server, seconds, bytes, ok, round, result_preview}`; failures become `WorkerResult(status="error")` — a tool error is escalated, never guessed around.
* The **Inspector** is a second tool user: `agent/evaluator.py` → `_quality_checks` → `quality_mcp.call("quality_check_facts", category=…, facts_json=…)` checks every passenger figure of the facts against the station × day-type × hour boundaries; for event answers it also reads `golden_episodes`. If the quality database was never built (`ml/quality_db.py build`) the call returns `None` and the Inspector simply has no quality boundaries — it never blocks an answer. Ground-truth and knowledge-base checks run in-process before that.
* The **Writer** calls no tool; its references (datasets, tools, model, knowledge-base entries) are taken from what was actually used.

## 5. Calling the servers from outside

Streamable HTTP (knowledge, quality; data with `WITH_DATA_MCP=1`):

```bash
npx @modelcontextprotocol/inspector            # connect to http://127.0.0.1:8768/mcp (quality) or :8766/mcp (knowledge)
```

```python
import asyncio, json
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8768/mcp") as c:
        r = await c.call_tool("quality_check_value", {"station": "Kurfürstendamm", "at": "2026-09-24 09:15:00", "value": 2600})
        print(r.data)
asyncio.run(main())
```

stdio (any MCP client, e.g. Claude Code `.mcp.json`):

```json
{ "mcpServers": {
    "nextmove-quality":   { "command": "./.venv/bin/python", "args": ["mcp_server/quality_server.py"] },
    "nextmove-knowledge": { "command": "./.venv/bin/python", "args": ["mcp_server/knowledge_server.py"] },
    "ubahn-flow-data":    { "command": "./.venv/bin/python", "args": ["mcp_server/server.py"] } } }
```

(The parameters of every tool are in [`generated/mcp_tools.md`](generated/mcp_tools.md).)

## 6. Environment that matters for tools

| Variable | Effect |
| --- | --- |
| `TABPFN_API_TOKEN` | required for `predict_*`, `scenario_flow` and `rank_pressure` (TabPFN API) unless `SCENARIO_ENGINE=empirical` (no ML, no token) |
| `SCENARIO_ENGINE` / `TMT_CONFIG.engine` | `tabpfn` (default) or `empirical` |
| `MCP_TRANSPORT`, `MCP_PORT`, `MCP_DATA_PORT`, `MCP_KNOWLEDGE_PORT`, `MCP_QUALITY_PORT` | transport and ports of the servers |
| `QUALITY_MCP` | `off` switches the Inspector's quality check and the `golden_*` reads off (default `on`) |
| `DATA_DIR`, `ML_CACHE_DIR`, `CHECKPOINT_DIR` | dataset folder, parquet cache of the feature table, TabPFN checkpoint folder |
| `COGNEE_ENABLED`, `COGNEE_API_BASE_URL`, `COGNEE_API_KEY` | memory agent (Cognee): knowledge base + feedback are mirrored to it; **off = nothing leaves the machine** (`COGNEE_ENABLED=0` for local tests) |

## 7. What can go wrong

| Symptom | Cause / fix |
| --- | --- |
| first answer takes ~30 s | data server warm-up (TabPFN model, feature table) — the agent starts the server at boot (`WARM_ON_START=0` disables that) |
| `status: error` in the facts | a tool failed; the answer says so instead of guessing (see the operations log: `ok: false`) |
| Inspector reports no quality boundaries | quality database not built: `./.venv/bin/python ml/quality_db.py build --offline` (`make up` does it once) |
| knowledge tools return empty | the knowledge base was not built: `scripts/tasks.py kb-build` (`make up` does it once) |
| `predict_*` fail | missing `TABPFN_API_TOKEN` / no network |
