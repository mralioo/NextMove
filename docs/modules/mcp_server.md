# `mcp_server/` — the three MCP servers

**Purpose.** Everything the agents know about the data is reached through MCP tools (FastMCP 4.x). Three servers with different jobs, so the data server stays warm and the knowledge / quality servers start in ~1 s. Tool catalogue (generated): [`../generated/mcp_tools.md`](../generated/mcp_tools.md). How they are called: [`../mcp_and_tools.md`](../mcp_and_tools.md).

| File | Server (FastMCP name) | Tools | Transport |
| --- | --- | --- | --- |
| `server.py` | data + TabPFN (`ubahn-flow-data`): dataset, station and prediction tools; it also registers the tools of the two files below | 17 in total | stdio (spawned by `agent/mcp_runtime.py`); HTTP `:8765/mcp` with `MCP_TRANSPORT=http` |
| `disruption_tools.py` | Category C solver: `resolve_closure` → `apply_closure` → `alternate_paths` → `scenario_flow` | 4 | registered on the data server |
| `analytics_tools.py` | Categories A, P, B, E, F, G, H: `event_impact`, `rank_pressure`, `find_anomalies`, `energy_efficiency`, `network_resilience_ranking`, `correlated_stations`, `reroute_behaviour` | 7 | registered on the data server |
| `knowledge_server.py` | knowledge + memory + graph + operator knowledge base (`nextmove-knowledge`) | 18 | stdio or HTTP `:8766/mcp` |
| `quality_server.py` | quality database + golden data (`nextmove-quality`) | 22 (`quality_*` ×12, `golden_*` ×10) | stdio, HTTP `:8768/mcp`, in-memory (Inspector) |

## Design rules

* **Read-only and deterministic**, except the TabPFN prediction tools (call the TabPFN API; need `TABPFN_API_TOKEN`) and the two write tools of the knowledge server (`add_insight`, `kg_add_case`).
* The data tools reuse `dashboard/utils/data_loader.py` — the same code path as the Streamlit dashboard — so agent and dashboard cannot silently disagree; the merged training + testing files are what they read.
* Tools return JSON (dicts / lists) with the numbers **and** the assumptions and boundaries that apply; nothing is prose the model could misquote.
* Station names are resolved fuzzily (`resolve_station`, `difflib`), but every other tool takes the exact data name.
* Startup: the data server warms its caches in a background thread (feature table parquet, disruption model checkpoint, prediction cache).

## Run alone

```bash
./.venv/bin/python scripts/tasks.py mcp-server      # data + analytics + TabPFN, stdio
./.venv/bin/python scripts/tasks.py mcp-knowledge   # knowledge / sanity / graph, stdio
./.venv/bin/python scripts/tasks.py mcp-quality     # quality + golden, stdio
MCP_TRANSPORT=http MCP_PORT=8768 ./.venv/bin/python mcp_server/quality_server.py    # HTTP
```

## Configuration

`MCP_TRANSPORT` (stdio | http), `MCP_PORT`, `MCP_DATA_PORT`, `MCP_KNOWLEDGE_PORT` (8766), `MCP_QUALITY_PORT` (8768), `DATA_DIR`, `ML_CACHE_DIR`, `CHECKPOINT_DIR`, `SCENARIO_ENGINE` (tabpfn | empirical), `TABPFN_API_TOKEN`, `COGNEE_*` (knowledge server memory tools).

## Tests

`tests/test_disruption.py` (graph logic, no TabPFN), `tests/test_quality_pipeline.py` (quality database and MCP tools), `tests/test_workflow_v2.py` (analytics tools via the playbooks).

## Limits

TabPFN tools depend on an external API (latency, token, network); `find_anomalies`, `event_impact`, `correlated_stations` and `reroute_behaviour` describe statistical association in simulated flows — they do not prove causes; there is no capacity, delay or cost data behind any tool.
