# `dashboard/` — Streamlit developer / analyst dashboard

**Purpose.** The analyst's and developer's view of the data, the ML engine, the agent workflow and its evaluation. Not the operator screen (that is `frontend/`). Runs on http://localhost:8501 (`make up`, service `dashboard`; `PORT`), Docker image via `dashboard/Dockerfile`.

| Page (`views/`) | What it shows |
| --- | --- |
| Overview (`home.py`) | dataset summary, KPI cards, navigation |
| 1 Network Explorer | stations and connections, centrality |
| 2 Passenger Flow | flows per station / line / time |
| 3 Events | events and their effect on nearby stations |
| 4 Weather | weather vs flow |
| 5 Closures | recorded closures |
| 6 Energy | energy per line, per passenger |
| 7 ML Engine (TabPFN) | model results, case study |
| 8 Observability | runs, spans (traces), tokens, latencies from `agent_obs.db` |
| 9 Evaluation | evaluation runs, metrics, judge scores |
| 10 Experiments | component study (router / memory / MCP transport / engine / writer) |
| 11 Agent Workflow | the diagram, supervisor tester, guardrails, schemas, loop traces, knowledge graph |
| 12 Resources | every resource (Cognee, knowledge base, graph, Neo4j, MCP tools, models, keys), health and links |
| 13 Submission runs | organiser-workbook runs and their comparison |

`app.py` is the router (`st.navigation`); `utils/` = `data_loader.py` (the same loader the MCP data server uses), `obs_data.py`, `ml_results.py`, `submissions.py`, `ui.py`.

## Notes

The data loader is shared with the MCP data server on purpose: dashboard and agent read identical numbers. The Streamlit runtime is also imported (cache decorators) when backend / doc scripts import the loader — the `No runtime found` warnings on the console are harmless.

## Limits

Read-mostly, single user, no authentication; heavier pages (Experiments, Workflow) read the observability database and may be slow on a large one.
