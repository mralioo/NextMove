# Module reference

One page per top-level module: what it is for, the files in it and what each does, its entry points, inputs / outputs, configuration, tests, and its limits. System-level views (design, user flow, data flow, events, schema, tools) are in the [root README](../../README.md) and [`../events_and_data_flow.md`](../events_and_data_flow.md), [`../data_schema.md`](../data_schema.md), [`../mcp_and_tools.md`](../mcp_and_tools.md), [`../api_reference.md`](../api_reference.md).

| Module | Folder | One line | Page |
| --- | --- | --- | --- |
| Agents | `agent/` | Dispatcher → Analyst ⇄ Inspector → Writer as an ADK app, guardrails, knowledge base, knowledge graph, feedback loop, observability | [agent.md](agent.md) |
| MCP servers | `mcp_server/` | The three FastMCP servers and their tools | [mcp_server.md](mcp_server.md) |
| ML | `ml/` | TabPFN models, disruption graph logic, normalization pipeline, quality database, golden data | [ml.md](ml.md) |
| Backend | `backend/` | Operator API (FastAPI), chat bridge, analytics and replay data | [backend.md](backend.md) |
| Frontend | `frontend/` (+ archived `frontend_old/`) | React operator UI: desk, Chat Copilot, Toby, analytics tabs | [frontend.md](frontend.md) |
| Dashboard | `dashboard/` | Streamlit developer / analyst dashboard (data pages, ML, observability, evaluation, workflow, resources) | [dashboard.md](dashboard.md) |
| Evaluation | `evaluation/`, `experiments/` | Ground truth, LLM judge, metrics, organiser workbook runs, experiments | [evaluation.md](evaluation.md) |
| Scripts, ops, tests | `scripts/`, `Makefile`, `tests/` | Service supervisor, task runner, doc generator, test suite | [scripts_and_ops.md](scripts_and_ops.md) |
| Data | `data/`, `knowledge/`, `observability/` | Raw, golden and derived data; knowledge base; stores | [data.md](data.md) |
