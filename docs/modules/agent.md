# `agent/` — the agent team

**Purpose.** Turns an operator's question into a verified, plain-language answer. Pitch names ↔ code names (see [`../agent_architecture_v3.md`](../agent_architecture_v3.md)): **Dispatcher** = supervisor, **Analyst** = worker, **Inspector** = evaluator, **Writer**.

**Entry points.** `agent/agent.py` exports the ADK `app` (`adk web agent/`, `adk run agent/`, `run_query.py`, the dashboard and the evaluation all load it). It is a `SequentialAgent` built by `fast_agent.build_fast_agent()`: `dispatcher` → `analyst` → `writer`, plus the observability plugin.

## Pipeline

```
question → Dispatcher (guardrails · router · history / follow-up · objective · route)
        → Analyst ⇄ Inspector  (playbook of MCP calls → facts + confidence → checks → accept | revise | reject; ≤ 2 rounds, 40 s)
        → Writer (one LLM call → brief · number guard · references · artifact) → answer
```

## Files

| File | Responsibility |
| --- | --- |
| `agent.py` | ADK entry point (`app`), model wiring per role |
| `fast_agent.py` | The three ADK agents (`SupervisorAgent`, `WorkerAgent`, `WriterAgent`), event emission (`customMetadata.kind`), session-state hand-over, storing the turn, artifact, graph case |
| `supervisor.py` | **Dispatcher**: `question → SupervisorPlan` (decision, category, entities, objective, route, guardrails, history hit, follow-up, knowledge-base boundaries) |
| `router.py` | Deterministic classifier + entity extraction (lines, exact station names, dates, times, durations, what-if); small-LLM JSON fallback only when unsure (`ROUTER_CONF_MIN`) |
| `guardrails.py` | Input (length, prompt injection, scope), process and output guardrails; `LIMITS` (env-overridable) |
| `threads.py` | One conversation = one situation: `relation(question, anchor)` → related / unclear / unrelated |
| `detail_ask.py` | Which messages ask for the full picture (why / evidence / tools / report) |
| `specialists.py` | `SPECIALISTS`: the playbook per category (A–H, P, X) and the single source of truth for capabilities |
| `executor.py` | The playbooks: MCP calls (parallel where independent), compression to the compact **facts** JSON |
| `worker.py` | **Analyst**: `WorkerTask → WorkerResult` with a documented deterministic confidence score |
| `evaluator.py` | **Inspector**: deterministic checks (objective, ground truth, boundaries, station names, dates), quality-database check via MCP, LLM check, verdict + validated adjustments |
| `loop.py` | The bounded Analyst ⇄ Inspector loop |
| `writer.py` | **Writer** + guard: brief by default (`TMT_ANSWER_MODE`), full report on request, number guard, template fallback, references |
| `artifacts.py` | The artifact bundle of an answer and `full_report()` |
| `schemas.py` | 21 pydantic hand-over models (`extra="forbid"`, versioned) |
| `config.py`, `llm_config.py` | Run configuration (`TMT_CONFIG`) and per-role LiteLLM settings (`SUPERVISOR_*`, `WORKER_*`, `ROUTER_*`, `WRITER_*`, `EVALUATOR_*`) |
| `mcp_runtime.py` | Persistent, pre-warmed stdio connection to the data MCP server; parallel calls |
| `quality_mcp.py` | In-memory MCP client to the quality server for the Inspector |
| `knowledge.py` | Knowledge base (`knowledge.json`), sanity checker, turn history, operator feedback tables, artifact store, Cognee client (optional, off the hot path) |
| `knowledge_build.py` | Builds the knowledge base from the raw CSVs |
| `kgraph.py`, `kgraph_build.py` | Local knowledge graph (SQLite + NetworkX, Neo4j mirror), seeding and export |
| `feedback.py` | Operator feedback loop: validation, storage, graph feeding, precedents, recommended actions |
| `observability.py` | OpenTelemetry spans and the `runs` row per question in `observability/agent_obs.db` |
| `resources.py` | Inventory / health of every resource (used by the dashboard and `tasks.py resources`) |
| `data_window.py` | The data window read from the files |
| `run_query.py`, `eval_router.py`, `eval_set_1.evalset.json` | CLI question runner, router accuracy check, ADK eval set |

## Key contracts

* **Dispatcher decisions:** `proceed`, `answer_from_history` (accepted identical answer ≤ 24 h, same data window, not rated 1–2), `follow_up` (`explain` from earlier facts, `rerun` with the operator's change), `decline` (related but unsupported), `need_input` (missing station / date), `bounce` (off-topic / manipulation / greeting).
* **Confidence** is a documented deterministic estimate (start 0.90; deductions for assumptions, proxies, failed checks), never a model probability.
* **Ground truth outranks the LLM:** a failed station / truth check cannot be accepted; an LLM "reject" cannot overrule passed checks.
* **Guard:** every number in the brief must occur in the facts; otherwise a template rendering of the same facts is shown.
* **Advisory only:** no capacity, bus-service or measured-pressure claims; the operator decides.

## Configuration (environment)

Models: `SUPERVISOR_LITELLM_MODEL` (+ `_API_BASE`, `_API_KEY`, `_API_VERSION`) — also the Inspector's model unless `EVALUATOR_LITELLM_MODEL`; `WORKER_*` (router fallback, small tasks); `ROUTER_*`; `WRITER_*` (falls back to SUPERVISOR); `WRITER_REASONING_EFFORT`, `WRITER_TIMEOUT_S`, `WRITER_BRIEF_MAX_WORDS`. Behaviour: `TMT_CONFIG` (JSON: router, memory, mcp, engine, writer), `TMT_ANSWER_MODE` (`brief` | `detail`), `TMT_HISTORY` (`off` = evaluation runs neither reuse history nor train the graph), `EVALUATOR_MODE`, `EVALUATOR_CONF_BELOW`, `LOOP_MAX_ITERS`, `LOOP_DEADLINE_S`, `GUARD_MAX_QUESTION_CHARS`, `GUARD_MAX_ANSWER_WORDS`, `GUARD_MIN_CONFIDENCE`, `WARM_ON_START`, `QUALITY_MCP`, `COGNEE_*`, `NEO4J_*`, `TMT_MEMORY_DB`, `TMT_KG_DB`, `OBS_DB`.

## Tests

`tests/test_agent_schema_v3.py` (schemas, guardrails, supervisor decisions, worker confidence, evaluator loop) · `test_fast_pipeline.py` (router, number guard) · `test_workflow_v2.py` (categories, closure matching) · `test_brief_and_artifacts.py` · `test_operator_feedback.py` · `test_trace_payloads.py` · `test_eval_and_observability.py`. All offline; `evaluation/guardrail_suite.py` (46 decisions) and `agent/eval_router.py` (71/75 on the question bank) need no LLM either.

## Limits

Category X (strategy / investment) is planned, not built and is declined honestly; A, P, B, G, H are `partial` (answer with stated proxies); the "Cognee history session" is served by the local turn store (Cognee recall is 4–12 s and is not on the hot path); the router is rule-based with an LLM fallback, so unusual phrasing can route to `need_input` or `decline`.
