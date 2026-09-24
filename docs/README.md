# Documentation index

| Document | What it is | Status |
| --- | --- | --- |
| [`../README.md`](../README.md) | **Start here** — system design, user flow, data flow, events, data schema, tool calling and MCP, API summary, repository map, limits | current |
| [`modules/`](modules/README.md) | One page per module: `agent`, `mcp_server`, `ml`, `backend`, `frontend`, `dashboard`, `evaluation`, `scripts_and_ops`, `data` | current |
| [`events_and_data_flow.md`](events_and_data_flow.md) | Operator user flow, system context and data flow, sequence of a chat turn, feedback loop, event catalogue (ADK `customMetadata.kind`, SSE steps), spans, what is stored when | current |
| [`data_schema.md`](data_schema.md) | Every data shape: raw files, golden data, knowledge base, SQLite stores, artifact bundle, agent messages, knowledge graph, UI state | current |
| [`mcp_and_tools.md`](mcp_and_tools.md) | Tool calling: the three MCP servers, category → playbook → tools, how a call travels, calling from outside | current |
| [`api_reference.md`](api_reference.md) | Operator API and the ADK endpoints it uses: groups, examples, the SSE streaming format, errors | current |
| [`generated/`](generated/mcp_tools.md) | **Generated** (`make docs`): MCP tool catalogue, operator-API endpoint table, `openapi.json` | current |
| [`frontend_migration.md`](frontend_migration.md) | Migration to the new UI: what came from where, kept functions, new streaming, API mapping, removed placeholder data, limits | current |
| [`../data/data_schema_high_quality.md`](../data/data_schema_high_quality.md) | Schema of the golden pre-processed data (`data/normalized`, `data/processed`, derived `data/quality`) and the MCP tools that read it | current |
| [`ml_preprocessing_pipeline.md`](ml_preprocessing_pipeline.md) | Normalization pipeline (normal flow, weather, rest) integrated in `ml/`, the quality database with boundaries, the quality MCP server and how the Inspector uses it | current |
| [`conversation_threads_and_graph.md`](conversation_threads_and_graph.md) | One conversation = one situation: topic-switch choice, chat history, resumed conversations; categorised knowledge graph (situations, domains, action types), audit / repair | current |
| [`operator_desktop.md`](operator_desktop.md) | Behaviour of the operator desk, Toby, operations column and feedback controls (design of the first desktop; the UI was migrated, see `frontend_migration.md`) | current for behaviour; file names in §1–2 refer to `frontend_old/` |
| [`operator_feedback_api.md`](operator_feedback_api.md) | Operator feedback loop (score of a response, action taken → knowledge graph, precedents) and the endpoints for the UI | current |
| [`brief_answers_and_operator_kb.md`](brief_answers_and_operator_kb.md) | Brief answers by default, full report on request (why / evidence / sources / tools); artifact bundles saved to the operator knowledge base, graph and memory | current |
| [`presentation_guide.md`](presentation_guide.md) | Talk for the fair: story, slides, demo script, Q&A, plus the full report (specs, stack, design, numbers, limits) | current |
| [`running_the_system.md`](running_the_system.md) | `make up` / `down` / `status`, what starts, ports, troubleshooting | current |
| [`agent_architecture_v3.md`](agent_architecture_v3.md) | Supervisor / worker ⇄ evaluator / writer, unified schema, guardrails, knowledge graph (SQLite + Neo4j), Cognee | **current design** |
| [`submission_run_report.md`](submission_run_report.md) | The organiser workbook (TRAINING, FINAL_TEST, TEAM_EVIDENCE) answered by the workflow: answers, tokens, times, configuration, run comparison | current |
| [`workflow_v2_report.md`](workflow_v2_report.md) | Specialists for every category, knowledge base, Cognee, LangSmith-style evaluation, pressure-ranking validation | current results |
| [`challenge_alignment_review.md`](challenge_alignment_review.md) | Critical review against the problem statement (before v2) | reference |
| [`test_questions.md`](test_questions.md) | Question bank, grading checklist, **ADK eval set** with approximate answers | current |
| [`observability_and_evaluation.md`](observability_and_evaluation.md) | Traces, LLM-judge scoring, metrics | current |
| [`experiments_plan.md`](experiments_plan.md) | First component study (router / memory / MCP transport / engine / writer) | results of the earlier pipeline; arms R2 (TF-IDF), R3 (JEV) and M2 (episodic memory) no longer exist in the code |
| [`latency_optimization.md`](latency_optimization.md) | How ~60 s became ~4 s | still valid for the writer / symbolic hand-off |
| [`disruption_case_study.md`](disruption_case_study.md) | Category C ML case study (TabPFN vs baseline) | current |
| [`agentic_system_design.md`](agentic_system_design.md), [`Talk_To_My_Train_Engineering_Blueprint.md`](Talk_To_My_Train_Engineering_Blueprint.md) | Early analysis and blueprint | historical |
| [`agents.md`](agents.md), [`system_design.md`](system_design.md), [`../SESSION_SUMMARY.md`](../SESSION_SUMMARY.md) | Description of the first LLM-supervisor design | **superseded** |
