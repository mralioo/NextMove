# Documentation index

| Document | What it is | Status |
| --- | --- | --- |
| [`conversation_threads_and_graph.md`](conversation_threads_and_graph.md) | One conversation = one situation: topic-switch choice, chat history, resumed conversations; categorised knowledge graph (situations, domains, action types), audit / repair | current |
| [`operator_desktop.md`](operator_desktop.md) | React operator desktop: city map with replay, Toby the floating assistant, operations column, feedback controls; the endpoints it added | current |
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
