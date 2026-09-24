# `evaluation/` and `experiments/` — measuring the system

**Purpose.** Score the agent honestly: ground truth from the raw files (never from the agent's tools), deterministic gates plus an LLM judge, the organiser's answer workbook, and small component experiments. Narrative: [`../observability_and_evaluation.md`](../observability_and_evaluation.md), [`../submission_run_report.md`](../submission_run_report.md), [`../experiments_plan.md`](../experiments_plan.md), [`../test_questions.md`](../test_questions.md).

| File | Responsibility |
| --- | --- |
| `dataset.py` | The evaluation dataset = the organiser's workbook (`team_answers_template v2.xlsx`): TRAINING (T01–T11), FINAL_TEST (F01–F05), TEAM_EVIDENCE; challenge questions |
| `ground_truth.py` | Answers for the checkable questions computed from the raw CSVs with plain pandas |
| `metrics.py` | Metrics mapped to the organiser's scoring criteria (relevance, accuracy, …): answered, completeness, number checks, latency |
| `judge.py`, `rejudge.py` | LLM-as-judge (question, rubric, ground truth, evidence) and re-scoring of stored runs |
| `run_eval.py` | Run the workflow over the dataset and score it (default: one brutal multi-part question; more only with `--allow-many`) |
| `guardrail_suite.py` | 46 labelled scope / guardrail / routing / follow-up decisions — no LLM, ~1 s |
| `conversation_demo.py` | Scripted 6-turn conversation (bounce, follow-up, explanation, history hit, decline) |
| `submission_run.py`, `submission_db.py`, `team_evidence.py` | Answer the organiser workbook with the current configuration, store the run with its full configuration (`observability/submissions.db`, `evaluation/submissions/<run>.json`); write TEAM_EVIDENCE from measured numbers; fills `evaluation/Nextmove_team_(A1).xlsx` (the filled organiser workbook) |
| `validate_pressure.py` | Skill of the pressure ranking on replay days (`knowledge/validation.json`) |
| `langsmith_eval.py`, `langsmith_run.py` | LangSmith-style offline evaluation and optional live experiments (needs `LANGSMITH_API_KEY`) |
| `make_adk_evalset.py` | Writes the ADK UI eval set `eval_set_1` |
| `experiments/` | `arms.py`, `questions.py`, `run_arm.py`, `run_experiments.py`, `scoring.py`, `report.py`, `rescore.py`: component study, 2 questions × 10 arms |

## Commands

```bash
make check                                                      # tests + guardrail suite + router accuracy, no LLM
./.venv/bin/python scripts/tasks.py eval --suite challenge --ids CH1,CH3 --cheap
./.venv/bin/python scripts/tasks.py submission-run --label baseline [--cheap] [--env K=V] [--config JSON]
./.venv/bin/python scripts/tasks.py ls-eval                     # offline judge over the latest stored runs
```

## Principles

Evaluation runs use `TMT_HISTORY=off` (they neither reuse history nor train the graph), keep the number of LLM calls small by default (one question), and never upload anything to a third party unless asked (`--upload`, `LANGSMITH_API_KEY`). A decline or "not answerable from this data" is scored as honest, not as a failure.

## Limits

The judge is an LLM (rubrics reduce, not remove, its noise); ground truth exists only for the checkable questions; challenge questions are few; the flows are simulated.
