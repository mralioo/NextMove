# Brief answers by default, full report on request — and the operator knowledge base

**Date:** 2026-09-24 · **Branch:** `agentic-v3-supervisor-evaluator` · **Stored comparison run:** `sub-20260924-162021` (*brief-answers*) vs `sub-20260924-151046` (*main-model*, the long format) in the dashboard page *Submission runs → Compare runs*.

## 1. Why

A control-room operator under stress does not read a 150-word answer with five evidence bullets, a caveat and a sources line. They need: *what happened / what to do / what to watch out for / how sure*. Everything else is valuable — but only when asked for. So the writer now has two modes:

| | **Brief (default)** | **Full report (on request)** |
| --- | --- | --- |
| When | every question | the operator asks *why*, *evidence*, *proof*, *sources*, *which tools / functions did you call*, *how did you calculate / decide*, *show me the details / the steps*, *full report*, *in detail* … (`agent/detail_ask.py`) |
| Content | **Verdict** (1 sentence, key number) · **Do now** (max 3 short bullets, only if the question asks what to do) · **Watch out** (the one assumption that matters) · one line: *Confidence high/medium/low (95 %) — ask "why", "evidence", "sources" or "which tools" for the full report* | Verdict · Evidence (each number with its source) · Do now · Caveat · Argument · **How this was worked out** (tools with arguments and times, data and model, evaluator checks ✓/✗, knowledge-base entries, confidence reasons, timing, LLM calls/tokens) · Sources |
| Size | ≈ 45–90 words | 250–550 words |

**Before** (U7 closure, one of the stored answers): verdict + 4 evidence bullets with sources + actions + caveat + a two-line sources list, ≈ 155 words.
**Now:**

```
**Verdict:** Track maintenance closes U7 between Hermannplatz and Karl-Marx-Strasse from 20:45 to 22:45.

**Do now:**
- Replacement bus between Hermannstr. and Neukölln (about 0.85 km).
- Replacement bus between Leinestr. and Karl-Marx-Str. (about 0.87 km).
- Deploy staff at Neukölln and Hermannplatz; monitor Schönleinstr.

**Watch out:** Pressure is a scenario: diversion share and station-history baseline are assumed; no capacity data.
_Confidence high (95%). Ask “why”, “evidence”, “sources” or “which tools” for the full report._
```

Then *"why?"* or *"which tools did you call?"* → the full report, **built from the stored record of that answer, nothing recomputed**. A second request returns the same stored report instantly (0.0 s, no LLM).

## 2. What changed in the system

* **Writer (`agent/writer.py`)** — two system prompts sharing the same rules (`WRITER_SYSTEM_BRIEF` = short head + rules, `WRITER_SYSTEM` = long head + rules); `write(..., mode="brief"|"detail")`; `WANTS_ACTION` / `DETAIL` flags in the prompt (a question that does not ask for actions gets no "Do now"); deterministic `shorten()` (verdict ≤ 30 words, ≤ 3 bullets of ≤ 14 words, one watch-out of ≤ 22 words) used when the model writes too much *and* as the brief version of the template fallback; `confidence_footer()`; the brief has its own length guard (cap 95 words, detail cap 230); the number guard, banned-claim guard and disclosure of silent assumptions apply to both modes. Reasoning models spend part of `max_tokens` on hidden reasoning, so the cap for the brief is 1 000 (a cap of 400 returned empty text on the main model and silently fell back to the template — found in the first comparison run and fixed).
* **Router / supervisor** — a message asking for evidence / sources / tools / a full report after an answer is a *follow-up about the last answer* (never a re-run); `WHY_FOLLOW` now includes those requests (`detail_ask.py` is the single source of the phrases).
* **Switches** — `TMT_ANSWER_MODE=detail` makes every answer the full report (for evaluation runs that want long answers); `WRITER_BRIEF_MAX_WORDS`. The mode is recorded in each submission run's design (`switches.answer_mode`).
* **Artifacts (`agent/artifacts.py`)** — the bundle of one answered turn; `full_report()` = the writer's narrative + a deterministic *How this was worked out* section **built only from the bundle** (so "which tools did you call, with what arguments" can never be invented by the model).

## 3. The artifact bundle (one per answered question)

`artifact_version, turn_id, session_id, created_at, question, category, specialist, decision, source, answer_mode, objective, entities, route, assumptions, confidence, confidence_reasons, verdict{verdict, score, issues, rationale, model, rounds, checks[{id, ok, detail}], ground_truth_ids, boundary_ids, similar_cases}, tools[{tool, server, args, seconds, bytes, ok, round, result_preview}], datasets, ml_engine, llm[{role, model, seconds, tok_in, tok_out}], facts, brief, report, references, sources_line, timing{supervisor_s, worker_evaluator_s, mcp_s, writer_s, total_s}, guard, sanity, data_window, problem_key`. `report` stays empty until someone asks for it, then it is stored back (first request pays one LLM call, later ones none).

## 4. Where the artifacts are saved and indexed (the operator knowledge base)

| Store | What is kept | How to use it |
| --- | --- | --- |
| **Session memory** | `last_artifact` in the ADK session state | "why?" in the same chat |
| **Operator knowledge base** (SQLite `observability/memory.db`, table `turns.artifact_json`) | every bundle (also survives restarts / new sessions: the *latest accepted* bundle is restored for that operator); the full report is attached when first generated | `KnowledgeBase.find_artifacts / get_artifact / last_artifact / attach_report`; dashboard **Agent Workflow → 5c Operator knowledge base** (table, brief, full report or what the report would add) |
| **Knowledge MCP server** (now 15 tools) | `operator_kb_search(query, category, k)` → similar past answers with brief, confidence, tools, datasets, has-report; `operator_kb_get(turn_id)` → the whole bundle | for the evaluator / other agents / any MCP client ("what did we do last time this happened?") |
| **Knowledge graph** (SQLite + NetworkX, mirrored to Neo4j) | `Problem –HAS_ARTIFACT→ Artifact –USED_TOOL→ Tool`, `–USED_DATASET→ Dataset`, `–USED_MODEL→ Model`, `–CITES→ KBEntry`; the Artifact node carries the brief, confidence, verdict, tool list and (once generated) the report | Neo4j browser / graph page; only accepted answers of production runs are added (evaluation runs with `TMT_HISTORY=off` do not train the graph) |
| **Memory agent (Cognee)** | each accepted answer is mirrored with a compact bundle summary (question, decision, confidence, tools, datasets, assumptions, KB entries) as the entry context; a newly generated full report is mirrored as its own entry | recall through the existing Cognee tools; **this is an upload to a third party, as before** (`COGNEE_ENABLED`); nothing new is sent when memory is `session` |

Verified live: after one U8 closure question with history on, the KB held the bundle (`has_report` true after "which tools did you call?"), the graph had `Artifact 1, Tool 4, Dataset 6, Model 1, KBEntry 3` with the matching relations, and the second identical request returned the stored report in 0.0 s.

## 5. Measured effect (same 23 questions, same models, `sub-20260924-162021` vs `sub-20260924-151046`)

| | long format (before) | brief default (now) |
| --- | ---: | ---: |
| Median words of the 17 data answers that stay in brief mode | 144 | **57** (mean 62 vs 137) |
| Mean words over all 23 answers | 124 | 82 |
| Median / p95 latency | 3.7 / 9.3 s | 3.9 / 6.6 s |
| Tokens out / LLM inference | 5 793 / 93.6 s | 5 162 / 94.4 s |
| Number guard, evaluator accepts, MCP calls | 23 / 23 / 37 | 23 / 23 / 37 |

**Honest reading:** the brief is 2.5× shorter to *read*, but it is **not faster to produce** on the main model: the writer's time is dominated by the hidden reasoning of the model and the network round trip, not by the visible words (completion tokens include the reasoning). To cut the *time*, the levers are a lower reasoning effort for the brief, a smaller/faster writer model for routine briefs (the all-small-model run wrote in ≈ 3.4 s median), or streaming the Verdict first — not done. Two organiser questions that literally ask for an explanation (T05 "what factors explain…", T08 "explain the mechanism…") are answered in the long format on purpose (they match the detail request phrases); with `TMT_ANSWER_MODE=brief` semantics they could also be briefed — a product decision for the operator.

## 6. Limits and open points

* A follow-up about an answer given **before** artifacts existed cannot list its tools ("the tool calls … were not recorded; ask again to recompute") — the narrative is still written from its facts.
* The first request for a full report pays one LLM call (2–6 s); later requests are free. A report generated by the small model can still word pressure loosely (the number guard passes, the banned-claim guard blocks "capacity"); the deterministic method section is exact.
* "Do now" is only shown when the question asks for actions; the main model sometimes fills the bullets with generic monitoring advice — worth an expert review of the briefs.
* The confidence line is deterministic (≥ 0.8 high, ≥ 0.55 medium, else low).
* The Neo4j browser and the Agent Workflow page were exercised through code / Streamlit's test harness, not looked at in a browser.

## 7. How to try it

```
make up                                   # ADK chat on http://localhost:8000 (app "agent")
# ask: "Line U7 is suspended between Hermannplatz and Karl-Marx-Strasse on 2026-09-25 from 20:45 for 2 hours. Where should we send staff?"
# then: "why?"  ·  "which tools did you call?"  ·  "show me the evidence"
./.venv/bin/python scripts/tasks.py submission-run --label full --env TMT_ANSWER_MODE=detail   # every answer as a full report
```
Tests: `tests/test_brief_and_artifacts.py` (shorten, detail/action detection, footer, length guard, report-from-bundle, knowledge base round trip, graph links) — 101 tests pass, guardrail suite 46/46, router bank 71/75.
