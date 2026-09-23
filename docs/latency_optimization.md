# Latency optimisation — from ~60 s to ~4 s per answer

The challenge has a response-time limit; the original agent needed **59 s** for the training Q3
(disruption) question. This document records where the time went, what was changed, and what it costs.
Reproduce every number with `make bench`, `make eval-router`, `make test`.

## 1. Where the 59 s went (measured, per ADK event)

| Step (old design) | Time |
| --- | ---: |
| Supervisor LLM decides which specialist to call | 4.8 s |
| Specialist LLM #1 → `resolve_closure` (plus spawning a *new* MCP server process) | 4.2 s |
| Specialist LLM #2 → picks the next three tools | 1.5 s |
| **Tools: cold start** (feature table rebuilt: 12 s; model restore + probe: 3 s; scenario = 2 TabPFN API calls: 9 s) | **22.2 s** |
| Specialist LLM writes a 3 000-character draft from **9 k tokens** of raw tool JSON | 11.8 s |
| Supervisor **re-types the whole draft** into the verifier | 6.1 s |
| Verifier LLM | 1.4 s |
| Supervisor rewrites the answer again | 7.1 s |
| **Total** | **59.4 s** (37 s LLM, 22 s tools) |

Two independent problems: a **cold ML/tool stack rebuilt on every question**, and **six sequential LLM
calls that pass long prose to each other**.

## 2. What changed

### 2.1 Architecture: route → execute → write (`agent/`)

```
question ─► ROUTER ──plan JSON──► EXECUTOR ──facts JSON──► WRITER + GUARD ─► operator brief
            ~1 ms                  parallel MCP calls        1 short LLM call     numbers verified
            (LLM only if unsure)   no LLM                    deterministic guard
```

| Your suggestion | What was built |
| --- | --- |
| **Optimise each agent's prompt** | Writer prompt is static and ~700 tokens (cacheable); router LLM prompt ~250 tokens; the executor has **no** LLM. The 9 k-token specialist read of raw JSON became ~1 k tokens of facts. |
| **Supervisor outputs JSON naming the datasets/tools, forwarded on** | The router emits a plan `{"cat":"C","data":["closures","connections",…],"tools":["resolve_closure","apply_closure","alternate_paths","scenario_flow"],"lines":["U6"],"stations":[exact names],"dates":[…],…}`. The executor runs the playbook for that category straight from the plan (ML engine = the `scenario_flow` / predict tools). Independent tool calls run in parallel. |
| **Symbolic hand-off instead of long text** | Stages exchange JSON in session state (`plan`, `facts`, `last_facts`) with short keys, rounded numbers, stripped station names and assumption *codes* (A1–A5, decoded once in the writer's system prompt). |
| **Last agent summarises in simple, fact-driven words** | The **writer** turns facts into ≤110 words: *Answer / Key facts / Do now / Caveat*. It is the only generative step on the hot path. |
| **"JEV model" for routing** | See §5 — not adopted (unverifiable). The router is a deterministic classifier instead (faster than any model call). |

The old LLM supervisor → specialists → verifier loop is kept behind `AGENT_MODE=llm` for open-ended
questions the playbooks don't cover.

### 2.2 The verifier became a guard (milliseconds instead of two LLM round trips)

`agent/writer.py` checks the writer's text deterministically: every number ≥ 13 (or non-integer) must occur
in the facts, in the operator's own question, or in a small fixed allowance (25/50/75 % diversion shares,
26 recorded closures, the 15-minute grain). It also bans claims the data can't support ("exceeding its
capacity", "bus services are available"). On any violation the answer is replaced by a template rendering of
the same facts, so it is always grounded — and the executor precomputes derived numbers (e.g. prediction
error) so the writer never has to do arithmetic.

### 2.3 ML-engine and MCP speed-ups (benefit both modes)

| Change | Effect |
| --- | --- |
| Feature table cached as parquet (`ml/table_cache.py`, fingerprinted on the data files) | 12 s → ~0.1 s |
| Prepared baseline table cached (`DemandBaseline.prepare`) | 2.2 s → ~0.3 s |
| Checkpoint restored **without** a verification probe; lazy refit if the server forgot the fit | −2.8 s |
| **One** dense-quantile TabPFN call instead of two (quantiles + mean); the mean is integrated from the quantile function (within 1.2 % of TabPFN's own mean, r = 0.9999; the 13 quantiles are bit-identical) | 2 API calls → 1 |
| Prediction cache per (station, timestamp), on disk; `make checkpoints` pre-warms all 26 dataset closures | cached closure: `scenario_flow` 9 s → **0.3 s**; a new what-if costs one 2–3 s API call |
| One persistent, pre-warmed MCP server per agent process (background thread) instead of a new process per specialist | no per-question spawn; warm-up overlaps start-up |
| Server warms itself in threads at start-up (Category C model first) with separate locks | first tool call no longer blocks behind unrelated loading |

## 3. Results

Warm process (what `adk web` / a served agent looks like), `make bench`:

| Question | Old LLM loop¹ | **Fast pipeline** |
| --- | ---: | ---: |
| C — training Q3 (closure → reroute → pressure → staff) | 59 s | **6.4 s** |
| C — follow-up ("how confident…") | – | 4.1 s |
| C — another closure | – | 5.0 s |
| C — what-if (not in `closures.csv`, one TabPFN call) | – | 5.0 s |
| D — training Q4 (Rudow peak) | 25 s | **2.7 s** |
| D — prediction at a timestamp | – | 3.8 s |
| A — unsupported (honest decline) | – | 1.9 s |
| Out of scope (capacity) | – | 1.8 s |
| **Mean over 8** | | **3.8 s** |

¹ Q3: the original 59 s measurement, before any speed-up. Q4: measured later, with the ML/MCP speed-ups already applied (so the old loop's Q3 is 37 s on today's tool stack).

Cold start of a fresh process (`make agent-query`): **16.7 s** for Q3 (7.8 s of it waiting for the server
warm-up, paid once per process). The old loop with only the ML/MCP speed-ups: **37 s**.

Router: **96 % correct (72/75)** on the question bank (`docs/test_questions.md`) at ~1 ms per question;
the misses are zero-confidence cases that fall through to the LLM router or need conversation history.
The rest of the time is the writer LLM (`azure/gpt-5.6-luna`, `reasoning_effort=low`: 2–8 s).

## 4. Safeguards that keep speed from costing correctness

* **Hard time budget:** the writer has `WRITER_TIMEOUT_S` (default 10 s); if the LLM is slow or fails, the
  deterministic template answer ships instead — response time has a ceiling.
* **Ambiguity is surfaced, not guessed:** `need` answers ask for the missing input (e.g. several closures
  match; a date outside the data; a second station). "In December" answers with the recorded July closure
  flagged as such rather than inventing a December scenario.
* **Station-closure caveat preserved:** "groups cut off" is reported only as
  `isolated_only_if_no_through_trains` (the tool assumes trains run through) so the writer must qualify it.
* **Typo tolerance:** "Rudov" → Rudow, "Alexanderplaz" → Alexanderplatz; a non-station ("Kreuzberg") is not guessed.
* **Follow-ups** reuse the previous facts from session state — no tools, one short LLM call.

## 5. The "JEV" model

The linked Hugging Face blog post describes a "Jev AI" service but has no model card, no weights and no
repository, and points to a third-party API domain (`thejevai.com`). I could not verify that it is a real,
usable model, and sending operator questions to an unverified external endpoint is a risk we don't need — so
it was **not** integrated. It also isn't needed: routing here is a ~1 ms rule-based classifier (96 % on the
bank) with a small JSON-mode LLM fallback. If a learned router is wanted later, the cheap, verifiable options
are a TF-IDF + logistic-regression classifier trained on the question bank (local, sub-millisecond) or a
small sentence-embedding model; both would slot in behind `router.route()`.

## 6. Configuration (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_MODE` | `fast` | `llm` = the original supervisor + specialists + verifier loop |
| `WRITER_LITELLM_MODEL` | falls back to `SUPERVISOR_*` then `WORKER_*` | model for the final brief (strong instruction-following matters most here) |
| `WRITER_REASONING_EFFORT` | `low` for `gpt-5*` | reasoning models can't take a temperature; "low" is their fastest setting |
| `WRITER_TIMEOUT_S` | `10` | LLM budget before the template answer is used |
| `ROUTER_LITELLM_MODEL` | `WORKER_*` | model for the low-confidence router fallback |
| `ROUTER_CONF_MIN` | `0.55` | below this the LLM router is consulted |
| `WARM_ON_START` | `1` | start the MCP server and open the LLM connection when the agent loads |

## 7. Limits and next steps

* Only categories **C and D** have tools; A, B, E, F, G, H, X return an honest "not supported yet"
  (about 2 s). Adding tools for E and F is cheap (the dashboard already computes both) and would reuse this
  exact executor pattern (a playbook + a facts compactor + a legend line in the writer prompt).
* The writer is the floor: 2 s for short answers, 4–8 s for disruption briefs, with occasional spikes
  (one 13 s call observed before the timeout guard existed). Streaming the answer would improve *perceived*
  latency but not total time.
* Routing is English/German patterns; unusual phrasing falls back to the LLM router (+~2 s).
* Cold single-shot CLI runs pay the server warm-up (~7 s); long-running processes pay it once.
* The prediction cache and parquet caches live in `ml/cache/` (regenerated automatically when the data or the
  model checkpoint changes).
