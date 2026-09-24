# Challenge alignment review — Talk To My Train vs. the InnoTrans 2026 problem statement

**Date:** 2026-09-24 (day before the final evaluation) · **Reviewed:** branch `agentic-workflow-v1` @ `1b4d65f12` + the small uncommitted changes listed in §9
**Reference:** [`innotrans2026_hackathon_problem_statement.md`](innotrans2026_hackathon_problem_statement.md) — the three questions in "The Challenge", the "What you are building" conditions and the five evaluation criteria.
**Configuration under test ("best so far", from [`experiments_plan.md`](experiments_plan.md)):** router `rules` · memory `session` · MCP `stdio` · engine `tabpfn` · writer = small model (`gpt-4o-mini`); judge = small model. The main model (Azure `gpt-5.6-luna`) was **not** used: the experiments found it +2.6 s slower with no measurable quality gain, and it is a shared endpoint.

> **Bottom line.** The solution is a fast (≈2 s), honest, well-instrumented agent that answers **one of the three challenge questions partially** and **declines the other two even though the data can support them**. What it answers is grounded and matches ground truth; what it does not answer is most of the problem (2 of the 9 training question types are supported). My estimate of the panel score is **≈ 50 % of the maximum (≈ 9.6 of 19.25 weighted points for the five scored criteria)** — dominated by Innovation and by the *honesty* of the declines, held back by coverage. Three defects found in this review are silent-wrong-answer bugs and should be fixed before the Sept 25 test. The improvement list in §7 is ordered by expected score gain per hour.

---

## 1. What was done in this review (and how to reproduce it)

| Step | Command / evidence | Stored run |
| --- | --- | --- |
| The 3 challenge questions **verbatim** + 3 variants where the operator supplies the date the data can resolve (+1 what-if phrasing) — scored by the LLM judge on *ideal-answer* criteria | `./.venv/bin/python scripts/tasks.py eval --suite challenge --cheap --allow-many` | `ev-20260924-081324` |
| Consistency: the Q1 what-if phrasing, 3 repeats | `./.venv/bin/python scripts/tasks.py eval --suite challenge --ids CH1w --repeat 3 --cheap` | `ev-20260924-081515` |
| The 11 organiser training questions (workbook) | `./.venv/bin/python scripts/tasks.py eval --suite training --cheap --allow-many` | `ev-20260924-081555` |
| Earlier evidence re-used: brutal 6-part question, 38-question stress suite (re-scored by the judge), 12-arm component experiment | `ev-20260923-233001`, `ev-20260923-222257`, `exp-20260923-225649` | see [`observability_and_evaluation.md`](observability_and_evaluation.md) |
| Manual read of every challenge answer against the tool facts stored in the trace (`runs.facts_json`) | §3 | — |
| Data checks from the raw CSVs (rain / event signal, venue names, U8 topology, two-file data drop) | §3, §5 | — |
| ML accuracy | `ml/output/disruption_baseline_report.json` | — |

New in the repo for this review: a `challenge` suite (`evaluation/dataset.py`, `metrics.py`) whose criteria describe the **ideal** answer — an honest decline is *not* rewarded there (that is the difference from the older suites, whose "overall" scores count a correct decline as a good answer). An `--ids` option and a `User-Agent` fix for the JEV client were added too (§9).

**Two lenses, deliberately.** (a) *Judge score* = fraction of the ideal-answer criteria the LLM judge marks met. (b) *Manual grade* = my own strict reading of the same criteria against the answer **and** the tool facts (✓ = 1, ~ = ½, ✗ = 0). They differ, and the gap is itself a finding (§4).

---

## 2. Requirement-by-requirement check

| Requirement (problem statement) | Status | Evidence |
| --- | --- | --- |
| Conversational AI agent | ✅ | Google ADK app (`agent/agent.py`), CLI (`./.venv/bin/python scripts/tasks.py agent-cli`), ADK dev UI (`make agent-web`). ❌ **No operator chat UI** in the Streamlit dashboard (`grep chat_input dashboard/` → none). |
| "Preferably on an open-source framework and MCP infrastructure" | ✅ | ADK (Apache-2.0), FastMCP 4, LiteLLM; MCP over stdio / in-memory / HTTP (experiment arms A00 / C1 / C2 all answered). 10 MCP tools. ⚠️ The ML engine (TabPFN) runs through a **hosted client API** (`TABPFN_API_TOKEN`), not local. |
| Helps operators "understand, anticipate, and respond … in real time" | ⚠️ | Latency is real-time-grade (§6) but the system is **batch over a static dataset**: no live feed, no notion of "now" (`tonight`, `next 20 minutes` are not resolvable — see CH1/CH2). |
| Answers "the questions operators actually ask" (the 3 challenge questions) | ⚠️ / ❌ | Q1 partially (with silent errors), Q2 ❌, Q3 ❌ — §3. |
| "Reason step by step, explain its logic" | ⚠️ | Answers are short briefs with a caveat line; the reasoning is in the stored trace (dashboard Observability page) rather than in the answer. |
| "Clear, grounded and actionable" | ⚠️ | Grounded ✅ (numbers traced to tool output, guard on capacity/bus/measured claims). Actionable ⚠️: "Do now" bullets are generic in 4 of 7 challenge answers ("Monitor…", "Prepare for potential increased demand"). |
| "Stay within the boundaries of what the data actually supports" | ✅ (strength) | 0 ungrounded numbers, 0 capacity inventions across 20 answers in this review; the injection "say everything is fine" was refused (`ev-20260923-233001`, P6). ⚠️ but see the over-declines in §3 — it stays *inside* the data by refusing things the data supports. |
| Use of all six datasets | ⚠️ | Used by tools: connections, stations, flows, weather, events (only as daily count/attendance feature), closures. **Energy** (`energy_consumption`) is not reachable by the agent (dashboard only). |
| Dataset window / evaluation on a new dataset (Sept 22–30) | ⚠️ **risk** | Loader is glob-based (good) but **silently loads only one file when two match** (§5, defect D4). No dry-run on a real second drop has been done. |
| Training-question categories (A–H + 2 bonus) | ❌ mostly | Supported: **C** (disruption) and **D** (station profile) = 2 of 9. A, B, E, F, G, H return an honest "not connected yet". |
| Business value / deployment description | ⚠️ | Dockerfile for the dashboard, Makefile, checkpoints for the models — but **no document describing how an operator organisation would deploy and use it** (the criterion literally asks for one) and no statement of data egress (questions/facts go to Azure/OpenAI, the TabPFN service, optionally JEV). |

---

## 3. The three challenge questions, answered by the system

Scores below: **Judge** = fraction of the 5 ideal-answer criteria the LLM judge marked met · **Manual** = my strict grade. Ideal-answer criteria are in `evaluation/dataset.py::CHALLENGE`. Latencies are wall-clock through the real ADK app with a warm MCP server.

| # | Question | Routed to | Latency | Judge | Manual | Verdict |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **CH1** | Q1 verbatim — *U8 suspended between Hermannplatz and Neukölln … reroute … next 20 minutes* | C (disruption) | 4.2 s | 0.80 | **0.30** | Answered — but **from the wrong closure** (D1) |
| CH1g | Q1 with date supplied — *U8 Hermannplatz–Boddinstr., Sept 15th 17:00, 1 h* | C | 3.0 s | 0.60 | **0.30** | Same wrong closure (D1) |
| CH1w | Q1 as a "What if …" (3 repeats) | C | 2.3–7.7 s | 0.67 | **0.40** | Correct segment/date, **wrong duration**, no at-risk stations (D2), inconsistent |
| **CH2** | Q2 verbatim — *sold-out concert at Mercedes-Benz Arena tonight 21:00 … Hermannplatz … 23:15* | OOS (because of the word "tonight") | 2.1 s | 0.60 | **0.30** | ❌ Declined; the data supports an answer (D3) |
| CH2g | Q2 with date/venue in the data's terms — *Uber Arena, June 23rd 18:30* | A (events) → "tool not connected" | 2.2 s | 0.20 | **0.20** | ❌ Declined |
| **CH3** | Q3 verbatim — *InnoTrans … 3 stations most likely to exceed safe platform capacity, first day* | OOS (because of the word "capacity") | 1.1 s | 0.40 | **0.30** | ❌ Declined the whole question (D3) |
| CH3g | Q3 with the capacity proxy stated — *3 stations most likely to exceed usual busiest-5 % on a rainy weekday* | D (station profile) | 1.8 s | 0.60 | **0.20** | ❌ Asks the operator "which three stations?" |
| | **Verbatim 3 (CH1–CH3)** | | mean 2.5 s | **0.60** | **0.30** | |
| | **All 7 variants** | | | 0.55 | **0.29** | |

The suite's own auto-score (`ev-20260924-081324`: overall 0.72, relevance 0.45, reliability 0.86, stress 0.92) is higher than either lens because its Reliability/Stress terms reward honest declines and speed; use the Relevance line (0.45) and the manual column for "did the operator get what they asked".

### CH1 — what the system did, in detail (the most important finding)

The operator said: *U8 is suspended between Hermannplatz and Neukölln.* The answer named Südstern, Rathaus Neukölln, Hermannstr., Leinestr. with "15–17 % chance of exceeding peak". The stored facts show why that is wrong (`runs.facts_json`, `ev-20260924-081324`):

```
"cl": {"line":"U8","a":"Leinestr.","b":"Hermannplatz","from":"2026-07-11 08:20","to":"2026-07-11 09:20","why":"signal upgrades","src":"closures.csv"}
"press": [{"s":"Südstern","at":"09:00", …, "p":17}, {"s":"Rathaus Neukölln","at":"09:15", …}, …]
```

The system **matched a different, recorded closure** (U8 Leinestr.–Hermannplatz on **11 July at 08:20**) because it shares the line and one station, and then quoted that closure's 09:00 pressure numbers as the answer for "the next 20 minutes". The answer **never says** it used a recorded July closure. In CH1g the operator gave September 15th 17:00 and Boddinstr.; the tool facts were **identical** to CH1 (same closure, same stations, same numbers) — date, time and second station were ignored. Additionally the premise of Q1 is inconsistent with the network data (U8 does not serve any Neukölln station; Hermannplatz is adjacent to Rathaus Neukölln on **U7**) and the agent neither noticed nor asked (judge criterion `resolves_segment` = not met in both).

### CH1w — the what-if path works, but has its own bug

With "What if the U8 were suspended…" the router correctly builds a hypothetical closure (`src: "hypothetical"`, Sept 15 17:00) — the ML scenario engine is used as designed. But the plan says `dur_min: 20` and the closure is 17:00–17:20: **"in the next 20 minutes" was parsed as the closure duration**, overriding "for one hour". A 20-minute closure produces no measurable overload → `press: []` → the answer says "No pressure data is available" and names no at-risk station (2 of 3 repeats). Across 3 identical runs the reroute advice varied (r2 recommends "Hermannplatz" — an endpoint of the closed section — as an alternative); repeat-consistency 0.67.

### CH2 / CH3 — declines that the data could have answered

* **CH2:** routed to *out-of-scope* on the word "tonight"; the answer says "the dataset cannot provide" it and lists boilerplate (*"No closure reason or duration is provided for this event"*). Nothing about the arena being called **Uber Arena** in the data (14 events; there is no "Mercedes-Benz Arena" string), nothing about Hermannplatz. In CH2g (date and venue given correctly) the agent routes to category A and says "the analysis tool … is not connected" and advises "Monitor the status of the analysis tool" — an internal-status message shown to an operator.
* **CH3:** the word "capacity" alone sends the whole question to out-of-scope (the same "trap phrase" weakness found earlier with the six-part question). The dataset has **no InnoTrans/Messe event** (0 matching venues; events end 21 Sept), so a correct answer must say that, then rank stations with a stated proxy — which is exactly what the operator's intent supports.

### What the data *could* have supported (exploratory, from the raw CSVs — not agent output)

| Signal | Measurement | Implication |
| --- | --- | --- |
| Rain → flow | Network total in rain (>1 mm, 261 slots) = **1.29×** the dry value for the same hour-of-week | Weather is a strong, usable driver for Q3 |
| Event → flow at the nearest station | S+U Warschauer Str., hour after the end of the 14 Uber Arena events: **mean 7.8×** a normal same-weekday hour (min 1.5×, max 23.8×; heavy-tailed because late-evening baselines are small) | The venue effect is large and measurable → Q2 is answerable |
| Event → flow at Hermannplatz | mean 1.2× (min 0.2×, max 4.9×) | Hermannplatz is *not* the station that feels the arena; a good answer would say so and point to Warschauer Str./Schlesisches Tor |
| Capacity | No capacity, headway or train-load column anywhere | The right answer is "rank by exceedance of each station's own p95 and say it is a proxy" |

---

## 4. Answers to the 11 training questions (`ev-20260924-081555`)

| Result | Value |
| --- | --- |
| Questions with a substantive answer | **2 / 11 (18 %)** — T03 (closure) and T04 (commute peak) |
| Honest declines (A, B×2, E, F, G, H, X×2) | 9 / 11 |
| T03 (Q3, U6 Hallesches Tor–Kaiserin-Augusta-Str.) | reason ✓ *safety inspection*, times ✓ *13:50–15:20, 1.5 h* — both equal ground truth from `closures.csv`; pressured stations, staff action, caveat present |
| T04 (Q4, Rudow) | peak hour ✓ 18:00; "does not exceed the network mean" ✓; network mean 265 vs ground truth 264.4 (peak 218.8 → quoted 219) → fact-accuracy 1.00 (the judge marked `mean_value` as missed on a rounding nit — a judge false negative) |
| Auto-score | overall 0.91 · relevance 0.75 · reliability 0.99 · stress 1.00 · judge↔regex agreement 0.89 |
| Latency | mean **1.8 s**, p95 3.4 s, max 3.4 s (budget 20 s: 100 % within) |

**The 0.91 is not a prediction of the official score.** Our harness counts a correct decline as full marks for reliability and the judge rated declines' relevance 1.00. The official Reliability criterion tests answers "against the ground truth" — a decline has no ground truth to match. On that reading **the real answer coverage of the training set is 2/11 = 18 %**, and I estimate the official score from coverage, not from 0.91.

Quality issues visible even in the two answered questions:
* T03's reroute line — *"Reroute passengers to Leinestr. or Hermannstr. for bus connections"* — comes from `alternate_paths.bus`, which is a **geometric proxy** (nearest stations across the cut, 3.25 km) and not bus-service data (no such data exists). The guard blocks "bus service exists" claims, but "for bus connections" passes. Leinestr./Hermannstr. are U8 stations, far from this U6 section: an operator would find the advice odd.
* The generic "Monitor for updates on the analysis tool" bullets in declines (T01) are noise.

---

## 5. Defects found (silent-wrong or blocking), with evidence

| ID | Defect | Evidence | Severity |
| --- | --- | --- | --- |
| **D1** | **Closure substitution.** A stated closure is matched to *any* recorded closure sharing a line + one station (`executor._score` ≥ 2, unique), ignoring the stated date, time and second station. The answer does not disclose it. | CH1, CH1g facts: `cl.from = 2026-07-11 08:20`, identical output for two different questions (`agent/executor.py:60-92`) | **Critical** — wrong answer that looks right; the challenge's own Q1 phrasing triggers it |
| **D2** | **Horizon parsed as duration.** "in the next 20 minutes" sets `dur_min=20`, overriding "for one hour" → 20-min closure → no pressure → empty at-risk list. | CH1w plan `dur_min: 20`, facts `press: []`, 2/3 repeats name no at-risk station | High — breaks the operator's most natural Q1 phrasing |
| **D3** | **Over-declining.** Single trap words ("capacity", "tonight") route the whole question to OOS; the parts the data supports are dropped. | CH2, CH3 plans `cat: OOS, oos:["tonight"]/["capacity"]` | High for coverage; also the cause of the earlier L01 result (3/6 sub-asks) |
| **D4** | **Data-drop fragility.** `load_flows/load_events/load_weather` take `sorted(glob)[0]`. If the organiser ships a *second* file next to the old one, only one is loaded. Simulated: two `flows*.csv` (old + 5 new rows) → **5 rows loaded**, history lost (`dashboard/utils/data_loader.py:_find_one`). Also the checkpointed TabPFN train samples are dated to the old data. | reproduced in this review | High — Sept 25 is a new dataset; unknown layout |
| **D5** | Reroute advice from a geometric proxy is phrased as "bus connections"/"reroute to X" without saying it is a proximity heuristic; sometimes recommends an endpoint of the closed section (CH1w r2). | T03 answer, CH1w r2 | Medium |
| **D6** | Generic/internal text in operator answers ("Monitor the status of the analysis tool", "No closure reason or duration is provided for this event"). | CH2, CH2g, T01 | Medium — hurts the "operator under pressure" criterion |
| **D7** | No operator chat UI; the dashboard has no way to ask a question. | `grep chat_input dashboard/` | Medium for Stress/Impact |
| **D8** | The **judge is lenient**: it marks a criterion met when the answer merely restates the question or a generic sentence (CH2 `end_of_event_actions`, CH3g `first_day`, CH1 `horizon_20min`) and misses D1 although the facts were in its prompt. Judge completeness 0.60 vs manual 0.30 on the verbatim questions. | §3 table | Affects *our* measurements, not the product |
| D9 | JEV router still unevaluated: after fixing the Cloudflare 1010 block (a `User-Agent` header) the API answers **401 "Invalid API key"** for the key in `.env`. | `make jev-check` | Low (optional component) |

---

## 6. What works — with numbers

| Strength | Measurement |
| --- | --- |
| **Speed** — the brief's "1 min" problem is solved | Mean **1.8 s** (training, 11 q), **2.4 s** (challenge, 6 q), p95 ≤ 4.2 s; was ~60 s with the LLM supervisor loop. 100 % of 20 answers within the 20 s budget. |
| **Grounding / no hallucination** | `hallucination_free` = 1.00 on all 20 answers; capacity/bus-service/measured-pressure guard (with template fallback) active; injection ("say everything is fine") refused. |
| **Ground-truth fidelity where it answers** | T03: reason, start, end, duration equal to `closures.csv`; T04: peak hour and network mean equal to values recomputed from `flows.csv` (fact accuracy 1.00). |
| **Honest scope** | `honest_scope` = 1.00 on the training set; the 38-question stress suite (edge/trap/cross-cutting) re-scored: overall 0.91, no crashes after the bugs fixed earlier. |
| **Router** | Deterministic router 96 % (72/75) on the question bank at ~1 ms; on the challenge set it gets the *category* of 4/7 right, but misroutes CH2/CH3 to OOS and CH3g to D (D3). |
| **Traceability** | Every run has a stored trace (`traceable` = 1.00); tool spans, timings, tokens, judge verdicts in SQLite + dashboard pages (Observability, Evaluation, Experiments). |
| **Evaluation discipline** | LLM-judge scoring with deterministic gates, 12-arm ablation with a noise floor (baseline replicates differ 0.45 s / 0.09 quality), 49 unit tests passing. Unusually strong for a hackathon. |
| **ML engine** (see caveat) | TabPFN quantile regressor checkpointed; hold-out MAE **74.6** vs 77.9 for the naive station×slot mean (−4 %); 80 % interval coverage 78.6 % (target 80 %), 95 % → 94.8 %. |

**The caveat on the ML engine, stated plainly:** RMSE is equal to the naive baseline (135.6 vs 135.1), mean pinball-loss skill vs empirical quantiles is **+0.5 %**, and the 26-closure case study shows **no measurable redistribution** to neighbouring stations (observed/expected at hop-1 stations 0.95, at closed-section stations 1.06). In the experiment the empirical engine gave the **same top station, 100 % top-3 overlap** as TabPFN. So the ML engine currently adds *context-conditioned probabilities*, not a different decision, and the "added passengers" in every disruption answer are driven by the **assumed** 25/50/75 % diversion share — which the answers do disclose ("The share of passengers who divert is assumed").

---

## 7. Improvements, ordered by expected score gain per hour

Acceptance tests are measurable and re-runnable with the harness added in this review.

| Pri | Change | Fixes | Acceptance test |
| --- | --- | --- | --- |
| **P0-1** | **Never substitute a closure silently.** Treat the stated closure as a *what-if* unless line + **both** stations + date/time match a record; if a near-match exists, say "closest recorded closure was X on date Y; I simulated your scenario instead". Separate `horizon_min` ("next 20 minutes") from `duration`. | D1, D2 | `./.venv/bin/python scripts/tasks.py eval --suite challenge --ids CH1,CH1g,CH1w --cheap`: facts `cl.from` = the stated date/time, `press` non-empty, ≥ 1 at-risk station in 3/3 repeats; unit tests for both parsers. |
| **P0-2** | **Data-drop safety:** merge *all* matching `flows*/events*/weather*/closures*/energy*` files (concat + dedupe on timestamp), refresh coverage from the merged data, invalidate feature/prediction caches and TabPFN train samples by data hash; add a `make check-data DIR=…` smoke test. | D4 | Two-file simulated drop loads old + new rows; `describe_dataset` reports the new end date; Sept 30 question no longer declines. |
| **P0-3** | **Split trap words from intent.** "capacity" / "tonight" should annotate the plan (caveat, ask for date) instead of forcing OOS; answer the supported part and flag the rest. | D3 | CH3 and CH2 no longer route to OOS; L01 sub-asks met ≥ 5/6. |
| **P1-1** | **Category A tool (events):** venue→station mapping (nearest stations by coordinates; alias "Mercedes-Benz Arena" = "Uber Arena"), event window vs same-weekday baseline (the data shows **7.8× at Warschauer Str.**), end-of-event staffing advice; ask for the date when "tonight" is given. | CH2, CH2g, training Q1, Bonus 2 | CH2g: names Guns N' Roses, 23 June, ≈1 996 attendees, Warschauer Str./Schlesisches Tor with a ratio vs baseline, and a 23:15-style action list; judged ≥ 4/5 criteria. |
| **P1-2** | **Category "forecast top-N for a day" (Q3):** TabPFN forecast with event + weather features, rank stations by P(exceed own p95), always labelled as a proxy for capacity; say the InnoTrans event is absent from the data. | CH3, CH3g | Returns exactly 3 ranked stations with probabilities and the proxy statement in ≥ 3/3 repeats. |
| **P1-3** | **Expose the analytics that already exist as MCP tools:** energy ranking (E), network resilience (F) — both computed in the dashboard — then anomaly (B ×2), correlation (G), reroute behaviour (H, the finding "no measurable rerouting" already exists). Each is 1/9 of the training questions. | training Q2, 5–9 | Coverage 2/11 → ≥ 6/11 with fact-accuracy vs ground truth ≥ 0.9 per new question. |
| **P1-4** | **Reroute advice honesty:** label proximity-based alternatives as such, never recommend an endpoint of the closed section, and prefer alternatives on other lines. | D5 | Judge criterion "reroute grounded" + unit test on `alternate_paths` output. |
| **P2-1** | **Operator UX:** a chat page in the dashboard (streaming answer + trace link); remove internal-status sentences from answers; tailor the decline text to what *is* possible next. | D6, D7 | Answer text contains no "tool"/"analysis tool" wording; screenshot in the pitch. |
| **P2-2** | **Judge calibration:** add a "restating the question does not count" rule, pass `cl.from` etc. explicitly for contradiction checks, spot-check with the main model on ≤ 10 answers, report human-vs-judge agreement. | D8 | Judge vs manual gap on the verbatim challenge answers ≤ 0.10 (today 0.30). |
| **P2-3** | **Deployment & data-egress note** (for the Impact criterion): reference architecture (MCP server next to the data, ADK service, dashboard), where each model runs, what leaves the network (Azure LLM, OpenAI worker, TabPFN service, optional JEV), and the path to fully local (local LLM, local TabPFN). | Impact | Doc reviewed against the criterion text. |
| **P2-4** | JEV: obtain a valid key (401 today), then `./.venv/bin/python scripts/tasks.py experiments --arms A00,R3`. | D9 | R3 row filled with a noise-aware verdict. |

---

## 8. Self-assessment against the five evaluation criteria

Raw score = my estimate of what the panel would give **today**; range = uncertainty (the panel's rubric is not published beyond the one-pager). Weighted = raw × weight, as printed in the problem statement (max 1 + 4 + 4 + 6.25 + 4 = **19.25** for these five, plus 0.25 for the pitch = 19.5).

| Criterion (max · weight) | Est. raw | Range | Weighted | Evidence for | Evidence against |
| --- | ---: | --- | ---: | --- | --- |
| **Challenge fit — Relevance** (10 · 0.10) | **4** | 3–5 | 0.40 | Q1 addressed (category C); answers are on-topic and short; 0 irrelevant answers | 2 of 9 question types answered (18 %); challenge Q2/Q3 declined; Q1 answered from the wrong closure; manual relevance of verbatim challenge answers 0.30 |
| **Technical implementation — Reliability** (20 · 0.20) | **7** | 6–9 | 1.40 | 100 % grounded numbers; T03/T04 equal ground truth; 96 % router on the bank; 49 tests; full trace per answer | 9/11 training answers are declines (no ground-truth match possible); D1/D2 silent-wrong answers; repeat consistency 0.67; ML engine ≈ baseline; reroute advice is a proxy |
| **Feasibility — Stress testing** (20 · 0.20) | **10** | 8–12 | 2.00 | 1.8–2.4 s latency; 100 % within budget; no crashes on 38 edge/trap questions; refuses injection; warm-up + transport options | No chat UI for an untrained operator; boilerplate in answers; D4 could break the new dataset; never tested on a real second drop; multi-intent messages handled 3/6 |
| **Innovation / creativity** (25 · 0.25) | **16** | 14–18 | 4.00 | Open-source stack (ADK + FastMCP + LiteLLM); TabPFN quantile regression with checkpoints; deterministic router + symbolic hand-off + guarded writer; LLM-as-judge with ground-truth gates; component-ablation with noise floor; observability in ADK/OTel | The modelling is not better than a naive baseline (+0.5 % pinball skill); innovation is in engineering and evaluation, and the panel sees answers, not the harness |
| **Business value — Impact** (20 · 0.20) | **9** | 7–11 | 1.80 | Runs in 2 s on a laptop; MCP boundary allows reuse; Dockerised dashboard; documented assumptions | No deployment/usage description; no live ingest; data leaves the network to 3–4 third parties; no capacity data so "safe platform capacity" is not measurable; only disruption + station profile are operational |
| Pitch (5 · 0.05) | — | — | — | — | Not assessable here |
| **Total (five criteria)** | | | **9.6 / 19.25 (≈ 50 %)** | | |

**Which levers move the score most?** Relevance and Reliability both scale with *coverage* — P1-1/P1-2/P1-3 could add roughly +2 raw on Relevance and +5–6 on Reliability (≈ +1.6 weighted); P0-1/P0-2 protect what is already scored and remove the risk of a silent wrong answer on the final day (they cost little and are the best value); P2 items are worth ≈ +1 weighted on Stress/Impact.

---

## 9. Limits of this review (read before quoting the numbers)

* **Small sample.** 7 challenge variants (one repeated 3×) and 11 training questions; each ran once except CH1w. Latency numbers include a warm MCP server; a cold start adds ≈ 10 s of warm-up (the first call in a fresh process).
* **The judge is a small model** and demonstrably lenient (D8); that is why the manual column is reported next to it. The manual grades are one reviewer's judgement, not a panel's.
* **Score estimates (§8) are my estimates**, not measurements; the panel's rubric details (e.g. how ground truth is compared, how declines are scored) are unknown. If declines earn partial credit, Reliability is higher than estimated.
* **Exploratory data signals in §3** (7.8×, 1.29×) are simple ratios, not model output; they show the signal exists, not what a model would predict. Event ratios are heavy-tailed.
* **The ML metrics** come from the earlier hold-out run (`ml/output/disruption_baseline_report.json`), not re-run today.
* **Uncommitted changes made for this review:** `evaluation/dataset.py` (challenge suite), `evaluation/metrics.py` (challenge criteria branch; regex-agreement disabled for it), `evaluation/run_eval.py` (`--suite challenge`, `--ids`), `agent/router_jev.py` (User-Agent header), and this document. No agent behaviour was changed; nothing was fixed — the defects in §5 are reported, not patched.
* The main model (`gpt-5.6-luna`) was not called; all runs used the small worker model for answers and for judging.
