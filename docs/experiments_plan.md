# Experiment plan and study — which component choices matter?

**Purpose.** The workflow is built from interchangeable parts: a router, a memory, an MCP transport, an ML engine and a writer.
This suite measures — quantitatively and qualitatively — what each choice changes, using **two questions** and one
configuration ("arm") per choice. It is also a template: adding a component to the study means adding one arm.

The earlier evaluation on the organiser workbook's questions (`./.venv/bin/python scripts/tasks.py eval`, suites `limit`, `training`, `stress`, …) is unchanged and
independent of this suite.

```
./.venv/bin/python scripts/tasks.py experiments --list          # show the design (arms, parameters, hypotheses) without running anything
./.venv/bin/python scripts/tasks.py experiments                        # run all arms, one process each (~9 min; the shared main model is used by ONE arm, 3 calls)
./.venv/bin/python scripts/tasks.py experiments --arms A00,R3   # a subset (e.g. baseline + the JEV router)
./.venv/bin/python scripts/tasks.py experiments-report                 # regenerate the RESULTS block below from the database
make jev-check                          # verify JEV_API_KEY and the response format (sends one public question to the JEV API)
dashboard → "Experiments"               # comparison, noise-aware verdicts, answers side by side
```

## 1. Research questions

| # | Question | Factor |
| --- | --- | --- |
| RQ1 | Does a learned or LLM router beat the deterministic rules — in accuracy, speed, cost? Is the JEV classification API usable as a router? | router |
| RQ2 | What does memory buy: answering follow-ups (session memory) and reusing past analyses across sessions (episodic memory)? | memory |
| RQ3 | What does the MCP transport cost or save (subprocess/stdio, in-process, HTTP)? | MCP |
| RQ4 | Does the ML engine (TabPFN) change the operational answer compared with a naive empirical baseline? | engine |
| RQ5 | What does the writer contribute: small LLM vs the shared main LLM vs no LLM (template)? | writer |

## 2. Design

**Two questions, three turns — the same protocol in every arm.**

| Turn | Text | Why it is there |
| --- | --- | --- |
| **Q1** | The workbook's disruption question: U6 suspended between Hallesches Tor and Kaiserin-Augusta-Strasse — reason, duration, reroute, overloaded stations, staff | Answerable; ground truth recomputable from the raw csv (reason *safety inspection*, 13:50–15:20, 1.5 h) |
| **Q2** | *"Which of those stations should get staff first, and how sure are you about that ranking?"* — asked in the **same session** | Only meaningful with memory of Q1: isolates RQ2 and tests how each router handles a follow-up |
| **Q1r** | Q1 again in a **new session** | Cross-session memory, consistency of repeated answers, caches |

**One factor at a time (OFAT).** Baseline **A00** = `router=rules | memory=session | mcp=stdio | engine=tabpfn | writer=small`
(the production workflow with the small writer model). Every other arm changes **exactly one** factor, so a difference is attributable to that
change. **A01** and **A02** are identical replicates of A00: the **range across A00–A02 is the noise floor** — a difference smaller than
it is not a finding. OFAT also keeps the bill small: the shared main model is used by one arm (3 calls); every other arm uses the small model or no LLM,
and the judge is the small model as well.

**Run naming.** Every arm's name spells out its whole parameter set, e.g. `router=llm|memory=session|mcp=stdio|engine=tabpfn|writer=small`.
It is stored in `exp_runs` with a setup snapshot (git commit, python/ADK versions, models, the *effective* configuration, protocol), and every
run row records the effective configuration, so a result can never be separated from the setup that produced it. Each arm runs in its own
process (own caches, own memory store, own MCP server).

### Factors and levels

| Factor | Levels | Meaning |
| --- | --- | --- |
| router | `rules` (baseline) · `llm` · `tfidf` · `jev` | deterministic rules + small-LLM fallback when unsure · small LLM always · local TF-IDF + logistic regression (category only; never trained on the two test questions) · JEV classification API |
| memory | `session` (baseline) · `none` · `episodic` | previous answer's facts kept for follow-ups · nothing kept · session memory + a persistent store of past facts reused across sessions (tools skipped) |
| mcp | `stdio` (baseline) · `inmemory` · `http` | server as a subprocess over pipes · server inside the agent process · server as a separate process over streamable HTTP |
| engine | `tabpfn` (baseline) · `empirical` | disruption scenarios from the TabPFN demand model · same logic from naive station × hour quantiles (no ML) |
| writer | `small` (baseline) · `main` · `template` | gpt-4o-mini · the shared main model · deterministic template (no LLM) |

### The JEV router (API, no local install)

`agent/router_jev.py` calls the JEV classification API exactly as documented in the post you linked
(`POST https://thejevai.com/v1/systemone`, `Authorization: Bearer $JEV_API_KEY`, a `state` plus a `choice` question whose `criteria` are the
question categories A–H, X, OOS; the answer's `choice` and `confidence` become the plan's category and confidence).

To run the arm: add `JEV_API_KEY=…` (and `JEV_API_URL=…` if the endpoint differs) to the repo's `.env`, run `make jev-check` once to confirm the key and
the response shape, then `./.venv/bin/python scripts/tasks.py experiments --arms A00,R3` (baseline and JEV in the same run, so they are directly comparable). Without a key the arm is
recorded as *skipped*, as in the results below.

Things to keep in mind: the adapter follows the documented format but has only been tested against a mocked response (no key was available while building it);
each JEV-routed turn sends the operator's question text to the third-party service; the post publishes no model card, weights or source, so the results
describe *the service as reached through its API* and are only reproducible while that service is available.

## 3. Hypotheses (written before the first result)

| Arm | Hypothesis |
| --- | --- |
| A01, A02 | Identical to A00; their differences show the noise floor. |
| R1 router=llm | Slower than rules and no more accurate on these two questions; may mis-route the follow-up. |
| R2 router=tfidf | Matches rules on category at ~ms cost; entities still come from rules. |
| R3 router=jev | Comparable category accuracy to rules; adds a network round trip per turn; sends questions to a third party. |
| M1 memory=none | The follow-up cannot be answered (no referent). |
| M2 memory=episodic | Q1r is faster (tools skipped) with identical facts. |
| C1 mcp=inmemory | Saves start-up and per-call overhead; identical answers. |
| C2 mcp=http | Adds overhead vs stdio; identical answers. |
| E1 engine=empirical | Similar ranking, different probabilities; speed similar or better. |
| W1 writer=template | Fastest and fully grounded, but stiffer and weaker on the follow-up. |
| W2 writer=main | Follows the style/honesty rules better, but slower and costs shared credit. |

## 4. Metrics

**How answers are scored — an LLM judge plus deterministic gates.** The semantic judgement is made by an **LLM judge** (`evaluation/judge.py`; the small
model, never the shared main one). For every turn it receives the question, the criteria in plain language, the **ground truth recomputed from the raw csv files**,
the tool evidence and the answer, and returns per criterion *met / not met with a quoted justification*, 1–5 scores for relevance, faithfulness and clarity, and the
list of claims it finds unsupported. What a machine verifies better than a language model stays **deterministic** (facts equal to ground truth, every number traceable
to tool output, latency, routing, stored trace, word count). The earlier keyword rubric is kept only as a **calibration signal**: `judge↔regex` reports how often the
two agree, and it is the fallback when the judge is unreachable.

**Quantitative** (per turn, then per arm)
* Speed: total latency and its split (route / tools / write), MCP start-up time, speed-up of Q1r over Q1.
* Cost: LLM calls, tool calls, tokens.
* **Quality score** = mean of route correctness, judge-decided completeness (Q1: reason, times, reroute, pressured stations, staff action, caveat), fact accuracy
  against ground truth (deterministic), grounding (no ungrounded number, no capacity / bus-service / measured-pressure claim; deterministic), readability, assumption
  disclosure, and the judge's relevance / faithfulness / clarity.
* **Follow-up completeness** (Q2, judge: names the earlier stations, gives an order, says how sure it is and why).
* Consistency Q1↔Q1r (agreement of the numbers quoted), memory hit.
* ML engine: similarity of the pressure ranking to the baseline arm (same top station, top-3 overlap, mean probability difference, pair-order agreement).

**Qualitative**
* The answers side by side (dashboard section 5) for any turn and any set of arms.
* The judge's per-criterion verdicts with quoted justifications, its unsupported-claim list and one-line main criticism, and the failing checks per turn.
* A manual read of the differences (see section 6).

**Decision rule.** With one run per arm, a latency **or quality** difference counts only if it exceeds the **noise range across A00–A02** (measured separately for each).
Qualitative observations are reported as observations (arm, turn), not as effects.

**Scoring history (transparency).** The first analysis of `exp-20260923-225649` used the regex rubric only (every arm scored ~1.00). The same recorded answers were then
**re-scored with the LLM judge** (`experiments/rescore.py`; arms were not re-run) — quality changed for 9 of 12 arms, and the tables below are the judge-based ones.

## 5. Results

<!-- RESULTS:START -->
### Results of `exp-20260923-225649`

Baseline = mean of the 3 identical replicates: **3.28 s** mean latency per turn; noise range across replicates **0.45 s** (3.28, 3.51, 3.06 s); baseline quality 0.97 (replicates 0.91, 1.00, 1.00; noise range **0.09**). A latency or quality difference counts only if it exceeds the noise range. Quality and judge scores come from the **LLM judge + deterministic gates** (see section 4).

#### All arms

| Arm | Run name (full parameter set) | Mean latency (s) | Δ vs baseline | vs noise | Quality | Δ quality | vs noise | Judge | Judge↔regex | LLM calls | Tool calls | Tokens |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A00 | `router=rules\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=small` | 3.28 | +0.00 | baseline replicate | 0.91 | -0.06 | replicate | 0.93 | 0.89 | 3 | 6 | 4909 |
| A01 | `router=rules\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=small` | 3.51 | +0.23 | baseline replicate | 1.00 | +0.03 | replicate | 1.00 | 1.00 | 3 | 6 | 4958 |
| A02 | `router=rules\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=small` | 3.06 | -0.23 | baseline replicate | 1.00 | +0.03 | replicate | 1.00 | 1.00 | 3 | 6 | 4939 |
| R1 | `router=llm\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=small` | 4.35 | +1.07 | **beyond noise** | 1.00 | +0.03 | within noise | 1.00 | 1.00 | 6 | 6 | 4980 |
| R2 | `router=tfidf\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=small` | 3.37 | +0.08 | within noise | 0.97 | +0.00 | within noise | 0.93 | 1.00 | 3 | 6 | 4921 |
| R3 | `router=jev\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=small` | – | – | skipped: JEV_API_KEY not set — the JEV router is opt-in (it sends the | – | – | – | – | – | – | – | – |
| M1 | `router=rules\|memory=none\|mcp=stdio\|engine=tabpfn\|writer=small` | 3.47 | +0.18 | within noise | 0.85 | -0.12 | **beyond noise** | 0.87 | 0.78 | 4 | 6 | 4393 |
| M2 | `router=rules\|memory=episodic\|mcp=stdio\|engine=tabpfn\|writer=small` | 3.07 | -0.22 | within noise | 0.97 | -0.00 | within noise | 0.93 | 0.94 | 3 | 4 | 4949 |
| C1 | `router=rules\|memory=session\|mcp=inmemory\|engine=tabpfn\|writer=small` | 2.85 | -0.43 | within noise | 0.96 | -0.01 | within noise | 0.93 | 0.89 | 3 | 6 | 4957 |
| C2 | `router=rules\|memory=session\|mcp=http\|engine=tabpfn\|writer=small` | 2.94 | -0.34 | within noise | 0.97 | +0.00 | within noise | 0.93 | 1.00 | 3 | 6 | 4947 |
| E1 | `router=rules\|memory=session\|mcp=stdio\|engine=empirical\|writer=small` | 3.19 | -0.10 | within noise | 0.99 | +0.02 | within noise | 1.00 | 0.94 | 3 | 6 | 4941 |
| W1 | `router=rules\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=template` | 0.23 | -3.05 | **beyond noise** | 0.99 | +0.02 | within noise | 1.00 | 1.00 | 0 | 6 | 0 |
| W2 | `router=rules\|memory=session\|mcp=stdio\|engine=tabpfn\|writer=main` | 5.91 | +2.63 | **beyond noise** | 1.00 | +0.03 | within noise | 1.00 | 1.00 | 3 | 6 | 5508 |

#### Router (RQ1)

| Arm | Route time Q1 (ms) | Route time Q2 (ms) | LLM calls | Quality | Follow-up completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| A00 | 20 | 14 | 3 | 0.91 | 0.67 |
| R1 | 1743 | 1321 | 6 | 1.00 | 1.00 |
| R2 | 1729 | 12 | 3 | 0.97 | 1.00 |

_R3 (JEV): skipped — JEV_API_KEY not set — the JEV router is opt-in (it sends the question to a third-party API whose model card / weights are not published)._

#### Memory (RQ2)

| Arm | Follow-up completeness | Quality | Tools time Q1 (s) | Tools time Q1r (s) | Tool calls | Speed-up Q1r | Memory hit on Q1r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A00 | 0.67 | 0.91 | 0.46 | 0.19 | 6 | 11% | False |
| M1 | 0.00 | 0.85 | 0.49 | 0.20 | 6 | 17% | False |
| M2 | 1.00 | 0.97 | 0.48 | 0.00 | 4 | 24% | True |

#### MCP transport (RQ3)

| Arm | Server start-up (s) | Tools time Q1 (s) | Tools time Q1r (s) | Mean latency (s) |
| --- | ---: | ---: | ---: | ---: |
| A00 | 3.1 | 0.46 | 0.19 | 3.28 |
| C1 | 1.9 | 0.21 | 0.21 | 2.85 |
| C2 | 3.3 | 0.48 | 0.21 | 2.94 |

#### ML engine (RQ4)

| Arm | Top-3 stations | Same top station | Top-3 overlap | Mean abs Δ probability (pts) | Pair-order agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| A00 | Kaiserin-Augusta-Str., Mehringdamm, Ullsteinstr. | True | 100% | 0.0 | 100% |
| E1 | Kaiserin-Augusta-Str., Mehringdamm, Ullsteinstr. | True | 100% | 9.2 | 100% |

#### Writer (RQ5)

| Arm | Write time Q1 (s) | Tokens (in+out) | Quality | Judge | Consistency Q1↔Q1r | Guard on Q1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A00 | 3.39 | 4909 | 0.91 | 0.93 | 0.60 | pass |
| W1 | 0.00 | 0 | 0.99 | 1.00 | 1.00 | template (config) |
| W2 | 6.88 | 5508 | 1.00 | 1.00 | 0.76 | pass |

_Consistency Q1↔Q1r across arms that should be equivalent: A00 0.60, A01 0.58, A02 0.54, C1 0.55, C2 1.00, R2 1.00 — a wide spread, i.e. this metric is noisy at n = 1._
<!-- RESULTS:END -->

## 6. Findings for `exp-20260923-225649` (first valid full run, LLM-judge scoring; written by hand from the tables above and the judge's justifications)

**Noise first.** The three identical baseline runs differ by **0.45 s** in mean latency (3.06–3.51 s) and by **0.09** in quality (0.91, 1.00, 1.00 — the low one is a follow-up
answer that says "measured pressure", see finding 1). Every difference below is judged against those two ranges.

### Router (RQ1)
* Rules, TF-IDF and LLM routers all sent Q1 to the disruption playbook and the follow-up to the right place. On two questions of one category the router choice cannot show
  *accuracy* differences; the router evaluation on the 75-question bank (96 % for rules, ~1 ms) is the accuracy evidence.
* **LLM router: +1.07 s (beyond noise), 6 LLM calls instead of 3, ~1.7 s of routing per turn instead of 20 ms — for no quality gain** (quality 1.00, within noise).
* **TF-IDF:** +0.08 s (within noise), quality 0.97 (within noise), but its first routing call took 1.7 s because it **trains lazily on first use** (sklearn import + fit) — a
  start-up cost; pre-training at start-up is the obvious fix (not applied, to keep the run faithful).
* **JEV: not run** (no API key yet). Run it as described in section 2 and regenerate the report.
* **Verdict:** rules are the best trade-off here — as good, ~85× faster than the LLM router, no extra calls.

### Memory (RQ2)
* **Session memory is decisive for follow-ups.** Without it the follow-up is routed to a station lookup and answered *"I need the name of the station"*: the judge marks all three
  follow-up criteria unmet (follow-up completeness **0.00** vs 1.00 in most other arms) and gives relevance 1/5; arm quality **0.85** vs a replicate mean of 0.97 — **beyond the 0.09 quality noise**.
* **Episodic memory worked mechanically** — the repeated situation in a new session came from memory (tool time 0.00 s, 4 instead of 6 tool calls) — but the latency gain is
  **not measurable**: −0.22 s (within noise); speed-up of Q1r 24 % vs 19 % for the baseline replicates. Tools were already ~0.2 s (warm server, prediction cache) and the writer LLM (~3 s)
  dominates. It would matter where tools are slow (cold server, uncached model calls) and for auditability, not for speed here. Quality 0.97 (within noise; the one miss is a sampled Q1r
  answer that omitted the reroute).

### MCP transport (RQ3)
* In-memory: server start-up **1.9 s vs 3.1 s** (stdio) and Q1 tool time 0.21 s vs 0.46 s, but the mean-latency effect is −0.43 s — **within noise** (noise 0.45 s). HTTP: start-up 3.3 s,
  tool time equal to stdio, −0.34 s (within noise). The hypothesis that HTTP adds overhead is **not supported** at this scale (localhost, ~6 calls).
* Quality 0.96 / 0.97 (within noise). Answer content was the same in every transport; once the server is warm, the transport is a deployment choice, not a performance lever.

### ML engine (RQ4)
* **Same operational ranking:** same top station (Kaiserin-Augusta-Str.), top-3 overlap 100 %, pair-order agreement 100 %. What changes is the **confidence**: that station exceeds its own
  p95 with **78 %** (TabPFN) vs **61 %** (empirical); mean probability difference 9.2 points.
* The empirical engine's "without closure" probability is **5 % by construction** (the threshold *is* its own p95), while TabPFN reports **17 %** for that peak slot: it conditions on
  time/weather/events and finds the slot busier than typical. Latency −0.10 s and quality 0.99 (both within noise).
* With the earlier finding that TabPFN ≈ the naive baseline on point accuracy: **the ML engine currently adds context-conditioned probabilities, not a different decision.**

### Writer (RQ5)
* **Template:** −3.05 s (**beyond noise**; ≈ 0.2 s per turn, zero LLM calls), always grounded, quality 0.99 — but the judge **consistently** finds the staff action missing in Q1 and Q1r
  (Q1 completeness 0.83): the template lists pressured stations but never says where to deploy staff. It is the natural safety net (it is what ships when the LLM is slow).
* **Main model:** +2.63 s (**beyond noise**), ~11 % more tokens, quality 1.00 (within noise) — on these questions it bought **latency and cost, not measurable quality**.
* **Small model:** adequate, with one real, repeatable defect (finding 1).

### Qualitative findings
1. **The small writer overclaims in the follow-up — now confirmed by the judge in two arms (A00 and C1):** "the ranking is based on *measured* pressure and observed demand" — false; it is a
   model-based scenario. The judge marks the follow-up's *states how sure and why* criterion unmet, lists the phrase as an unsupported claim and gives faithfulness 3/5; the old keyword rubric passed it
   (it saw the word "assumed"). The numeric guard cannot see wording like this. *Post-hoc fix (not re-run):* the guard now rejects "based on measured pressure/demand", with unit tests; re-run the
   writer arms to confirm the effect.
2. **`M1`'s failure is polite but useless:** it asks for a station name and then lists generic bullets instead of saying it has no earlier analysis to refer to — the writer prompt needs a rule for
   "asked to follow up without context".
3. **Template vs LLM prose:** the template states probabilities compactly ("78 % chance vs 17 % normally"), arguably *clearer* to an operator, but loses the action list; a hybrid (template facts + one
   advisory LLM sentence) is worth testing.
4. **What the judge does and does not add:** per-criterion verdicts with quotes were informative and caught what regexes missed (finding 1, W1's missing staff action, M1's non-answer). Its aspect scores
   (relevance / faithfulness / clarity) are lenient — W1 got 5/5/5 while failing a criterion — so the **criterion verdicts carry the signal**. Judge↔regex agreement per arm ranges 0.78–1.00, with the
   disagreements concentrated in the follow-up turn.
5. **The consistency metric is noisy:** baseline replicates score 0.54–0.60, yet C2 and R2 score 1.00 with no plausible mechanism — at n = 1 it cannot support a claim.

### Pre-registered hypotheses vs outcome

| Hypothesis | Outcome |
| --- | --- |
| R1 slower than rules, no more accurate | **Confirmed** (latency beyond noise, no quality gain, 2× LLM calls) |
| R2 matches rules at ~ms | **Partly** — same quality; first call cost 1.7 s (lazy training) |
| R3 JEV | **Not run** (no key) |
| M1 follow-up fails | **Confirmed** (completeness 0.00, quality beyond noise) |
| M2 Q1r faster | **Mechanism confirmed, speed effect not measurable** |
| C1 saves overhead | **Partly** — start-up −1.2 s and less tool time; total latency effect within noise |
| C2 adds overhead | **Not supported** |
| E1 similar ranking, different probabilities | **Confirmed** |
| W1 fastest, stiffer | **Confirmed** (and the judge finds it drops the staff action) |
| W2 better quality but slower | **Slower confirmed; quality gain not measurable** |

## 7. Threats to validity

* **n = 1 per arm** (three baseline replicates give a noise *range*, not a variance estimate). Small effects are indistinguishable from noise.
* **Two questions from one category plus one follow-up** show *mechanisms* (what breaks without memory, what a transport costs) but not router accuracy or generality.
* **OFAT ignores interactions** (e.g. episodic memory × slow tools; template writer × follow-up).
* **The judge is a small model** grading small-model output (self-preference bias, lenient aspect scores, some run-to-run variability); its criterion verdicts should be spot-checked against the
  main model (`python evaluation/rejudge.py --judge-model main` for evaluation runs) and read alongside the quoted justifications. Quality still sits near 1.0 for healthy configurations.
* **Shared-API variance:** latency includes network/LLM jitter that changes by the hour; compare arms only within one experiment run.
* **The first full run was invalid and was discarded:** it applied only the environment-driven factors (every arm looked identical — which is what exposed the bug). Every arm now
  verifies that its effective configuration equals its parameters (and refuses to run otherwise), and each run row records it.

## 8. Extending the study

* **Add an arm:** add a level to `agent/config.py` (`CHOICES`), implement it where the configuration is read, add `_arm(...)` with a hypothesis in `experiments/arms.py`, run
  `./.venv/bin/python scripts/tasks.py experiments --arms A00,<new>`, then `./.venv/bin/python scripts/tasks.py experiments-report`.
* **Next experiments:** (1) a factorial or fractional design over memory × writer × engine; (2) more questions across categories (the 75-question bank) using only the
  deterministic/TF-IDF/JEV routers (no LLM cost); (3) ≥ 3 repeats per arm with a real variance estimate; (4) a hybrid writer; (5) pre-train the TF-IDF router at start-up and re-measure;
  (6) a two-rater human rubric to calibrate the judge (agreement with human raters, not only with the old regex).
