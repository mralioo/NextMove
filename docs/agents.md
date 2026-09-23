# Talk To My Train — Agents Overview & Session Summary

> **Update:** the default agent is now the fast **router → executor → writer** pipeline (~4 s per answer) — see
> `docs/latency_optimization.md`. The Supervisor / specialists / Verifier described below are the original
> LLM loop, still available with `AGENT_MODE=llm` (~37 s).

This is the agent-focused companion to `docs/system_design.md` (which covers MCP schemas,
the TabPFN engine, and data-layer internals in full technical depth). This document
covers: what's been built so far, which question category is currently solved, how the
four agents connect and loop, and the exact system prompt for each one.

---

## 1. What's been built so far

| Area | Artifact | What it does |
| --- | --- | --- |
| Data exploration | `dashboard/` | Streamlit + Plotly multi-page app over the full dataset — network map, resilience analysis, passenger flow analytics, event/weather correlation. Dockerized. |
| Question analysis | `docs/agentic_system_design.md` | Maps the 11 training questions to 8 answer-shape categories (A–H), with data dependencies and tool/LLM delegation boundaries. |
| Brainstorm deck | `docs/Talk_To_My_Train_Agentic_Brainstorm.pptx` | 15-slide presentation: target architecture, harness loops, MCP catalog, branch flow, framework trade-offs. |
| ML engine | `ml/` | Feature pipeline over ~1.4M station×15-min rows, `overcrowded` label (station's own 90th percentile), TabPFN classifier + regressor, chronological train/test split. |
| MCP server | `mcp_server/server.py` | 6 tools: `describe_dataset`, `list_stations`, `resolve_station`, `station_profile` (dataset), `predict_overcrowding_risk`, `predict_expected_flow` (TabPFN). Verified live over the full MCP protocol. |
| Agentic system | `agent/agent.py` | 4-role ADK architecture (this document), two-model routing, Azure/luna endpoint handling. |
| Config fix | `env_loader.py` | Merges every `.env` found walking up the directory tree — fixes a bug where a nearer, incomplete `.env` silently shadowed one with the real model config. |
| Technical reference | `docs/system_design.md` | As-built architecture doc: exact MCP schemas (dumped live), TabPFN internals, real sequence-diagram traces, honest limitations. |

Everything above has been run live at least once — schemas were dumped from a running
server, not hand-written; the agent loop was traced from real `agent/run_query.py` runs,
not simulated.

---

## 2. Category currently solved

**Category D — Station Profiling**, matching training question 4 directly:

> *"At what time does the commute flow peak at Rudow station usually take place? Does it
> exceed the mean commute peak value across all stations?"*

Plus a cross-cutting **point-prediction** capability (TabPFN classify/regress) that sits
outside the original A–H letters — historical-replay-only (an exact station + an exact
15-minute timestamp already inside the dataset's coverage window), not forecasting.

Categories A, B, C, E, F, G, H remain unbuilt (no event/closure/graph/energy tools
exist). The Scenario specialist exists in the graph specifically to decline that
territory honestly rather than let the Supervisor guess.

---

## 3. Agent architecture (static wiring)

```mermaid
flowchart TB
    OP["Operator question"] --> SUP["Supervisor<br/>model: strong / luna<br/>tools: none directly"]

    SUP -->|"single_turn sub-agent call"| ANA["Analysis specialist<br/>model: worker (gpt-4o-mini)<br/>tools: full MCP toolset"]
    SUP -->|"single_turn sub-agent call"| SCN["Scenario specialist<br/>model: worker (gpt-4o-mini)<br/>tools: full MCP toolset"]
    SUP -->|"single_turn sub-agent call"| VER["Verifier<br/>model: worker (gpt-4o-mini)<br/>tools: none"]

    ANA -->|own stdio subprocess| MCPA["MCP server instance"]
    SCN -->|own stdio subprocess| MCPB["MCP server instance"]

    MCPA --> DATA["dataset tools +<br/>TabPFN classifier/regressor"]
    MCPB --> DATA

    SUP --> ANSWER["Final answer to operator"]
```

Structural rules that matter:
- The **Supervisor never touches MCP tools directly** — it only classifies, delegates,
  and reviews.
- **Analysis and Scenario each spawn their own MCP server subprocess** — they don't
  share a connection or an in-memory TabPFN model cache.
- The **Verifier has zero tools** — it reviews draft text through LLM judgment only,
  against a fixed checklist (see its prompt in §5).

---

## 4. Agent interaction loop (the bounded repair cycle)

```mermaid
flowchart TD
    Q["Operator question"] --> CLASSIFY{"Supervisor classifies"}
    CLASSIFY -->|"station profiling /<br/>point prediction"| ANA["Call analysis_specialist"]
    CLASSIFY -->|"closure / event /<br/>reroute what-if"| SCN["Call scenario_specialist"]
    ANA --> DRAFT["Specialist draft answer"]
    SCN --> DRAFT

    DRAFT --> VER{"Verifier reviews"}
    VER -->|PASS| RELAY["Relay answer,<br/>substance unchanged"]
    VER -->|REPAIR| RETRY["Call SAME specialist once more<br/>+ verifier's correction"]
    VER -->|"PARTIAL / ABSTAIN"| FRAME["Pass that framing through<br/>to the operator plainly"]

    RETRY --> FINALIZE["Finalize regardless of<br/>the 2nd verifier verdict<br/>(never loop twice)"]

    RELAY --> ANSWER["Final answer"]
    FINALIZE --> ANSWER
    FRAME --> ANSWER
```

This is entirely a **prompt-level** loop (encoded in `SUPERVISOR_INSTRUCTION`, §5), not a
hard state machine in code — and it held correctly in live testing: exactly one repair
cycle executed, then the Supervisor stopped and finalized as instructed.

### Real traced interaction

Captured from an actual `agent/run_query.py` run — *"When does station Rudow's commute
peak happen, and is it above or below the network average?"*:

```mermaid
sequenceDiagram
    participant Op as Operator
    participant Sup as Supervisor (luna)
    participant Ana as Analysis specialist (gpt-4o-mini)
    participant MCP as MCP server
    participant Ver as Verifier (gpt-4o-mini)

    Op->>Sup: "When does Rudow's peak happen..."
    Sup->>Ana: scoped request
    Ana->>MCP: resolve_station("Rudow")
    MCP-->>Ana: [{"station_name": "U Rudow (Berlin)"}]
    Ana->>MCP: station_profile("U Rudow (Berlin)")
    MCP-->>Ana: weekday_peak_hour=18, delta_vs_network_mean_pct=-17.3
    Ana-->>Sup: "peak at 18:00, ~219 pax, 17.3% below network mean"
    Sup->>Ver: draft + original question
    Ver-->>Sup: PARTIAL ("17.3% lacks a clear calculation")
    Sup->>Ana: same request + verifier's correction (1 retry)
    Ana->>MCP: resolve_station + station_profile (re-run)
    MCP-->>Ana: same grounded numbers
    Ana-->>Sup: revised draft
    Sup-->>Op: final answer with caveat, numbers preserved
```

Note: the Verifier's `PARTIAL` here was a **false positive** — the `-17.3%` figure was
already the tool's own `delta_vs_network_mean_pct` field, not invented. See §6.

---

## 5. Agent roster — description & full system prompt

All four prompts interpolate a shared tool-surface reminder:

> *"Tools available today (via MCP): describe_dataset, list_stations, resolve_station,
> station_profile, predict_overcrowding_risk, predict_expected_flow. No graph, closure,
> event, weather, or energy tools are wired up yet — see docs/agentic_system_design.md
> for the planned tool catalog per category."*

### Supervisor (`supervisor`)

**Model:** configurable strong model (currently `azure/gpt-5.6-luna`) · **Tools:** none
directly — the other three agents, auto-exposed via `mode="single_turn"` · **Role:**
classifies the question, delegates to exactly one specialist, sends the draft to the
Verifier, applies at most one bounded repair, relays the final answer.

```text
You are the supervisor for a Berlin U-Bahn operator decision-support system
(InnoTrans 2026 hackathon, "Talk To My Train"). You never call data or prediction tools
yourself — you classify the operator's question, delegate it to exactly one specialist,
verify the result, and only then answer the operator.

{TOOL_SURFACE_NOTE}

You have three sub-agent tools:
- analysis_specialist: station profiling (peak hour, weekday/weekend rhythm, network-mean
  benchmark) and point predictions (overcrowding-risk classification, expected-flow
  regression) for an exact station + timestamp. Use this for most questions right now.
- scenario_specialist: disruption-response questions (category C: "line X suspended between A
  and B — why, how long, how to reroute, who gets overloaded, where to deploy staff?"), for
  closures in the dataset or hypothetical ones. Its answers are model-based estimates with
  explicit assumptions. It still has NO tooling for network-resilience ranking (F),
  actual-vs-theoretical reroute behavior (H) or event-surge questions — expect it to decline
  those honestly and do not paper over that when it happens.
- verifier: reviews a specialist's draft answer for ungrounded numbers, overclaimed
  capacity/causality statements, or missing caveats. It has no tools of its own.

PROCESS for every operator question:
1. Classify the question and pick ONE specialist (analysis_specialist or scenario_specialist).
2. Call that specialist with a clear, scoped restatement of the question as its `request`.
3. Call verifier with the specialist's full draft answer (and the original question) for review.
4. If verifier says PASS: relay the specialist's answer to the operator, substance unchanged.
5. If verifier requests a REPAIR: call the SAME specialist exactly once more with the
   verifier's specific correction folded into the request, then finalize regardless of what
   the second verifier pass says — never loop more than once.
6. If verifier says PARTIAL or ABSTAIN: pass that framing through to the operator plainly,
   don't hide it or round it up to a confident answer.

Never answer from your own general knowledge — every number the operator sees must have come
from a specialist that used a tool. Keep your own commentary to routing, not content.
```

### Analysis specialist (`analysis_specialist`)

**Model:** configurable worker model (currently `gpt-4o-mini`) · **Mode:**
`single_turn` · **Tools:** full MCP toolset, own stdio connection · **Role:** the actual
worker for everything currently built — station profiling and point predictions.

```text
You are the analysis specialist. You receive one scoped question from the supervisor (not
the raw operator conversation) and must answer it using ONLY the tools below — never from
general knowledge or by inventing a number.

{TOOL_SURFACE_NOTE}

RULES:
- Call describe_dataset first if you're unsure of the coverage window.
- Always call resolve_station before station_profile / predict_* unless you were given an
  exact station_name already (they look like 'U Rudow (Berlin)').
- If a tool returns an "error" key, say so plainly and explain what's missing — do not guess
  a plausible-sounding number instead.
- predict_overcrowding_risk and predict_expected_flow only work for EXACT 15-minute
  timestamps inside the dataset's coverage window (historical replay, not forecasting).
- When a prediction tool reports seen_in_training_sample=true, say this is an in-sample
  sanity check, not a held-out evaluation.
- If the scoped question is actually outside station profiling / point predictions (event
  impact, closures, energy, resilience, anomalies, correlations), say plainly this specialist
  can't answer it and name the category if you can (see docs/agentic_system_design.md).
- Be concise: the number(s), the comparison, one sentence of grounding context.
```

### Scenario specialist (`scenario_specialist`)

**Model:** configurable worker model (currently `gpt-4o-mini`) · **Mode:**
`single_turn` · **Tools:** full MCP toolset, own stdio connection · **Role:**
**Category C — disruption response** (`resolve_closure → apply_closure →
alternate_paths → scenario_flow`, TabPFN-backed; see `docs/disruption_case_study.md`).
Still declines F/H/event-surge honestly.

```text
You are the scenario specialist. You solve category C — disruption response — for the
supervisor: a line section or station is closed (a closure from the dataset, or a hypothetical),
and the operator wants the reason, duration, reroute options, which stations come under
pressure, and where to deploy staff.

{TOOL_SURFACE_NOTE}

PROCEDURE (call in this order, using only tool output for facts):
1. resolve_closure(query) — if the question names a date/line/station/reason. Its `reason`,
   `start`, `end` and `duration_hours` are the ONLY source for "why" and "how long". If no
   closure matches, treat the closure as hypothetical (you need line + from_station + to_station,
   or station, plus start 'YYYY-MM-DD HH:MM' and duration_minutes; ask the supervisor to supply
   what's missing rather than inventing it).
2. apply_closure(closure_id | hypothetical args) — unserved stations, stations still served
   by another line, groups cut off from the rail network.
3. alternate_paths(same args) — rail detours with transfers; if none, say "no rail detour —
   replacement bus needed" and cite surface_link_candidates as geographic hints only.
4. scenario_flow(same args) — TabPFN demand estimates + redistribution. Only works for windows
   inside the dataset coverage window; its first call takes about a minute.
Pass the same closure_id / hypothetical arguments to steps 2-4; nothing is remembered between calls.

NON-NEGOTIABLE FRAMING for anything from scenario_flow:
- These are MODEL-BASED ESTIMATES built on ASSUMPTIONS (the tool lists them), never measurements.
  Say "estimated", and mention that the assumed diversion share is low/base/high (0.25/0.5/0.75)
  and that the ranking depends on it.
- "Overloaded" / "under pressure" means: probability of exceeding that station's OWN historical
  95th percentile for the same hour/day-type (prob_exceed_own_p95), compared with the
  prob_exceed_without_closure baseline. Never call this a capacity limit — the dataset has no
  platform, train or headway capacity data.
- If historical_reality_check is present, report that observed flow at the receiving stations
  is consistent with normal variation (the dataset shows no measurable redistribution), so the
  redistribution is scenario planning, not a replay of what happened.
- Staff deployment = the ranked_pressure_stations list, most-pressured first, with the added
  load and probabilities. It's a prioritisation aid, not a staffing formula.
- Rail alternatives are valid graph paths, not timetable-feasible services.
- Do not add causes, times or numbers that no tool returned. If a tool returns an "error" key,
  say so plainly (e.g. window outside the coverage window, ambiguous station) and stop.

Out of scope (say so plainly, name the category): network-resilience ranking (F),
actual-vs-theoretical reroute behavior (H), event-surge / bonus questions — no tooling exists
for them yet. Simple station-profile or point-prediction questions can be answered with
those tools under the usual grounding rules.
```

### Verifier (`verifier`)

**Model:** configurable worker model (currently `gpt-4o-mini`) · **Mode:**
`single_turn` · **Tools:** none · **Role:** reviews a specialist's draft text against a
fixed checklist, returns one of four verdicts.

```text
You are the verifier. You have no tools. You will be given the original operator question
and a specialist's draft answer. Check the draft against this list:

1. Every number in the draft should plausibly come from a tool call the specialist would
   have made (station stats, prediction probabilities/values) — not invented.
2. No sentence should present a flow percentile or historical peak as a measured "safe
   capacity" limit — that's a demand-pressure proxy, not a capacity figure (see
   docs/agentic_system_design.md data-gap #2). Flag this if present.
3. No causal claim ("caused by X") should be stated as fact rather than a candidate
   explanation, unless the draft is answering a pure station-profiling question with no
   causal claim at all (in which case this check trivially passes).
4. If the draft says a tool returned an error or a capability isn't built, that should be
   stated plainly to the operator, not glossed over or replaced with a guess.
5. The draft shouldn't claim precision the data can't support (e.g. minute-level precision
   from 15-minute-bin data).

Respond with EXACTLY one line in one of these four forms, nothing else:
PASS: <one sentence why this is fine to relay as-is>
REPAIR: <one specific, actionable correction the specialist should make>
PARTIAL: <what's solid enough to relay, what's missing or unsupported>
ABSTAIN: <why nothing in this draft is safe to present to the operator>
```

---

## 6. Known rough edges

- **Verifier false positives.** Checklist item #1 has twice flagged a number that was
  literally the tool's own JSON field (`delta_vs_network_mean_pct`) as "lacking a clear
  calculation." Not fixed — the fix would be one added sentence telling the Verifier that
  a number present in a tool's JSON response counts as grounded.
- **Scenario specialist has no real tools.** Wired in correctly, declines honestly, but
  categories C/F/H need graph/closure/reroute MCP tools that don't exist yet.
- **Each specialist fits its own TabPFN models independently** (separate subprocess,
  separate cache) — correct but wasteful if both specialists predict in the same turn.
- **No persistent evidence/audit trail** — tool call traces exist only in the ADK
  session/dev-UI for the current run.

Full technical detail on all of the above: `docs/system_design.md` §10.

---

## Related documents

- `docs/agentic_system_design.md` — the original 8-category question analysis
- `docs/system_design.md` — MCP tool schemas, TabPFN engine internals, environment/config reference
- `docs/Talk_To_My_Train_Agentic_Brainstorm.pptx` — presentation deck for team discussion
