# Agentic System Design — Question Categories & Data Dependencies

Working analysis for **Talk To My Train**: how the 9 training questions (+2 bonus) decompose into
capability categories, what each category needs from the provided data, and what that implies for
the agent architecture.

## 0. The key finding that shapes this design

Every scenario referenced in the training questions is **a literal row in the training dataset**,
not a hypothetical:

| Question | Verified against data |
| --- | --- |
| Q1 — Guns N' Roses at Uber Arena, June 23 | `berlin_events_summer_2026_pre_innotrans.csv` row 55: *"Guns N' Roses – World Tour 2026"*, `2026-06-23T18:30:00+02:00`, venue `Uber Arena`, `Uber-Platz 1`, attendance 1996 |
| Q3 — U6 suspended Hallesches Tor ↔ Kaiserin-Augusta-Str. | `closures_pre_innotrans.csv` row 8: `7/13/2026 13:50`, `1h30min`, *"Line U6 suspended on a section between Hallesches Tor and Kaiserin-Augusta-Str. due to safety inspection."* |
| Q4 — Rudow station | `U Rudow (Berlin)` is a real flow column with a full 15-min series |

This is the single most important design implication: **the agent's core job is grounded retrieval,
entity resolution, and correlation over known data — not forecasting.** The LLM's value-add is
(a) resolving fuzzy natural-language references ("the concert", "U6 suspension") to exact dataset
rows/entities, (b) orchestrating the right sequence of deterministic tool calls, and (c) turning
computed facts into an operationally-worded answer. It should **not** be doing arithmetic, path-finding,
or correlation "in its head" — every number in an answer must come from a tool call.

The two bonus questions are the exception — InnoTrans 2026 runs after the training window
(Sept 22 onward, training data ends Sept 21), so they require **analogical extrapolation** from the
closest comparable historical pattern rather than direct lookup. Flag these explicitly as lower-confidence.

## 1. Question categories

### Category A — Entity-anchored event impact (retrieval + spatial join)
**Q1, Bonus 2**
> "There's a concert at X on date Y. What will flow look like at neighboring stations, what should we do?"

- Resolve event name/date → row in `berlin_events_summer_2026.csv`
- Resolve venue/address → nearest U-Bahn station(s) (see §3, data gap)
- Pull historical flow at those stations for that date/time window, and for comparable past events
  at the same venue (14 Uber Arena events in-window — good comparison sample)
- Cross-check `closures.csv` and `weather_data.csv` for the same window (confounders)
- Operational recommendation = LLM reasoning layered on top of the retrieved numbers

### Category B — Anomaly detection + root-cause attribution
**Q2, Q7**
> "Find a flow peak caused by bad weather in this week" / "find 3 anomalies not explained by closures on this date, find the real cause"

- Needs a deterministic **anomaly scorer**: per-station-per-timeslot baseline (e.g. same
  hour-of-week mean/std over the full training window) → z-score or IQR outlier flag
- Needs a **rule-based elimination chain**: for each flagged anomaly, check in order —
  overlapping `closures.csv` row → overlapping `berlin_events_summer_2026.csv` row (venue near
  station) → `weather_data.csv` extremes at that timestamp → else "unexplained by provided data"
- This is the hardest category because it's multi-source AND requires *negative* evidence
  ("not explained by X") — the tool must be able to say "no closure/event matches" confidently,
  which means the elimination tool must run exhaustively, not partially

### Category C — Disruption operational response (graph + flow redistribution)
**Q3**
> "Line X suspended between A and B. Why, how long, how to reroute, who gets overloaded, where to deploy staff?"

- Reason/duration: direct lookup in `closures.csv` (deterministic, exact)
- Rerouting: shortest-path recomputation on `berlin_ubahn_connections.csv` graph **with the closed
  edge(s) removed** (networkx already used in the dashboard's resilience page)
- Overload prediction: redistribute the suspended segment's typical flow (from `flows.csv` baseline
  at the closed stations) across the alternate-path stations, compare against those stations' own
  baseline — flag which exceed some threshold
- **Data gap**: there is no platform-capacity field anywhere in the schema. "Overloaded" must be
  defined operationally (e.g. >X% above that station's own historical p95) and the agent must state
  this assumption explicitly rather than imply a real capacity limit is known

### Category D — Pure statistical station profiling
**Q4**
> "When does station X's commute peak occur, does it exceed the network mean?"

- 100% answerable from `flows.csv` alone via groupby(station, hour, weekday) aggregation
- No ambiguity, no other data source needed, no LLM inference required beyond phrasing
- Already implemented in the dashboard's Passenger Flow → "Commute-peak benchmark" section

### Category E — Efficiency / resource-allocation ranking
**Q5, Bonus 1 (partially)**
> "Which line has the worst energy-per-passenger ratio, why, what would help?"

- Ratio itself: deterministic, `energy_consumption.csv` ÷ line-level flow (flow approximated by
  summing stations serving that line — already flagged as an approximation in the dashboard since
  interchange stations serve multiple lines)
- **The "why" and "what would help" parts are NOT answerable from data** — the schema has no
  rolling-stock, route-length, or headway data. The agent must clearly label this part as a
  hypothesis/domain-knowledge inference, not a data-grounded fact, and ideally tie it back to the
  one thing it *can* check: whether the ratio is driven by low ridership vs high energy draw

### Category F — Network topology / resilience ranking
**Q6, Bonus 1 (partially)**
> "Rank stations whose closure fragments the network most, estimate impact, suggest mitigation."

- 100% graph theory + flow join, no LLM inference needed for the ranking itself: articulation
  points / betweenness centrality on the 168-node, 182-edge, single-connected-component graph,
  weighted by each station's average daily ridership
- Already implemented (`network_resilience()` in the dashboard)
- Mitigation suggestions are LLM domain reasoning grounded in the computed ranking + whether the
  station is an interchange (23 of 168 stations serve >1 line)

### Category G — Latent correlation discovery
**Q8**
> "Find station pairs with correlated demand despite no direct connection, explain the mechanism."

- Needs a **correlation-mining tool**: pairwise Pearson/cross-correlation across all 168×167/2
  station time series, filtered to exclude pairs that share an edge in `berlin_ubahn_connections.csv`
  (the "despite no direct connection" constraint), ranked by correlation strength
- Mechanism hypothesis is LLM reasoning over auxiliary signals the tool can also surface: shared
  line membership, geographic proximity (haversine distance from lat/lon), or a shared nearby event
  venue — the LLM should not invent a mechanism with no supporting signal from these

### Category H — Disruption behavioral inference
**Q9**
> "During disruptions, which alternate routes do passengers actually take vs. the theoretical shortest path?"

- Needs a **before/after comparison per closure event**: theoretical alternate path from the graph
  tool vs. actual flow deltas (closure-window flow minus matched-baseline flow) at all
  topologically-reachable alternate stations, not just the theoretical-shortest-path ones
- This is a "diff-in-diff" style tool: same weekday/hour baseline outside the closure window,
  compared to the closure window, per station
- Genuinely interesting/hard category — likely the strongest "innovation" scoring opportunity in
  the rubric since it requires combining graph + flow tools rather than reading either alone

## 2. Category → data dependency matrix

| Category | flows | stations/connections/lines | events | closures | weather | energy | Needs new tool |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| A. Event impact | ✅ | ✅ (venue→station) | ✅ | check | check | – | venue geocoder |
| B. Anomaly + root cause | ✅ | – | ✅ | ✅ | ✅ | – | anomaly scorer + elimination chain |
| C. Disruption response | ✅ | ✅ | – | ✅ | – | – | graph reroute + redistribution |
| D. Station profiling | ✅ | – | – | – | – | – | (aggregation only) |
| E. Energy efficiency | ✅ (approx.) | ✅ (line membership) | – | – | – | ✅ | none beyond dashboard's ratio calc |
| F. Network resilience | ✅ | ✅ | – | – | – | – | none — reuse `network_resilience()` |
| G. Latent correlation | ✅ | ✅ (exclude direct edges) | – | – | – | – | correlation-mining tool |
| H. Reroute behavior | ✅ | ✅ | – | ✅ | – | – | graph tool + before/after diff tool |

**Answer to "which categories need access to the data to make a decision":** all of them. There is no
question in the set answerable from general knowledge alone — every category requires at least one
grounded tool call, and B/C/G/H require *multiple* sources joined together. This is exactly the
"stay within the boundaries of what the data actually supports" requirement in the problem statement,
and it's also the evaluation panel's ground-truth check (Category: Technical Implementation/Reliability,
weight 0.2) — an answer that isn't traceable to a tool call is a hallucination risk by construction.

## 3. Data gaps to close before the agent can fully answer

1. **Venue → station mapping.** Events have address/venue text, no lat/lon or station key. Only
   **26 unique venue names / 47 unique addresses** across 417 events — small enough to geocode once
   (any geocoder, or manual lookup for the handful of major venues like Uber Arena, Olympiastadion,
   Waldbühne, Tempodrom) and cache as a static `venue_to_station.csv` join table (nearest station by
   haversine distance to `stations_with_ubahn.csv` lat/lon). Do this as a **build-time enrichment
   step**, not a live LLM guess — a wrong venue→station mapping silently poisons every event-impact answer.
2. **No platform/train capacity figure.** "Overcrowding" / "safe capacity" has no ground-truth
   threshold in the schema. Define it operationally per station (e.g. own-station p90/p95 of
   historical 15-min flow) and have the agent state that definition whenever it uses the word
   "overcrowded" or "at risk."
3. **Line-level flow is an approximation.** Flows are per-station; 23 interchange stations serve
   multiple lines, so summing member-station flows to get a "line total" double-counts at
   interchanges. Fine as a directional signal (already labeled as such in the Energy dashboard page)
   but the agent should say "approximate" rather than assert it as exact when asked line-level questions.
4. **No missing-value handling needed** — all provided CSVs are complete (verified zero NaNs across
   flows, weather, stations); no imputation logic required.

## 4. What to delegate to tools vs. leave to the LLM

**Delegate to deterministic tools (never let the LLM compute or path-find itself):**
- Any aggregation/statistic over `flows.csv` (means, percentiles, peaks, resampling)
- Any graph operation (shortest path, k-shortest alternates, articulation points, betweenness,
  subgraph-after-removal)
- Correlation/anomaly scoring (statistics, not vibes)
- Entity resolution / fuzzy lookup against events, closures, station names (string+date matching)
- Unit conversions and ratio math (energy/passenger, % deltas)

**Leave to the LLM (reasoning over tool outputs, not raw data):**
- Deciding *which* tools to call and in what order for a given natural-language question
- Turning a ranked list / z-score table into operator-facing prose ("deploy staff at X because...")
- Flagging uncertainty when a category-E/H-style "why" question outruns what the data can prove
- Synthesizing multiple tool outputs into one coherent multi-part answer (Q3 needs 4 tool calls:
  closure lookup, graph reroute, flow redistribution, threshold check)

## 5. Proposed agentic architecture

```
                         ┌─────────────────────────┐
   Operator question ──▶ │   Orchestrator LLM       │
                         │  (ReAct-style tool loop) │
                         └────────────┬─────────────┘
                                      │ tool calls (MCP servers)
        ┌───────────────┬────────────┼────────────┬───────────────┬──────────────┐
        ▼               ▼            ▼             ▼               ▼              ▼
   flows_query    network_graph  events_lookup  closures_lookup  weather_lookup  energy_query
  (agg/resample/   (shortest path, (fuzzy match    (fuzzy match     (window        (line ratio,
   percentile,      articulation    name/date/venue, name/date/line, lookup)        trend)
   z-score)         points,         → venue_to_      → structured
                    betweenness,     station join)    fields already
                    remove-edge                       parsed: type/
                    subgraph)                          line/segment)
        │               │            │             │               │              │
        └───────────────┴────────────┴─────┬───────┴───────────────┴──────────────┘
                                            ▼
                                 anomaly_detector / correlation_miner
                                 (composite tools built on flows_query +
                                  network_graph + closures/events lookups,
                                  for categories B, G, H)
                                            │
                                            ▼
                              Grounded answer, every number tagged
                              with the tool call that produced it;
                              explicit "not supported by data" flag
                              when a "why" question exceeds tool coverage
```

**Notes on the design:**

- Every tool listed above is already ~80% built as functions in `dashboard/utils/data_loader.py`
  (`flows_long`, `station_avg_flow`, `hourly_profile`, `build_graph`, `network_resilience`,
  `line_flow_series`, closure/event parsing). Wrapping these as **MCP tool endpoints** (per the
  problem statement's "preferably MCP infrastructure" ask) rather than duplicating logic is the
  fastest path — the dashboard and the agent should share one data layer.
- New tools needed beyond what exists: `venue_to_station` join table + lookup, `anomaly_detector`
  (baseline + z-score + elimination chain), `correlation_miner` (pairwise, direct-edge-excluded),
  `reroute_and_redistribute` (shortest-path-on-subgraph + proportional flow redistribution),
  `before_after_diff` (closure-window vs. matched-baseline flow delta per station).
- Keep the orchestrator to a **single LLM with a tool-calling loop**, not a multi-agent swarm — the
  question set doesn't need parallel specialized agents, it needs reliable sequential tool
  composition per question category above. Multi-agent overhead would add latency and hallucination
  surface without a clear benefit here; a swarm only starts to pay off if you later split "read-only
  Q&A" from "what-if simulation" as genuinely different response types.
- For the bonus/extrapolation questions (post-training-window), have the orchestrator explicitly
  search for the closest historical analog (largest past event, worst past weather week, etc.) via
  the same tools, then reason by analogy — and say so in the answer, since that's a materially
  different confidence level than the grounded-lookup categories.
