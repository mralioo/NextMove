# Case study — Category C (disruption response) with TabPFN regression

**Question class (training Q3):** *"Line X is suspended between A and B. Why, how long, how do
passengers reroute, which stations get overloaded, where do I deploy staff?"*

**Chain:** `resolve_closure → apply_closure → alternate_paths → scenario_flow`
**Answer contract:** valid paths + *assumption-based* demand estimates; **no measured capacity claim.**

Code: `ml/disruption.py` (graph/closure logic) · `ml/demand_baseline.py` (TabPFN model) ·
`ml/scenario.py` (redistribution + pressure) · `ml/train_disruption_baseline.py` (train/eval/case
study) · `mcp_server/disruption_tools.py` (MCP wrapper). Regenerate all numbers below with
`make train-disruption`; visual, chart-by-chart explanation with good/bad verdicts vs the baselines is on the
dashboard's **ML Engine** page.

---

## 1. Which dataset, and why not "closure → flow uplift"

The obvious supervised framing — learn how flow changes at neighbouring stations during a closure —
does not survive contact with the data:

| Fact (verified on the 26 closures in `closures.csv`) | Consequence |
| --- | --- |
| Only **26 closures**; 15 are line sections, 11 are single-station closures | Far too few events to learn redistribution |
| A closed station's flow is **exactly 0** in the window (simulation rule) | Trivially known — nothing to learn |
| Neighbour stations (1–2 hops), section endpoints and interchanges are **statistically indistinguishable from normal** (table in §3) | There is no redistribution signal to fit; a model would fit noise |
| No capacity, headway, rolling-stock or per-line load column exists anywhere | "Overload" can only be *relative*, never a capacity claim |

So the suitable dataset for TabPFN regression is **`flows.csv` itself** (168 stations × 15-min,
≈1.39 M station-slot rows, joined with weather, events, station metadata): the regression target is
`passengers`, and the model's job is to say *what normal demand would have been* at each
station/slot **with a calibrated predictive distribution**. The disruption logic then sits on top as
explicit, labelled assumptions.

## 2. The TabPFN model

* **Task:** `TabPFNRegressor` (client API, `v3.5_default`), target = passengers per station per
  15 min. Uses the regression capability's **`output_type="quantiles"`** (13 levels, 2.5 %–97.5 %) plus
  `"mean"` — one fitted model gives a full predictive distribution
  ([docs](https://docs.priorlabs.ai/capabilities/regression)).
* **Training set:** 10 000 rows sampled from the train pool.
* **Features (17):** hour, 15-min slot, day-of-week, weekend, month; station average, **station ×
  day-type × slot profile mean** (target-encoded on the train pool only), n_lines, interchange flag,
  primary line; temperature, precipitation, wind, cloud cover, weather code; daily event count and
  attendance (city-wide proxy — events aren't geocoded to stations).
* **Leakage rules:** every row inside *any* closure window is excluded from fitting, profile
  features and the p95 reference; chronological split (first 80 % of days = train, last 20 %, from
  2026-09-02, = held-out); profile features come from the train pool only.

### Held-out accuracy vs naive baselines (last 20 % of days, no closure windows, 4 000 rows)

The baselines are built from the *same training days* with no ML: the station × day-type × slot
median/mean, and the empirical station × day-type × hour quantiles (same 13 levels as TabPFN).

| Metric | TabPFN | Baseline | Verdict |
| --- | ---: | ---: | --- |
| MAE (median forecast) ↓ | **74.62** | 75.42 | −1.1 % — tie |
| RMSE (mean forecast) ↓ | 135.61 | 135.05 | +0.4 % — tie |
| R² ↑ | 0.354 | 0.359 | tie |
| Pinball loss, 13 quantiles ↓ | **22.14** | 22.25 | −0.5 % — tie |
| 80 % interval coverage (target 80 %) | 78.6 % | 81.7 % | both within 2 pts — tie |
| 90 % / 95 % coverage | 89.7 % / 94.8 % | 91.6 % / 95.4 % | both calibrated |
| 80 % interval width ↓ | 237.3 | 239.7 | tie |
| Global-mean MAE (naive floor) | 110.3 | | |

**Honest reading.** Against a *fair* baseline TabPFN is **a tie, not a win**. The simulated flows are
"station × time-of-day pattern + noise" (R² ≈ 0.35, with 15.8 k readings pinned at exactly 500); once
the historical pattern is supplied as a feature, weather and events add almost nothing, so a lookup
table is as accurate and as well calibrated. What TabPFN gives is a *sound, calibrated predictive
distribution out of the box* (coverage within 1.4 pts of nominal at every level) from one model, with no
hand-built per-station tables — a convenience and robustness argument, not an accuracy one.
(An earlier version of this document credited TabPFN's calibration as its advantage; that was written
before an interval baseline existed and is superseded by this table.)

The overcrowding classifier tells the same story: ROC-AUC 0.860 vs 0.859 for a station × day-type ×
hour historical-rate baseline, PR-AUC 0.446 vs 0.417, and the baseline is slightly better at its
best-F1 cut-off (0.726 vs 0.694).

## 3. Case study — the 26 historical closures

For each closure the model predicts the counterfactual distribution at the closed footprint and its
1- and 2-hop rings; we compare with what was actually observed. If stations near a closure absorbed
displaced passengers they would sit **above** their interval. Nominal for "no effect": 80 % inside
q10–q90, 10 % above q90, 10 % below q10, observed/expected = 1.

| Group | Rows | Observed / expected | Inside 80 % | Above q90 | Below q10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Section suspension — unserved stations | 412 | 1.06 | 81.3 % | 8.7 % | 10.0 % |
| Section suspension — endpoints & interchanges | 382 | 1.01 | 78.5 % | 11.5 % | 9.9 % |
| Section suspension — 1 hop | 474 | 0.95 | 80.4 % | 9.7 % | 9.9 % |
| Section suspension — 2 hops | 602 | 1.04 | 79.2 % | 11.8 % | 9.0 % |
| Station closure — closed station | 120 | **0.00** | 5 % | 0 % | **95 %** |
| Station closure — 1 hop | 242 | 0.83 | 77.3 % | 10.7 % | 12.0 % |
| Station closure — 2 hops | 258 | 0.84 | 74.8 % | 10.1 % | 15.1 % |

**Findings**

1. A *station* closure is real in the data: the closed station reads 0 (model expected ≈ 267
   passengers/15 min at Turmstr.), so the counterfactual demand that must go somewhere is large and
   measurable.
2. *Line-section* suspensions leave **no trace in the flows** — even the "unserved" interior stations
   keep their normal flow, and every neighbour ring sits at the nominal rate.
3. Nowhere do neighbours rise above q90 more than ~12 % of the time (nominal 10 %), so **the dataset
   contains no measurable redistribution**. That is the empirical reason `scenario_flow` reports a
   *scenario* (assumptions, low/base/high) and not a *prediction of what passengers did* — and the
   reason the answer contract forbids a capacity claim. (Station-closure rings actually sit slightly *below* expected, 0.83×, on only 11 events — the opposite direction to redistribution, and small-sample.) (Category H, actual-vs-theoretical reroute
   behaviour, would need a different dataset; nothing here can answer it.)

## 4. Network facts the solver surfaces

Berlin's U-Bahn graph here has 167 stations and 182 edges — almost a tree. Of the 26 closures only 2
line-section closures (ids 15, 25) and the 2 Turmstr. station closures have a rail detour at all (e.g. U1 Hallesches
Tor ↔ Schlesisches Tor has 4-, 8- and 9-hop detours via U3/U6/U7/U8). Most sections cut the network
in two, so `alternate_paths` returns "no rail detour — replacement bus" plus **geographic surface-link
candidates** (closest station pairs across the cut by straight-line distance; a plausibility hint, not
a service).

## 5. `scenario_flow` — how the estimate is built

Per 15-min slot in the window:

1. TabPFN gives every involved station its normal-demand distribution (mean + quantiles).
2. **Displaced demand:** unserved stations → 100 % of their expected demand. Section endpoints and
   still-served interchanges → *crossing passengers* = `diversion_share × ½ × Σ flow`
   (**assumption**; run at 0.25 / 0.5 / 0.75).
3. **Redistribution (assumption):** unserved stations' boardings spill to the nearest open stations
   (≤ 2 hops, weight 1/hops). Crossing passengers follow the alternate paths (weight ∝ 1/hops) and
   load the *transfer* stations on them.
4. **Pressure:** P(baseline + added load > the station's *own* historical p95 for that hour and
   weekday/weekend) read off the TabPFN quantile grid, shown next to the same probability *without*
   the closure. Ranking = staff-deployment candidates.
5. **Reality check** (closures inside the dataset): observed vs baseline per receiving station.

Every response carries `data_mode: "model-based estimate, NOT a measurement"` and the assumption
list (`ml/disruption.py::ASSUMPTIONS`).

### Worked example — closure 17 (U9, Walther-Schreiber-Platz ↔ Bundesplatz, 2026-08-27 12:40–16:40, "safety inspection")

* `apply_closure`: Friedrich-Wilhelm-Platz is unserved; the section splits the rail network (3
  stations cut off from the rail network).
* `alternate_paths`: no rail detour → replacement bus; nearest cross-cut pairs 1.26–1.32 km
  (Schloßstr. ↔ Breitenbachplatz, Walther-Schreiber-Platz ↔ Rüdesheimer Platz).
* `scenario_flow` (base assumption): Friedrich-Wilhelm-Platz's expected ≈ 85 passengers / 15 min is
  displaced (≈ 1 363 over the window); Walther-Schreiber-Platz +49 (P(> own p95) 19 % vs 15 % without
  closure), Schloßstr. 13 % vs 10 %, Bundesplatz 9 % vs 8 %. Small, honest numbers: at this hour the
  closure barely moves the needle, and the tool says so.
* Agent run (Supervisor → Scenario specialist → 4 MCP tools → Verifier) produced the reason,
  duration, "no rail alternative", the ranked stations with the p95 definition, and an assumptions
  section stating there is no capacity data and no measurable historical redistribution.

## 6. Limits (carried in the tool docstrings and agent prompts)

* Redistribution shares and the spill/transfer rules are assumptions, not fitted parameters.
* "Overloaded" = own-p95 exceedance probability; not platform, train or headway capacity.
* Edge→line attribution is inferred (an edge belongs to every line serving both endpoints).
* Events are city-wide daily proxies; weather is real; flows are simulated.
* Windows must fall inside the dataset coverage; the September 22–30 evaluation data will work as
  soon as it is placed in `data/` (the feature table and profiles rebuild per server process).
* `ml/features.py::in_closure` matches station closures by substring, so "Neukölln" also flags
  "U Rathaus Neukölln" (a separate station). The Category C code uses exact resolution
  (`Network.resolve_one`) and does not depend on that flag; the older overcrowding models still do.

## 7. Model checkpoints (inference without refitting)

TabPFN is served through an API, so there are no local weights: fitting uploads the training rows and
returns a server `model_id`. `make checkpoints` saves, per model, in `ml/checkpoints/<name>/`:
`model.json` (the client's `save_model()` record: model id + hyperparameters, no data),
`train_sample.csv.gz` (the exact fit rows) and `meta.json` (features, fingerprint, metrics).
Models: `demand_baseline` (Category C), `overcrowding_classifier` and `expected_flow_regressor`
(the MCP predict tools). The MCP server restores them at start-up — `loaded` if the TabPFN server
still has the fit, `refit-from-checkpoint` (same data/settings, new id) if it has forgotten it, and a
fingerprint mismatch (different dataset, features, split or model version) marks the checkpoint stale
so a new dataset can never be served by an old model. `make checkpoints FORCE=1` refits everything.
Loading requires the same TabPFN account/token that fitted the model. Verified: second run restores
all three with unchanged ids; a corrupted id triggers a clean refit from the saved sample.
