# ML preprocessing pipeline and the quality database

**Date:** 2026-09-24 · **Received as:** `docs/new_feature/nextmove_pipeline/` (kept untouched as the source) · **Integrated in:** `ml/nextmove_pipeline/` (package), `ml/quality_db.py` (database + boundaries + checks), `mcp_server/quality_server.py` (MCP server), `agent/quality_mcp.py` + `agent/evaluator.py` (the Inspector's use).
**Tests:** `tests/test_quality_pipeline.py` (7) · whole suite 125 pass · **Build:** ≈ 30 s first time (26 s pipeline incl. geocoding, 8 s database), 8 s afterwards.

## 1. What it is, in one paragraph

The raw flows are noisy and mix three things: the *normal* rhythm of a station (commuter waves, weekday / weekend), the effect of *weather*, and *events / closures / anomalies*. The pipeline separates them: per station it fits a **normal-flow model** on "clean" cells (event and closure windows excluded), then decomposes every 15-minute cell into `total = weather + rest` (log scale: 0 = normal, +0.1 ≈ +10 %, +0.69 ≈ twice as many). On top of that, `ml/quality_db.py` builds a **quality database**: normal flow, boundaries per station / day type / hour, ceilings, episode effects, unexplained spikes, outages and the data-quality issues found on the way. The **Inspector** (the evaluator agent) uses it through an **MCP server** to check every passenger figure of an answer before it is shown.

## 2. The pipeline (`ml/nextmove_pipeline`)

| Module | Job |
| --- | --- |
| `config.py` | every judgement call in one place: paths (training split = fit, test split = apply), thresholds, event size ≥ 2 000 visitors, MAX_HOP = 2, exclusion window −3 h / +2 h, school holidays, manual venue → station overrides |
| `loading.py` | reads the raw CSVs; **adapted to the organisers' test split**: UTF-8 with BOM, mojibake column names (`KurfÃ¼rstenstr.` → `Kurfürstenstr.`), the events file without header row (column names taken from the training file), `_pre_innotrans` / `_rest` file suffixes, network files taken from the training folder |
| `names.py` | station-name matching (`Spichernstr.` = `U Spichernstr. (Berlin)` = `Spichernstraße`) |
| `geo.py` | address → coordinates (OpenStreetMap Nominatim, 1 request/s, every answer cached in `data/processed/geocode_cache_pipeline.json`) → nearest stations; also a CLI |
| `episodes.py` | events → venue station (override → geocode → name in address); closures → **every station on the closed section**, not only the two ends |
| `baseline.py` | the model: per station ridge regression of `log1p(flow)` on time-of-day × day-type dummies, school holiday, rain now / previous hour, temperature (+ square), wind, linear trend; re-centred on the median of clean rest; `decompose()` → total / weather / rest / normal passengers |
| `pipeline.py` | end-to-end run; `--all` = fit on the training split and normalize the test split **with the same model** (suffix `_test`); episodes are built for the test split too |

Integration changes (everything else is the received code): repository-root paths; the loader repairs for the test split; the test split's events / closures are mapped even when the model is loaded (the database shows which episodes were active); `run_all()`; `geopy` added to `ml/requirements.txt`; span-free, standalone (no MCP dependency, as received).

```
./.venv/bin/python scripts/tasks.py quality-pipeline --all --freq 15min       # data/normalized/*.csv (+ model pickle)
./.venv/bin/python scripts/tasks.py quality-build [--offline]                 # pipeline if needed + data/quality (database)
./.venv/bin/python scripts/tasks.py quality-status
```

Outputs in `data/normalized/` (gitignored, 41 MB): `normalized_flows / _weather / _rest.csv` (+ `_test`), `normal_flow_passengers`, `normal_flow_coefficients`, `episodes`, `normal_flow_model.pkl`.

**Result on this data** (15-minute grid): 8 320 training slots + 720 test slots × 167 stations; the fit excludes 0.8 % of the cells (event / closure windows); 35 events ≥ 2 000 visitors → 31 mapped (15 by override, 16 by geocode; 4 without a station), 26 + 4 closures mapped; median network effects: rain +16 %, +5 °C −8 %, school holiday ≈ 0 %.

## 3. The quality database (`data/quality/`, gitignored, rebuilt by `make up` if missing)

`cells.npz` — every (timestamp × station) cell, training + test: `actual, normal, expected, total, weather, rest, excluded`. `quality.db` (SQLite):

| Table | Content |
| --- | --- |
| `meta` | build time, grid, training / test window, ceilings, pipeline description |
| `stations` | name, lines, coordinates, has flow data |
| `boundaries` (13 360 rows) | per station × day type (Mon–Thu / Fri / Sat / Sun) × hour: clean cells, normal median, **mean**, p01 / p05 / p50 / p95 / p99, max, log-residual quantiles and the **normal band** |
| `station_bounds` (167) | highest value ever observed, p99.9, max in event windows, daily mean, weekday / weekend peak hour and value, clipping share, **hard ceiling = 1.5 × max, soft ceiling = 1.15 × max** |
| `network_bounds` (80) | network totals per day type × hour: p05 / p50 / p95 / max / normal |
| `episodes` (67) | events and closures with anchors, window and the **measured effect** at the anchors (busiest-hour uplift, mean / min log deviation) |
| `event_effects` | summary by group (events < / ≥ 5 000 visitors, closures) |
| `coefficients` | rain / heat / holiday effects per station in % |
| `anomalies` (4 223) | unexplained **spikes** (rest ≥ 2, outside event / closure windows) |
| `outages` (2 061) | readings of exactly 0 where the station normally carries ≥ 100 per slot |
| `daily` | passengers per station and day |
| `data_issues` (9) | what a user of the raw data must know (below) |

**What the database says about the data itself** (`quality_data_issues`): 1.2 % of readings are exactly 500 (clipping) although values up to 3 000 exist; 0.4 % of well-used slots read exactly 0 outside any event / closure (outages — a zero is not proof of a closure); at 15 minutes the log-residual has σ ≈ 1.2 (single slots are noisy: judge patterns over hours); weather explains only 0.5 % of the variance (the flows are simulated with little weather signal); **the test split runs +0.10 (log) ≈ 10 % above the training normal** (InnoTrans week / trend), so held-out figures must also be judged against test-window boundaries; a duplicate interchange column (`U Stadtmitte (Berlin).1`) is dropped.

## 4. The MCP server (`mcp_server/quality_server.py`, port 8768, `make up` starts it as `mcp-quality`)

| Tool | Answers |
| --- | --- |
| `quality_status` | is it built, coverage, table sizes |
| `quality_normal_flow(station, at)` | actual vs normal vs expected-with-weather, weather effect %, rest %, episode active?, boundary for the slot |
| `quality_boundaries(station, at)` | normal median / mean, p05–p99, normal band, max ever, hard / soft ceilings |
| `quality_check_value(station, at, value)` | verdict `normal · low · high · extreme · impossible` with the reason |
| `quality_station_profile(station)` | bounds, weekday normal by hour, weather effects |
| `quality_episodes(date, station)`, `quality_event_effects` | events / closures and how much they moved the flow |
| `quality_anomalies(date, top_n)` | largest unexplained spikes of a day |
| `quality_weather_effects(station)`, `quality_network_bounds`, `quality_data_issues` | weather coefficients, network ranges, data issues |
| **`quality_check_facts(category, facts_json)`** | **the Inspector's check** (§5) |

Any MCP client can use it (`tasks.py mcp-quality` for stdio; the Resources page lists the HTTP endpoint).

## 5. How the Inspector (evaluator agent) uses it

The Inspector runs its usual checks (required content, station names, ground truth recomputed from the raw CSVs, confidence floor) and now also calls `quality_check_facts` through an **in-memory MCP client** (`agent/quality_mcp.py`: the server runs in the agent process, ≈ 10 ms per call, database loaded in the background at start). Span `inspector.quality_mcp` shows the call in the ADK trace.

| Category | Checks (hard = **H**, an answer must not be shown; soft = flag in the verdict) |
| --- | --- |
| D station profile | peak hour vs database (soft); weekday peak value within 40 % of the database (**H**) |
| B anomalies | the claimed observation equals the database cell (**H**); "usual" near the database normal / mean (soft); deviation ≥ 1 log unit and whether an event / closure window was active (soft) |
| C closure | predicted total per 15 min ≤ hard ceiling (**H**); "base" near the database normal / mean (soft) |
| P pressure | predicted load ≤ hard ceiling (**H**), ≤ soft ceiling (soft) |
| A events | excess ≤ hard ceiling (**H**); uplift vs the largest measured event uplift (soft) |
| F fragmentation | affected passengers ≤ the whole network's daily passengers (**H**); own daily passengers within 15 % of the database (soft) |
| H reroute | observed / expected ratios in 0–3 (**H**) |
| all | date inside the database window (soft) |

Hard failures become failed `Q-H*` checks → the verdict is `reject` (the Writer then ships the safe fallback instead of a figure the data contradicts); soft findings are added to the verdict's `issues` as "quality flag: …" (visible in the full report and the operations column); the boundaries used are added to the verdict (`Q-BOUND:<station>`) and the answer's Sources line names "quality database (normal-flow boundaries)". The knowledge base gained entries built from the database (`Q-NORMAL`, `Q-CEILING`, `Q-EVENTS`, `Q-WEATHER`, `Q-OUTAGE`, `Q-TESTSHIFT`, from `agent/knowledge_build.py`), so the boundaries also reach the Writer's caveats and the LLM stage of the Inspector.

**Measured on the 23 stored answers of the reference run:** 111 quality checks, **0 hard failures**, 1 soft flag (S1: a closure answer's "base" for Südstern is 2× the database mean); deliberately wrong facts (a Rudow peak of 900 at 03:00, an observation of 5 000 where the data says 2 605, a predicted load of 9 000) are caught (`tests/test_quality_pipeline.py`). Live: a new closure question took the Inspector 0.45 s including the quality call, and its Sources line lists the quality database.

## 6. Limits — please read

* **The model is simple** (ridge on log1p, per station, linear trend): at 15 minutes the residual is very noisy; boundaries therefore use empirical quantiles of clean slots, not the model's confidence bands. The *normal median* is below the *mean* (heavy tails, clipping at 500); checks compare against both.
* **Weather is almost irrelevant in this data**, so the weather / rest split changes little; the value of the pipeline here is the clean baseline, the episode windows and the boundaries.
* **Event uplift is a noisy statistic** (venue stations with a tiny normal flow show ratios of 8× and more); `Q-EVENT-RATIO` is only a soft check.
* **The database is built from the same CSVs the agent reads.** It is a consistency and plausibility check, not an independent source of truth; a systematic error in the raw data would pass.
* **Geocoding sends venue addresses to OpenStreetMap Nominatim** the first time (public venue addresses, cached afterwards in `data/processed/geocode_cache_pipeline.json`, committed); builds use `--offline` (cache only) by default in `make up`. Events without a station (4 of 35 ≥ 2 000 visitors) are ignored, events below 2 000 visitors are treated as noise.
* Only categories A, B, C, D, F, H, P are checked (E energy and G correlations use no per-station flows); the checks need dates inside the data window.
* `SCHOOL_HOLIDAYS` in `config.py` is marked "please verify" by the authors of the pipeline and was not verified here.
