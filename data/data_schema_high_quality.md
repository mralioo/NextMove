# High-quality (golden) data — schema

**What this is:** the pre-processed, normalized version of the organisers' raw data. It is produced by the normalization pipeline (`ml/nextmove_pipeline`, see `docs/ml_preprocessing_pipeline.md`) and lives in `data/normalized/` (tables and fitted model) and `data/processed/` (geocode cache). The quality database in `data/quality/` (boundaries, checks) is **derived** from these files. The raw files are described in `dataset_schema.md`.
**Access for agents:** the quality MCP server (`mcp_server/quality_server.py`, port 8768) — tools `golden_*` read exactly the files below; `quality_*` read the derived database. Start with `golden_datasets`.
**Regenerate:** `./.venv/bin/python scripts/tasks.py quality-build [--offline]` (≈ 30 s; `--offline` = no geocoding requests). Gitignored: `data/normalized/`, `data/quality/`; committed: `data/processed/geocode_cache_pipeline.json`.

```
raw CSVs (data/training dataset, data/testing dataset)
   │  loading.py  (BOM, mojibake, header-less events, file suffixes)
   ▼
nextmove_pipeline: episodes (events, closures) → normal-flow model fitted on CLEAN cells of the TRAINING split → decompose training AND test split with that same model
   ▼
data/normalized/   normalized_flows · normalized_weather · normalized_rest · normal_flow_passengers   (+ *_test)   episodes (+ _test) · normal_flow_coefficients · normal_flow_model.pkl
data/processed/    geocode_cache_pipeline.json
   ▼  ml/quality_db.py
data/quality/      cells.npz · quality.db   (boundaries, ceilings, episode effects, anomalies, outages, data issues)
```

## 1. Conventions (all normalized tables)

| Convention | Value |
| --- | --- |
| Grid | 15-minute slots, **naive Berlin local time**, 05:00 → 00:45 (no gap rows) |
| Splits | **train** 2026-06-10 05:00 → 2026-09-22 00:45 (8 320 slots); **test** 2026-09-22 05:00 → 2026-10-01 00:45 (720 slots). Files `*_test.csv` hold the test split, scored with the **same model** fitted on train |
| Stations | 167 flow stations, exact names as in `flows_*.csv` / `stations_with_ubahn.csv` (e.g. `U Rudow (Berlin)`, `S+U Warschauer Str. (Berlin)`); the duplicate interchange column `U Stadtmitte (Berlin).1` is dropped; mojibake in the test split's column names is repaired |
| Log scale | `total`, `weather`, `rest` are natural-log ratios: **0 = normal**, +0.1 ≈ +10 %, −0.2 ≈ −18 %, +0.69 ≈ twice as many, −4 ≈ 2 % of normal |
| Passengers | `normal_flow_passengers` is in passengers per 15 minutes; `actual` = the raw reading (reconstructable within rounding) |

**Identities that hold** (use them to cross-check any value):
`total = weather + rest` (exact) · `total = log1p(actual) − log1p(normal_passengers)` (`normal_passengers = expm1(normal)` ) ⇒ `actual ≈ expm1(total + log1p(normal_passengers))` (the tables are rounded — `total` to 4 decimals, `normal_passengers` to 0.1 — so the reconstruction matches the raw reading within rounding: median error 0.02 passengers, 99.96 % within 1 passenger, 99th percentile 4 %) · `weather = log1p(expected_with_actual_weather) − log1p(normal)` .

## 2. `data/normalized/` — wide tables (index `timestamp`, one column per station)

| File | Slots × stations | Value | Range (p05 / median / p95) | Meaning |
| --- | --- | --- | --- | --- |
| `normalized_flows.csv` | 8 320 × 167 | `total` = log1p(actual) − normal | −2.26 / 0.00 / +1.56 (min −6.5, max +6.1) | how far the reading is from normal: weather **plus** everything else |
| `normalized_weather.csv` | 8 320 × 167 | `weather` | −0.12 / 0.01 / +0.19 (min −0.78, max +0.48) | the part explained by rain now / in the previous hour, temperature (+ square), wind — small in this (simulated) data |
| `normalized_rest.csv` | 8 320 × 167 | `rest` = total − weather | −2.27 / 0.00 / +1.55 (min −6.6, max +6.1) | what is left: **events, closures, anomalies**, noise |
| `normal_flow_passengers.csv` | 8 320 × 167 | normal flow, passengers / 15 min | 0.3 / 51.7 / 221.1 (max 1 080.6; can be slightly negative, ≥ −0.2, at night at small stations) | the reference for "+1 200 people": what the station carries in a normal slot of that time of day and day type, typical weather |
| `*_test.csv` (4 files) | 720 × 167 | same | test rest p05 / median / p95 = −2.32 / 0.00 / +1.58 | the held-out days; total median is **+0.10** (≈ 10 % above the training normal: InnoTrans week / trend) |

No NaN cells (the night pause has no rows). Reading a cell: station `U Rudow (Berlin)`, `2026-07-14 17:00` → normal 133.2, actual 217.9, total +0.489, weather −0.067, rest +0.556 (`golden_series`).

## 3. `data/normalized/episodes.csv` (57 rows, train) and `episodes_test.csv` (10 rows, test)

Events and closures the model **excluded from its fit** (window −3 h … +2 h, stations within 2 hops of an anchor), and what the quality database uses to flag "an episode was active".

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | text | `ev_<row>` (event) or `cl_<row>` (closure) |
| `kind` | `event` \| `closure` | 31 events, 26 closures (train); 6 events, 4 closures (test) |
| `type` | text | event segment (`Music`, `Conference`, …) or `closure` |
| `name` | text | event name / closure description (as in the raw file) |
| `start`, `end` | timestamp | naive Berlin local time; a missing / invalid event end = start + 3 h |
| `anchors` | text | stations directly affected, joined with `; `. Event: the venue's station; closure: **every station on the closed section** (line-section path), or the single closed station |
| `attendance` | float | organiser's estimate (0 for closures); only events ≥ 2 000 visitors are episodes |
| `how` | text | how the anchors were found: `override` (domain knowledge, e.g. Uber Arena → Warschauer Str.), `geocode` (Nominatim + nearest station within 1 500 m), `name_in_address`, `single station`, `section U7 (3 stations)` … |

## 4. `data/normalized/normal_flow_coefficients.csv` (167 rows)

Per station, in **%** on the flow: `rain_now_pct`, `rain_prev_hour_pct`, `rain_both_pct` (median +16.4 %, range +1.2 … +39.8), `temp_plus5C_vs_mean_pct` (median −8.1 %), `school_holiday_weekend_pct`, `school_holiday_weekday_pct` (median 0 %). Column `station` = exact station name. Use for "rain adds ~+8 % here" style statements; the effects are small and noisy in this data.

## 5. `data/normalized/normal_flow_model.pkl` — the fitted model

A pickled `dict`: `B` (328 × 167 ridge coefficients: 320 level dummies = 80 time-of-day slots × 4 day types [Mon–Thu, Fri, Sat, Sun], then the 8 extra terms `holiday, holiday_weekday, rain_now, rain_prev_hour, temp_z, temp_z2, wind_z, trend`), `n_level`, `stats` (training slots, window, weather means / stds, mean 20.1 °C), `typical` (typical temperature / wind per slot), `shift` (median re-centring per station × slot × day type), `stations`, `freq` (`15min`), `extra_terms`. Model: per station ridge regression of `log1p(flow)` on clean cells only. `golden_normal_flow(station, at)` evaluates it for **any timestamp** (outside the training window the linear trend term is clipped to +1, so dates after the data are "normal flow of a typical day at the end of the window").

## 6. `data/processed/geocode_cache_pipeline.json` (17 entries)

`{"<venue>|<address>" (lower-case; empty venue → "|<address>"): {"lat", "lon", "source": "nominatim" | "not_found" | "manual", "query"}}` — every geocoding answer (also "not found", `lat/lon = null`) so each place is queried once; Berlin bounding-box filtered; older caches are read as seeds only. Used with the manual overrides (`config.VENUE_TO_STATION`) to place events at stations (`golden_venue_station`).

## 7. `data/quality/` — derived (for orientation)

`cells.npz` (every timestamp × station cell: actual, normal, expected, total, weather, rest, excluded) and `quality.db` (SQLite): `meta`, `stations`, `boundaries` (station × day type × hour: normal median / mean, p01…p99, max, normal band), `station_bounds` (max ever, ceilings = 1.5× / 1.15× max, peak hours, clipping share), `network_bounds`, `episodes` (+ measured uplift), `event_effects`, `coefficients`, `anomalies` (unexplained spikes), `outages` (zero readings), `daily`, `data_issues`. Tools `quality_*`; full description in `docs/ml_preprocessing_pipeline.md`.

## 8. MCP tools (server `nextmove-quality`, `mcp_server/quality_server.py`)

| Tool | Reads | Use it to |
| --- | --- | --- |
| `golden_datasets` | catalog of all files | discover what exists (start here) |
| `golden_describe(name)` | one file | schema, ranges, NaN share, model terms, cache size |
| `golden_series(station, start, end)` | the four wide tables | see actual / normal / total / weather / rest for a station over a period |
| `golden_slice(at, table, top_n, order)` | one wide table | "which stations deviated most at 09:15?" |
| `golden_episodes(date, station, kind)` | `episodes*.csv` | which events / closures were active, their stations |
| `golden_coefficients(station)` | `normal_flow_coefficients.csv` | rain / heat / holiday effect |
| `golden_normal_flow(station, at, periods)` | `normal_flow_model.pkl` | normal flow for any timestamp |
| `golden_model_info` | model | terms, window, grid |
| `golden_venue_station(venue, address)` | geocode cache + overrides | which station serves a venue |
| `golden_geocode_cache(query)` | geocode cache | look at cached coordinates |

In the agent: `await quality_mcp.golden("golden_series", station="Rudow", start="2026-07-14 16:00", end="2026-07-14 20:00")` (in-process MCP client, `agent/quality_mcp.py`); the Inspector uses it for the event check (`Q-EPISODE`) and the derived database for boundaries. External agents connect to `http://127.0.0.1:8768/mcp` (`make up`).

```python
import numpy as np, pandas as pd
total  = pd.read_csv("data/normalized/normalized_flows.csv", index_col=0, parse_dates=True)          # log ratio, 0 = normal
normal = pd.read_csv("data/normalized/normal_flow_passengers.csv", index_col=0, parse_dates=True)     # passengers per 15 min
actual = np.expm1(total + np.log1p(normal))                                                            # ≈ the raw reading, within rounding (section 1)
```

## 9. Caveats — please read before relying on a number

* The flows are **simulated** (organisers); weather explains 0.5 % of the variance; the model's "normal" is a statistical baseline, not a measured truth.
* At 15 minutes the log residual has σ ≈ 1.2: single slots are noisy (readings are clipped at 500 in 1.2 % of cells and are exactly 0 in 0.4 % of well-used slots); judge patterns over hours.
* `normal_flow_passengers` can be slightly negative (≥ −0.2) for small stations at night — treat as 0.
* The **test split is ~10 % above the training normal**; the model is not refitted on it.
* Only events ≥ 2 000 visitors mapped to a station and closures resolved to stations are episodes; smaller events are noise. Event stations come from overrides / OpenStreetMap geocoding (cached) / names; 4 events ≥ 2 000 visitors have no station.
* `SCHOOL_HOLIDAYS` (2026-07-09 → 2026-08-22) in `config.py` was marked "please verify" by the pipeline's authors and is unverified.
