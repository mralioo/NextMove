# `ml/` — models, disruption logic, normalization, quality database

**Purpose.** The numerical core under the tools: the TabPFN models, the closure graph logic, the normalization pipeline that separates what is *normal* from weather and from events / closures, the quality database with plausibility boundaries, and read access to the pre-processed ("golden") files.

| File | Responsibility |
| --- | --- |
| `features.py`, `table_cache.py` | Station × 15-minute feature table (time, station context, weather, event, closure context; ~1.4 M rows), cached as parquet keyed by a fingerprint of the inputs |
| `train_overcrowding_classifier.py`, `inference_models.py` | TabPFN overcrowding-risk classifier and expected-flow regressor (fitted on a stratified 8 000-row sample of the first 80 % of days) behind `predict_overcrowding_risk` / `predict_expected_flow` |
| `demand_baseline.py`, `train_disruption_baseline.py` | TabPFN **counterfactual demand baseline** for Category C: predictive distribution of what a station would carry if nothing were wrong; evaluated on the recorded closures (`ml/output/`) |
| `disruption.py` | Deterministic closure logic: `resolve_closure` → `apply_closure` (blocked edges, unserved stations) → `alternate_paths` (rail detours, transfer stations) |
| `scenario.py` | `scenario_flow`: baseline + diverted demand (25 / 50 / 75 %) → per-station pressure, with `at_risk` flags |
| `checkpoints.py`, `save_checkpoints.py` | TabPFN "checkpoints": the server-side `model_id` + hyper-parameters, so inference never needs a fresh fit; `ml/checkpoints/manifest.json` |
| `nextmove_pipeline/` | **Normalization pipeline**: `loading.py` (CSV loading), `names.py` (station-name matching), `geo.py` (venue → nearest stations, geocode cache), `episodes.py` (events / closures → time windows + affected stations), `baseline.py` (ridge "normal flow" model per station, decomposition `total = weather + rest`), `pipeline.py` (end to end), `config.py` (thresholds, holidays, overrides) |
| `quality_db.py` | Builds and serves the **quality database** (`data/quality/`): boundaries per station × day type × hour, station ceilings, network bounds, episodes with effect, anomalies, outages, data issues; `check_facts()` for the Inspector |
| `golden_data.py` | Read-only access to `data/normalized/*` and `data/processed/*` (used by the `golden_*` tools) |

## Normalization pipeline in one paragraph

Per station, a ridge regression on `log1p(flow)` with time-of-day × day-type levels, school holidays, rain (now and previous hour), temperature (linear + quadratic), wind and a linear trend is fitted on **clean cells only** (every station within a few hops of an event or closure is excluded from 3 h before to 2 h after). Then `normal` = prediction with typical weather, `expected` = with actual weather; `total = log1p(actual) − normal`, `weather = expected − normal`, `rest = log1p(actual) − expected`, re-centred per station × slot × day type on the median of the clean rest. It is **fitted on the training split** and **applied to the testing split** (the organisers' held-out days). Details, validation and caveats: [`../ml_preprocessing_pipeline.md`](../ml_preprocessing_pipeline.md); file schemas: [`../../data/data_schema_high_quality.md`](../../data/data_schema_high_quality.md).

## Commands

```bash
./.venv/bin/python scripts/tasks.py quality-build [--offline]    # pipeline (fit on training, apply to test) + quality database  (make up runs it once, ~30 s)
./.venv/bin/python scripts/tasks.py quality-status
./.venv/bin/python scripts/tasks.py quality-pipeline --all       # normalization pipeline only → data/normalized
./.venv/bin/python scripts/tasks.py checkpoints [--force]        # TabPFN checkpoints
./.venv/bin/python scripts/tasks.py train-overcrowding | train-disruption | validate-pressure
```

## Configuration

`TABPFN_API_TOKEN` (TabPFN API), `ML_CACHE_DIR`, `CHECKPOINT_DIR`, `DATA_DIR`, `SCENARIO_ENGINE`; pipeline thresholds in `ml/nextmove_pipeline/config.py`; geocoding: `--offline` uses only `data/processed/geocode_cache_pipeline.json` (no Nominatim request).

## Tests

`tests/test_disruption.py`, `tests/test_checkpoints.py`, `tests/test_quality_pipeline.py` (synthetic data for the pipeline; the database tests run on the built database).

## Limits

Flows are **simulated**; a model that fits them shows how the simulation behaves, not real Berlin ridership. Only 26–30 closures exist, so the disruption model predicts a *baseline*, not a "closure uplift": the diversion share is a stated assumption. TabPFN runs on the provider's servers (rows are uploaded when a model is fitted — checkpoints avoid re-fitting). The energy-per-passenger figure is a proxy (interchange stations count for every line they serve).
