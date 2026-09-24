# nextmove_pipeline

Raw training data -> three normalized tables (total / weather / rest), plus an
address -> nearest-station lookup. Run everything from the project root with the venv active.

## 1. Nearest stations for an address

```bash
python -m nextmove_pipeline.geo "Friedrichstraße 101" --venue Admiralspalast -k 3
python -m nextmove_pipeline.geo "Uber-Platz 1" --offline        # cache only, no network
```

From Python:

```python
from nextmove_pipeline.geo import nearest_stations_for_address
nearest_stations_for_address("Alexanderplatz 1", k=2)
# {'location': {'lat': .., 'lon': ..}, 'stations': [{'station_name': 'S+U Alexanderplatz Bhf (Berlin)',
#   'u_bahn_lines': ['U2', 'U5', 'U8'], 'distance_m': 75.4, 'has_flow_data': True}, ...]}
```

Geocoding uses OpenStreetMap Nominatim (max. 1 request/s). Each result, including "not found",
is stored in `data/processed/geocode_cache_pipeline.json`. The older caches
(`analysis_output/geocode_cache.json`, `data/processed/geocode_cache.json`) are only read, never
changed. Old entries that were really just a hit for "Berlin, Germany" (the city centre) are ignored.

## 2. Normalization pipeline

```bash
python -m nextmove_pipeline.pipeline --data ./training_dataset --freq 1h
# new data, same "normal" (no refit):
python -m nextmove_pipeline.pipeline --data ./new_data --model data/normalized/normal_flow_model.pkl --suffix innotrans
```

Output in `data/normalized/` (all log scale: 0 = normal, 0.1 ~ +10 %, 0.69 ~ twice as many):

| file | content |
|---|---|
| `normalized_flows.csv` | total = log1p(actual) - normal |
| `normalized_weather.csv` | part explained by rain / heat / wind |
| `normalized_rest.csv` | total - weather: events, closures, anomalies |
| `normal_flow_passengers.csv` | normal flow in passengers (to say "+1,200 people") |
| `normal_flow_coefficients.csv` | rain / heat / holiday effect per station in % |
| `episodes.csv` | events and closures with their stations (excluded from the fit) |
| `normal_flow_model.pkl` | fitted model for `--model` |

## Modules

| module | job |
|---|---|
| `config.py` | all thresholds, paths, holidays, manual venue -> station overrides |
| `loading.py` | read the raw CSV files, resample to the time grid |
| `names.py` | match station names ("Spichernstr." == "U Spichernstr. (Berlin)") |
| `geo.py` | geocoding with cache, nearest stations, CLI |
| `episodes.py` | events -> venue station; closures -> **every station on the closed section** |
| `baseline.py` | normal-flow model, decomposition, coefficient table |
| `pipeline.py` | end-to-end run, CLI |

## Differences to the old `normalize_flows.py`

- Standalone: no import of `event_effects_mcp.py` (so no `mcp` dependency).
- A closed line section counts every station between A and B on that line, not just the
  two end points. In the closure check, the rest during a closure goes from -0.14 (62 % negative)
  to -0.24 (73 % negative).
- Manual venue overrides are checked before geocoding, and use exact station names.
- "Rain in the previous hour" is based on time, so it means the same at 15 min and 1 h.
