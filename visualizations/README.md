# visualizations

Plot scripts based on the raw data and on the normalized tables from `nextmove_pipeline`.
One folder per topic; each folder has a `plot.py` and writes its PNGs into its own `output/`.

## Prerequisite

The normalized tables must exist (default location `data/normalized/`):

```bash
python -m nextmove_pipeline.pipeline --data ./training_dataset --freq 1h
```

## Run (from the project root, venv active)

```bash
python -m visualizations.make_all                      # everything
python -m visualizations.raw_flow.plot [--station "Alexanderplatz"]
python -m visualizations.weather_effect.plot
python -m visualizations.event_effect.plot [--event "Guns"] [--table rest]
```

All scripts also accept `--data` (raw CSVs), `--normalized` (pipeline output) and `--out`.

## Folders and figures

| folder | data used | figures |
|---|---|---|
| `raw_flow/` | raw `flows*.csv` | network total per day · average day by day type · busiest stations · one station for one week (`--station`) |
| `weather_effect/` | `normalized_flows.csv` (+ `normalized_weather.csv`, `normal_flow_coefficients.csv`) | flow vs. rain intensity · flow vs. temperature (dry hours) · daily rain and deviation over time · rain sensitivity per station |
| `event_effect/` | `normalized_flows.csv` (or `--table rest`), `episodes.csv`, `normal_flow_passengers.csv` | profile around events (aligned to start and end, by distance) · peak per event · attendance vs. peak · profile around closures · one event in detail (`--event`) |
| `common.py` | – | shared paths, loading, colors, chart style |

## How to read the numbers

- Normalized tables are **log differences from normal**. The weather and closure plots show them
  as % (`+20 %`), the event plots as a factor on a log axis (`8×` normal), because event peaks
  range from 2× to almost 90×.
- "Normal" = typical flow for that station, time of day and day type, with typical weather.
- Network values in the weather plots use the **median over stations**. The mean is pulled
  down about 5 % by time slots with 0 passengers.
- Weather plots show the observed normalized flow next to the **model weather part**: where the
  two bars agree, the model explains the weather effect. The model treats rain as on/off, so it
  overstates the effect of light rain and understates heavy rain.
- Event plots merge duplicate ticket listings of the same show (same station, same start time).

## Colors

The palette is colorblind-safe (blue / orange / aqua, checked with a palette validator). Lines are also
labeled directly or have a legend, so no chart depends on color alone. Red is used only for "less than normal".
