"""
End-to-end pipeline: raw training data -> the three normalized tables.

    cd ml && python -m nextmove_pipeline.pipeline --all --freq 15min          # training split (fit) + test split (same model)
    cd ml && python -m nextmove_pipeline.pipeline --data "../data/training dataset" --freq 1h
    cd ml && python -m nextmove_pipeline.pipeline --data ./new_data --model ../data/normalized/normal_flow_model.pkl

Steps
    1) load raw data (flows, weather, stations, network, events, closures)
    2) map events and closures to stations            (episodes.py, geo.py)
    3) fit the normal-flow model on clean cells       (baseline.py)   -- skipped with --model
    4) decompose every cell into weather part + rest  (baseline.py)
    5) write the outputs

Outputs (in --out, default data/normalized/; --suffix appends e.g. '_innotrans')
    normalized_flows.csv            total   = log1p(actual) - normal
    normalized_weather.csv          weather = part explained by rain / heat / wind
    normalized_rest.csv             rest    = total - weather (events, closures, anomalies)
    normal_flow_passengers.csv      normal flow in passengers (to turn log values into "+1,200 people")
    normal_flow_coefficients.csv    rain / heat / holiday effect per station in %
    episodes.csv                    events and closures with their stations (what the fit excluded)
    normal_flow_model.pkl           fitted model, to normalize new data with the same "normal"
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from . import config
from .baseline import coefficient_table, decompose, fit_normal_model
from .episodes import build_episodes, episodes_to_frame
from .geo import Geocoder
from .loading import (load_closures, load_events, load_flows, load_network, load_stations,
                      load_weather, resample)


def run(data_dir, out_dir, freq="15min", model_path=None, suffix="", offline=False,
        geocode_cache=config.DEFAULT_GEOCODE_CACHE):
    """Run the pipeline and return the dict of result tables (also written to out_dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{suffix}" if suffix else ""

    # ---------------------------------------------------------------- 1) load
    stations = load_stations(data_dir)
    flows = load_flows(data_dir, stations.station_name)
    weather = load_weather(data_dir)

    # ---------------------------------------------------------------- 3) fit or load model
    adj = load_network(data_dir, stations)
    if model_path:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        freq = model["freq"]    # new data must use the same time grid as the training data
        flows, weather = resample(flows, weather, freq)
        print(f"[info] model loaded: {model_path} (grid {freq})")
    else:
        flows, weather = resample(flows, weather, freq)
    # ---------------------------------------------------------------- 2) episodes (also for new data: the quality database shows which events / closures were active)
    geocoder = Geocoder(geocode_cache, offline=offline)
    episodes = build_episodes(load_events(data_dir), load_closures(data_dir), stations, list(flows.columns), adj, geocoder)
    episodes_to_frame(episodes).to_csv(out_dir / f"episodes{sfx}.csv", index=False)
    if not model_path:
        model = fit_normal_model(flows, weather, adj, episodes, freq)
        with open(out_dir / f"normal_flow_model{sfx}.pkl", "wb") as f:
            pickle.dump(model, f)
        coef = coefficient_table(model)
        coef.to_csv(out_dir / f"normal_flow_coefficients{sfx}.csv", index=False)
        print(f"[info] mean temperature {model['stats']['temp'][0]:.1f} C; "
              f"network-wide effects (median over stations, %):")
        print(coef.drop(columns="station").median().round(1).to_string())

    # ---------------------------------------------------------------- 4) decompose
    tables = decompose(model, flows, weather)

    # ---------------------------------------------------------------- 5) write
    files = {"total": "normalized_flows", "weather": "normalized_weather",
             "rest": "normalized_rest", "normal_passengers": "normal_flow_passengers"}
    for key, name in files.items():
        tables[key].to_csv(out_dir / f"{name}{sfx}.csv")

    t, r = tables["total"].to_numpy(), tables["rest"].to_numpy()
    print(f"[info] {t.shape[0]} timestamps x {t.shape[1]} stations -> {out_dir}")
    print(f"[info] median total {np.nanmedian(t):+.3f}, median rest {np.nanmedian(r):+.3f}, "
          f"weather explains {1 - np.nanvar(r) / np.nanvar(t):.1%} of the variance of the total")
    return tables


def run_all(out_dir=config.DEFAULT_OUT_DIR, freq="15min", offline=False, geocode_cache=config.DEFAULT_GEOCODE_CACHE):
    """The full build used by the quality database: fit the normal-flow model on the TRAINING split, then normalize the TEST split with that same model
    (suffix `_test`). Returns (training tables, test tables)."""
    train = run(config.DEFAULT_DATA_DIR, out_dir, freq, None, "", offline, geocode_cache)
    test = run(config.TEST_DATA_DIR, out_dir, freq, Path(out_dir) / "normal_flow_model.pkl", "test", offline, geocode_cache)
    return train, test


def main():
    ap = argparse.ArgumentParser(description="Build normalized flow / weather / rest tables.")
    ap.add_argument("--data", default=str(config.DEFAULT_DATA_DIR), help="folder with the raw CSV files")
    ap.add_argument("--out", default=str(config.DEFAULT_OUT_DIR), help="output folder")
    ap.add_argument("--freq", default="15min", help="time grid, e.g. 15min or 1h (1h = much less noise)")
    ap.add_argument("--model", help="existing normal_flow_model.pkl: normalize new data, do not refit")
    ap.add_argument("--suffix", default="", help="suffix for output file names, e.g. innotrans")
    ap.add_argument("--geocode-cache", default=str(config.DEFAULT_GEOCODE_CACHE))
    ap.add_argument("--offline", action="store_true", help="never query Nominatim, use caches only")
    ap.add_argument("--all", action="store_true", help="fit on the training split and normalize the test split with that model (what the quality database uses)")
    a = ap.parse_args()
    if a.all:
        run_all(a.out, a.freq, a.offline, a.geocode_cache)
        return
    run(a.data, a.out, a.freq, a.model, a.suffix, a.offline, a.geocode_cache)


if __name__ == "__main__":
    main()
