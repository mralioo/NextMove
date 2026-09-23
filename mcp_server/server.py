"""FastMCP server exposing the Berlin U-Bahn dataset + TabPFN prediction
tools to any MCP client (the ADK agent in `agent/`, or any other).

Scope right now: **Category D — station profiling** (the blueprint's "fast
path", e.g. training question 4: "At what time does commute flow peak at
Rudow station, does it exceed the network mean?"), plus two prediction tools
(`predict_overcrowding_risk` classification, `predict_expected_flow`
regression) that demonstrate the TabPFN integration the wider design calls
for. Categories A/B/C/E/F/G/H from docs/agentic_system_design.md are not
wired up yet — this is the first vertical slice, not the full blueprint.

Run standalone (stdio transport, for the ADK agent to spawn):
    ./.venv/bin/python mcp_server/server.py

All dataset tools are read-only and deterministic (they reuse
dashboard/utils/data_loader.py — the same code path the Streamlit dashboard
uses, so the agent and the dashboard can never silently disagree). The two
predict_* tools call the TabPFN API and require TABPFN_API_TOKEN (see
ml/train_overcrowding_classifier.py's docstring for where to put it).
"""
from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from env_loader import load_all_dotenvs  # noqa: E402
from features import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_feature_table,
    encode_categoricals,
)
from utils.data_loader import (  # noqa: E402
    DEFAULT_DATA_DIR,
    discover_dataset_dirs,
    hourly_profile,
    load_stations,
    station_avg_flow,
)

load_all_dotenvs()

mcp = FastMCP(
    "ubahn-flow-data",
    instructions=(
        "Read-only access to the Berlin U-Bahn training dataset (stations, passenger "
        "flow, weather, events, closures, energy) plus TabPFN-backed prediction tools. "
        "Only historical timestamps within the loaded dataset's coverage window are "
        "supported — this server does not forecast beyond the data it holds. "
        "Always call resolve_station first if you're not certain of the exact station "
        "name; station names are exact strings like 'U Rudow (Berlin)', not just 'Rudow'."
    ),
)

_FOLDER = str(next(iter(discover_dataset_dirs(DEFAULT_DATA_DIR).values())))

# Lazily-built, in-process caches. Rebuilding the ~1.4M-row feature table or
# re-fitting a TabPFN model on every tool call would be slow and (for TabPFN)
# needlessly burn API quota, so each is computed once per server process.
_feature_table_cache: pd.DataFrame | None = None
_model_cache: dict[str, object] = {}
_network_mean_weekday_peak_cache: float | None = None


def _get_feature_table() -> pd.DataFrame:
    global _feature_table_cache
    if _feature_table_cache is None:
        _feature_table_cache = build_feature_table(_FOLDER)
    return _feature_table_cache


def _all_station_names() -> list[str]:
    return sorted(load_stations(_FOLDER)["station_name"].unique())


@mcp.tool
def describe_dataset() -> dict:
    """Dataset coverage: date range, station/line counts, and which analysis
    categories this server currently supports. Call this first if you're
    unsure whether a question is in scope."""
    stations = load_stations(_FOLDER)
    table = _get_feature_table()
    return {
        "dataset_folder": _FOLDER,
        "coverage_start": str(table["timestamp"].min()),
        "coverage_end": str(table["timestamp"].max()),
        "n_stations": int(stations["station_name"].nunique()),
        "n_lines": int(stations["u_bahn_lines"].str.split(",").explode().str.strip().nunique()),
        "supported_categories": [
            "D: station profiling (station_profile) — peak hour, weekday/weekend rhythm, "
            "network-mean benchmark",
            "predict: overcrowding-risk classification and expected-flow regression via "
            "TabPFN, for any station+timestamp inside the coverage window above",
        ],
        "not_yet_supported": (
            "Categories A/B/C/E/F/G/H (event impact, anomaly root-cause, disruption "
            "rerouting, energy efficiency, network resilience, latent correlation, "
            "reroute behavior) — see docs/agentic_system_design.md"
        ),
    }


@mcp.tool
def list_stations() -> list[dict]:
    """List every station with its serving line(s). Use resolve_station
    instead if you only have a rough/partial name from the user."""
    stations = load_stations(_FOLDER)
    return stations[["station_name", "u_bahn_lines"]].drop_duplicates().to_dict("records")


@mcp.tool
def resolve_station(query: str, max_results: int = 5) -> list[dict]:
    """Fuzzy-resolve a user-typed station reference (e.g. 'Rudow', 'Alexanderplatz')
    to exact station_name strings the other tools require. Returns substring
    matches first, then closest spelling matches, most likely first."""
    names = _all_station_names()
    query_lower = query.strip().lower()
    substring_hits = [n for n in names if query_lower in n.lower()]
    fuzzy_hits = difflib.get_close_matches(query, names, n=max_results, cutoff=0.4)
    ordered = list(dict.fromkeys(substring_hits + fuzzy_hits))[:max_results]
    if not ordered:
        return [{"station_name": None, "note": f"No station resembling '{query}' found."}]
    return [{"station_name": n} for n in ordered]


@mcp.tool
def station_profile(station_name: str) -> dict:
    """Category D — station profiling. Given an EXACT station_name (use
    resolve_station first if unsure), returns: average daily ridership, the
    weekday commute peak hour + value, the weekend peak hour + value, and how
    the weekday peak compares to the mean weekday peak across all stations
    network-wide. This directly answers questions shaped like training
    question 4 ('when does station X's commute peak occur, does it exceed
    the network average?')."""
    table = _get_feature_table()
    if station_name not in set(table["station_name"]):
        return {"error": f"Unknown station_name '{station_name}'. Call resolve_station first."}

    avg_flow = station_avg_flow(_FOLDER)
    row = avg_flow[avg_flow["station_name"] == station_name].iloc[0]

    prof = hourly_profile(_FOLDER, (station_name,))
    weekday_prof = prof[~prof["is_weekend"]].set_index("hour")["passengers"]
    weekend_prof = prof[prof["is_weekend"]].set_index("hour")["passengers"]
    weekday_peak_hour = int(weekday_prof.idxmax())
    weekday_peak_val = float(weekday_prof.max())
    weekend_peak_hour = int(weekend_prof.idxmax())
    weekend_peak_val = float(weekend_prof.max())

    network_mean_weekday_peak = _get_network_mean_weekday_peak()

    return {
        "station_name": station_name,
        "avg_daily_passengers": float(row["avg_daily_passengers"]),
        "weekday_peak_hour": weekday_peak_hour,
        "weekday_peak_avg_passengers": weekday_peak_val,
        "weekend_peak_hour": weekend_peak_hour,
        "weekend_peak_avg_passengers": weekend_peak_val,
        "network_mean_weekday_peak_passengers": network_mean_weekday_peak,
        "exceeds_network_mean_weekday_peak": weekday_peak_val > network_mean_weekday_peak,
        "delta_vs_network_mean_pct": round(
            (weekday_peak_val - network_mean_weekday_peak) / network_mean_weekday_peak * 100, 1
        ),
        "method": "hour-of-day average across the full dataset window, weekday vs weekend split",
        "evidence": {
            "dataset_folder": _FOLDER,
            "tool": "flows.station_profile",
        },
    }


def _get_network_mean_weekday_peak() -> float:
    """Mean, across ALL stations, of each station's own weekday peak *hourly
    average* passenger count. Cached — same value for every station_profile
    call, and the underlying groupby is O(1.4M rows)."""
    global _network_mean_weekday_peak_cache
    if _network_mean_weekday_peak_cache is None:
        table = _get_feature_table()
        weekday = table[~table["is_weekend"].astype(bool)]
        per_station_hourly_peak = (
            weekday.groupby(["station_name", "hour"])["passengers"].mean()
            .groupby("station_name").max()
        )
        _network_mean_weekday_peak_cache = float(per_station_hourly_peak.mean())
    return _network_mean_weekday_peak_cache


def _get_train_pool_dates() -> tuple[set, object]:
    """Chronological train-pool dates, matching ml/train_overcrowding_classifier.py's
    80/20 split, so predict tools can flag whether a queried row was inside the
    model's training window (potential memorization) or genuinely held out."""
    table = _get_feature_table()
    dates = sorted(table["timestamp"].dt.date.unique())
    cutoff = dates[int(len(dates) * 0.8)]
    return {d for d in dates if d < cutoff}, cutoff


def _fit_models() -> dict:
    """Fit (once) a TabPFNClassifier on `overcrowded` and a TabPFNRegressor on
    `passengers`, both on the same chronological-train-pool stratified sample
    used by ml/train_overcrowding_classifier.py, so a live prediction here is
    directly comparable to that script's offline evaluation."""
    if _model_cache:
        return _model_cache

    token = os.environ.get("TABPFN_API_TOKEN")
    if not token:
        raise RuntimeError(
            "TABPFN_API_TOKEN not set. Add it to a .env file above this repo "
            "(see ml/train_overcrowding_classifier.py docstring)."
        )
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

    tabpfn_client.set_access_token(token)

    table = _get_feature_table()
    train_dates, cutoff = _get_train_pool_dates()
    train_pool = table[table["timestamp"].dt.date.isin(train_dates)]

    # Same stratified-sample size/strategy as the offline training script.
    n = min(8000, len(train_pool))
    frac_pos = train_pool["overcrowded"].mean()
    n_pos = min(int(round(n * max(frac_pos, 0.15))), (train_pool["overcrowded"] == 1).sum())
    n_neg = min(n - n_pos, (train_pool["overcrowded"] == 0).sum())
    sample = pd.concat([
        train_pool[train_pool["overcrowded"] == 1].sample(n_pos, random_state=0),
        train_pool[train_pool["overcrowded"] == 0].sample(n_neg, random_state=0),
    ]).sample(frac=1, random_state=0).reset_index(drop=True)

    X = encode_categoricals(sample[FEATURE_COLUMNS])
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]

    clf = TabPFNClassifier(model_path="v3.5_default", categorical_features_indices=cat_idx)
    clf.fit(X, sample["overcrowded"])

    reg = TabPFNRegressor(model_path="v3.5_default", categorical_features_indices=cat_idx)
    reg.fit(X, sample["passengers"])

    _model_cache["classifier"] = clf
    _model_cache["regressor"] = reg
    _model_cache["train_sample"] = sample
    _model_cache["cutoff_date"] = cutoff
    return _model_cache


def _seen_in_training_sample(models: dict, station_name: str, timestamp: str) -> bool:
    """Whether this exact (station, timestamp) row was literally part of the
    ~8,000-row sample TabPFN conditioned on — the precise leakage flag,
    stronger than just checking whether the date is pre-cutoff."""
    sample = models["train_sample"]
    ts = pd.to_datetime(timestamp)
    return bool(((sample["station_name"] == station_name) & (sample["timestamp"] == ts)).any())


def _lookup_query_row(station_name: str, timestamp: str) -> tuple[pd.Series | None, str | None]:
    table = _get_feature_table()
    ts = pd.to_datetime(timestamp)
    match = table[(table["station_name"] == station_name) & (table["timestamp"] == ts)]
    if match.empty:
        return None, (
            f"No data for station_name='{station_name}' at timestamp='{timestamp}'. "
            "Timestamps must be exact 15-minute marks inside the dataset's coverage "
            "window (call describe_dataset for the range); this server only supports "
            "historical replay, not forecasting beyond the loaded data."
        )
    return match.iloc[0], None


@mcp.tool
def predict_overcrowding_risk(station_name: str, timestamp: str) -> dict:
    """TabPFN classification: probability that this station's flow at this
    timestamp is at/above its own historical 90th percentile ('overcrowded' —
    a relative demand-pressure proxy, NOT a measured safe-capacity limit; see
    docs/agentic_system_design.md data-gap #2). `timestamp` must be an exact
    15-minute mark inside the dataset's coverage window, e.g.
    '2026-07-15 08:00:00' — call describe_dataset for the range and
    resolve_station for the exact station_name first."""
    row, err = _lookup_query_row(station_name, timestamp)
    if err:
        return {"error": err}

    try:
        models = _fit_models()
    except RuntimeError as e:
        return {"error": str(e)}
    X_row = encode_categoricals(
        pd.DataFrame([row[FEATURE_COLUMNS]]), reference=models["train_sample"][FEATURE_COLUMNS]
    )
    proba = float(models["classifier"].predict_proba(X_row)[0, 1])
    seen = _seen_in_training_sample(models, station_name, timestamp)

    return {
        "station_name": station_name,
        "timestamp": timestamp,
        "overcrowding_probability": round(proba, 3),
        "predicted_label": "overcrowded" if proba >= 0.5 else "normal",
        "actual_label": "overcrowded" if int(row["overcrowded"]) == 1 else "normal",
        "actual_passengers": int(row["passengers"]),
        "data_mode": "historical_replay",
        "seen_in_training_sample": seen,
        "evaluation_note": (
            "This exact row was in the ~8,000-row sample the model conditioned on — "
            "treat this as an in-sample sanity check, not a held-out evaluation. Use "
            "ml/train_overcrowding_classifier.py for a properly held-out accuracy report."
            if seen
            else "This row was not in the training sample (either post-cutoff, or "
            "simply not drawn in the stratified sample)."
        ),
        "model": "TabPFNClassifier(v3.5_default)",
    }


@mcp.tool
def predict_expected_flow(station_name: str, timestamp: str) -> dict:
    """TabPFN regression: expected passenger count at this station and
    timestamp, given time-of-day/weekday, weather, event, and closure context.
    Same input contract as predict_overcrowding_risk (exact station_name,
    exact 15-minute timestamp inside the dataset's coverage window)."""
    row, err = _lookup_query_row(station_name, timestamp)
    if err:
        return {"error": err}

    try:
        models = _fit_models()
    except RuntimeError as e:
        return {"error": str(e)}
    X_row = encode_categoricals(
        pd.DataFrame([row[FEATURE_COLUMNS]]), reference=models["train_sample"][FEATURE_COLUMNS]
    )
    pred = float(models["regressor"].predict(X_row)[0])
    actual = int(row["passengers"])
    seen = _seen_in_training_sample(models, station_name, timestamp)

    return {
        "station_name": station_name,
        "timestamp": timestamp,
        "predicted_passengers": round(pred, 1),
        "actual_passengers": actual,
        "absolute_error": round(abs(pred - actual), 1),
        "data_mode": "historical_replay",
        "seen_in_training_sample": seen,
        "evaluation_note": (
            "In-sample sanity check — this row was in the training sample"
            if seen
            else "Held-out prediction — this row was not in the training sample"
        ),
        "model": "TabPFNRegressor(v3.5_default)",
    }


if __name__ == "__main__":
    mcp.run()
