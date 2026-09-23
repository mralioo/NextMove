"""Feature/label engineering for the overcrowding-risk classifier.

Builds one row per (station, 15-min timestamp) with:
  - temporal context (hour, day-of-week, weekend, month)
  - station context (line membership, interchange flag, own ridership baseline)
  - weather at that timestamp
  - city-wide event context for that day (coarse proxy — events aren't geocoded
    to a specific station in the source data, see docs/agentic_system_design.md
    data-gap #1)
  - closure context (whether the station, or a line it serves, is under a
    simulated closure at that timestamp — approximate for line suspensions,
    since the closure only truly affects the named section, not every station
    on the line; see docs/agentic_system_design.md data-gap #3 for the same
    caveat applied elsewhere in this repo)

Label: `overcrowded` = 1 if that station's flow at that timestamp is at or
above the station's OWN historical 90th percentile (no platform-capacity
figure exists in the schema — see data-gap #2 in the design doc — so
"overcrowded" is defined relative to each station's normal operating range).

`passengers` (the raw flow count) doubles as the regression target for the
MCP server's `predict_expected_flow` tool — same feature table, two targets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

from utils.data_loader import (  # noqa: E402
    flows_long,
    load_closures,
    load_events,
    load_stations,
    load_weather,
)

OVERCROWD_QUANTILE = 0.90

# Shared feature list — imported by the training script and the MCP server so
# both stay in sync with what build_feature_table() actually produces.
FEATURE_COLUMNS = [
    "hour", "dow", "is_weekend", "month",
    "station_avg_passengers", "n_lines", "is_interchange", "primary_line",
    "temp", "prcp", "wspd", "cldc", "coco",
    "daily_event_count", "daily_event_attendance",
    "in_closure", "network_active_closures",
]
CATEGORICAL_COLUMNS = ["primary_line"]


def build_feature_table(folder: str) -> pd.DataFrame:
    long = flows_long(folder)  # timestamp, station_name, passengers, date, hour, dow, dow_name, is_weekend
    stations = load_stations(folder)
    weather = load_weather(folder)
    events = load_events(folder)
    closures = load_closures(folder)

    # --- label: station-relative overcrowding threshold ---
    thresholds = long.groupby("station_name")["passengers"].quantile(OVERCROWD_QUANTILE)
    long = long.merge(thresholds.rename("station_p90"), on="station_name")
    long["overcrowded"] = (long["passengers"] >= long["station_p90"]).astype(int)

    # --- station context ---
    # `stations_with_ubahn.csv` models some interchange stations (e.g. "U Stadtmitte
    # (Berlin)") as one row per line, sharing the same station_name. flows.csv has only
    # one column per physical station, so a plain merge on station_name would fan out
    # those rows. Collapse to one metadata row per name first.
    station_meta = (
        stations.groupby("station_name")["u_bahn_lines"]
        .apply(lambda s: ",".join(sorted(set(",".join(s).split(",")))))
        .reset_index()
    )
    station_meta["primary_line"] = station_meta["u_bahn_lines"].str.split(",").str[0]
    station_meta["n_lines"] = station_meta["u_bahn_lines"].str.split(",").apply(len)
    station_meta["is_interchange"] = station_meta["n_lines"] > 1

    station_avg = long.groupby("station_name")["passengers"].mean().rename("station_avg_passengers")
    long = long.merge(station_avg, on="station_name")
    long = long.merge(station_meta, on="station_name", how="left")

    # A pandas-auto-renamed duplicate flow column (e.g. "U Stadtmitte (Berlin).1", see
    # above) won't match any station_meta row — drop those unresolvable rows here so the
    # int casts below don't choke on NaN, rather than silently guessing its line/degree.
    long = long.dropna(subset=["primary_line", "n_lines", "is_interchange"])

    # --- temporal context ---
    long["month"] = pd.to_datetime(long["timestamp"]).dt.month
    long["is_weekend"] = long["is_weekend"].astype(int)
    long["is_interchange"] = long["is_interchange"].astype(int)

    # --- weather context ---
    long = long.merge(
        weather[["timestamp", "temp", "prcp", "wspd", "cldc", "coco"]],
        on="timestamp", how="left",
    )

    # --- event context (city-wide daily proxy; see docstring) ---
    daily_events = (
        events.groupby("date")
        .agg(daily_event_count=("event_name", "count"),
             daily_event_attendance=("estimated_attendance", "sum"))
        .reset_index()
    )
    long = long.merge(daily_events, left_on="date", right_on="date", how="left")
    long["daily_event_count"] = long["daily_event_count"].fillna(0)
    long["daily_event_attendance"] = long["daily_event_attendance"].fillna(0)

    # --- closure context ---
    long["in_closure"] = 0
    long["network_active_closures"] = 0
    ts = long["timestamp"]
    station_lines = long["u_bahn_lines"].fillna("")
    active_count = pd.Series(0, index=long.index)

    for _, c in closures.iterrows():
        in_window = (ts >= c["when"]) & (ts <= c["end"])
        active_count = active_count + in_window.astype(int)
        if c["closure_type"] == "Station closure" and c["affected_segment"]:
            matches = in_window & long["station_name"].str.contains(
                str(c["affected_segment"]).strip(), regex=False, case=False
            )
        elif c["closure_type"] == "Line suspension" and c["affected_line"]:
            matches = in_window & station_lines.apply(
                lambda ls, line=c["affected_line"]: line in [x.strip() for x in ls.split(",")] if ls else False
            )
        else:
            matches = pd.Series(False, index=long.index)
        long.loc[matches, "in_closure"] = 1

    long["network_active_closures"] = active_count

    keep = ["timestamp", "station_name", "passengers", "overcrowded"] + FEATURE_COLUMNS
    return long[keep].dropna(subset=FEATURE_COLUMNS)


def encode_categoricals(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Label-encode CATEGORICAL_COLUMNS in place (returns a copy).

    If `reference` is given, categories are fit on `reference` so a small
    single-row prediction frame gets the same codes the model was trained
    with (pandas' per-call `.astype("category")` would otherwise assign
    fresh, incompatible codes to a 1-row frame).
    """
    out = df.copy()
    basis = reference if reference is not None else df
    for col in CATEGORICAL_COLUMNS:
        categories = pd.Categorical(basis[col]).categories
        out[col] = pd.Categorical(out[col], categories=categories).codes
    return out
