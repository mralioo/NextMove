"""Ground truth for the answerable training questions, computed from the RAW CSV files with plain
pandas — deliberately NOT through the agent's own tools, so a bug in a tool cannot vouch for itself."""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def _find(pattern: str) -> str:
    hits = glob.glob(str(REPO / "data" / "**" / pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(pattern)
    return sorted(hits)[0]


def closure_truth(station_a: str, station_b: str) -> dict:
    """Reason / start / end / duration of the closure whose description names both stations."""
    df = pd.read_csv(_find("closures*.csv"))
    hit = df[df["description"].str.contains(station_a, regex=False) & df["description"].str.contains(station_b, regex=False)]
    row = hit.iloc[0]
    start = pd.to_datetime(row["when"], format="mixed")
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)min)?", row["duration"].strip())
    hours = int(m.group(1) or 0) + int(m.group(2) or 0) / 60
    return {"reason": re.search(r"due to (.+)\.", row["description"]).group(1),
            "start": start.strftime("%H:%M"), "end": (start + pd.Timedelta(hours=hours)).strftime("%H:%M"),
            "date": start.strftime("%Y-%m-%d"), "hours": hours}


def commute_peak_truth(station_column: str) -> dict:
    """Weekday hour with the highest average flow at `station_column`, the network mean of every station's own
    weekday peak, and whether the station exceeds it (all from flows.csv)."""
    flows = pd.read_csv(_find("flows*.csv"))
    ts = pd.to_datetime(flows["timestamp"], format="mixed")
    weekday = flows[ts.dt.dayofweek < 5].drop(columns="timestamp")
    hourly = weekday.groupby(ts[ts.dt.dayofweek < 5].dt.hour).mean()          # hour x station
    per_station_peak = hourly.max()
    return {"peak_hour": int(hourly[station_column].idxmax()), "peak_value": float(hourly[station_column].max()),
            "network_mean_peak": float(per_station_peak.mean()),
            "exceeds": bool(hourly[station_column].max() > per_station_peak.mean())}
