"""Data discovery, loading, parsing, and feature-engineering helpers for the
InnoTrans 2026 Berlin U-Bahn dashboard.

All loaders are cached with st.cache_data so navigating between pages is fast.
File discovery is glob-based (not hardcoded to the `_pre_innotrans` suffix) so
the same app keeps working once the September 22-30 evaluation dataset is
dropped into the `data/` directory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "data")

# Official-ish BVG U-Bahn line colors, used consistently across every chart.
LINE_COLORS = {
    "U1": "#52822f",
    "U2": "#da421e",
    "U3": "#16683d",
    "U4": "#f0d722",
    "U5": "#7e5330",
    "U6": "#8c6dab",
    "U7": "#528dc8",
    "U8": "#224f86",
    "U9": "#f3791d",
}


def discover_dataset_dirs(base_dir: str = DEFAULT_DATA_DIR) -> dict[str, Path]:
    """Find every folder under base_dir that looks like a dataset (contains
    a stations_with_ubahn.csv). Returns {label: path}, newest-looking first.
    """
    base = Path(base_dir)
    candidates: dict[str, Path] = {}
    if not base.exists():
        return candidates
    for hit in base.rglob("stations_with_ubahn.csv"):
        folder = hit.parent
        label = str(folder.relative_to(base)) if folder != base else base.name
        candidates[label] = folder
    if len(candidates) == 1:
        # the test split (`data/testing dataset/*_rest.csv`) reuses the training network files: when data files live outside the one folder that holds the
        # network files, the whole `data/` tree is the dataset (every matching file is merged, see _read_merged)
        only = next(iter(candidates.values()))
        if any(only not in f.parents for f in base.rglob("flows*.csv")):
            return {base.name: base}
    return candidates or {base.name: base}


def _find_one(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern)) or sorted(folder.rglob(pattern))     # a parent of dataset folders works too
    return matches[0] if matches else None


def _find_all(folder: Path, pattern: str) -> list[Path]:
    """Every file matching `pattern` in `folder` — and, when `folder` is a parent of several dataset folders, in them too.
    The Sept 22-30 evaluation data may arrive as an extra file next to the old one or as a second folder; both must be
    MERGED, never 'first file wins' (that silently dropped the history)."""
    return sorted(folder.rglob(pattern))


def _fix_mojibake(name: str) -> str:
    """'U KurfÃ¼rstenstr.' (UTF-8 read as latin-1 by whoever exported the test split) -> 'U Kurfürstenstr.'"""
    for enc in ("cp1252", "latin-1"):                          # ß = C3 9F: 'ÃŸ' only round-trips through cp1252
        try:
            return name.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return name


def _read_one(path: Path, names: list[str] | None, expect: str | None, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", **kw)
    if expect and names and expect not in df.columns:      # a file without a header row (the events test split): use the training file's column names
        df = pd.read_csv(path, encoding="utf-8-sig", header=None, names=names, **kw)
        for c in df.select_dtypes("object").columns:      # this export is also mojibake in its text ('ALIZÃ‰' -> 'ALIZÉ')
            df[c] = df[c].map(lambda v: _fix_mojibake(v) if isinstance(v, str) else v)
    return df.rename(columns={c: _fix_mojibake(c) for c in df.columns if isinstance(c, str)})


def _read_merged(folder: Path, pattern: str, expect: str | None = None, **kw) -> pd.DataFrame:
    """Merge every file matching `pattern`. `expect` = a column name every file WITH a header has; a file without it is read headerless."""
    files = _find_all(folder, pattern)
    if not files:
        raise FileNotFoundError(f"no file matching {pattern} under {folder}")
    names = None
    if expect:
        for f in files:
            cols = list(pd.read_csv(f, encoding="utf-8-sig", nrows=0).columns)
            if expect in cols:
                names = cols
                break
    frames = [_read_one(f, names, expect, **kw) for f in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=False)


@st.cache_data(show_spinner=False)
def load_stations(folder: str) -> pd.DataFrame:
    path = _find_one(Path(folder), "stations_with_ubahn.csv")
    df = pd.read_csv(path)
    df["primary_line"] = df["u_bahn_lines"].str.split(",").str[0].str.strip()
    df["n_lines"] = df["u_bahn_lines"].str.split(",").apply(len)
    df["is_interchange"] = df["n_lines"] > 1
    return df


@st.cache_data(show_spinner=False)
def load_connections(folder: str) -> pd.DataFrame:
    path = _find_one(Path(folder), "berlin_ubahn_connections.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_lines(folder: str) -> pd.DataFrame:
    path = _find_one(Path(folder), "berlin_ubahn_lines_used.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_flows(folder: str) -> pd.DataFrame:
    df = _read_merged(Path(folder), "flows*.csv")
    df["timestamp"] = _parse_mixed_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_weather(folder: str) -> pd.DataFrame:
    df = _read_merged(Path(folder), "weather_data*.csv")
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = _parse_mixed_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_events(folder: str) -> pd.DataFrame:
    df = _read_merged(Path(folder), "berlin_events*.csv", expect="event_name").drop_duplicates(["event_name", "began_local"], keep="last")
    df["began_local"] = pd.to_datetime(df["began_local"], utc=False, format="mixed")
    df["estimated_end_local"] = pd.to_datetime(
        df["estimated_end_local"], utc=False, format="mixed", errors="coerce"
    )
    df["date"] = df["began_local"].dt.date
    return df


_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)min)?")


def _parse_duration(text: str) -> pd.Timedelta:
    m = _DURATION_RE.fullmatch(text.strip())
    hours = int(m.group(1)) if m and m.group(1) else 0
    minutes = int(m.group(2)) if m and m.group(2) else 0
    return pd.Timedelta(hours=hours, minutes=minutes)


_LINE_CLOSURE_RE = re.compile(
    r"Line (U\d+) suspended(?: on a section)? between (.+?) and (.+?) due to (.+)\."
)
_STATION_CLOSURE_RE = re.compile(r"Station (.+?) closed due to (.+)\.")


@st.cache_data(show_spinner=False)
def load_closures(folder: str) -> pd.DataFrame:
    df = _read_merged(Path(folder), "closures*.csv").drop_duplicates(["when", "description"], keep="last").reset_index(drop=True)
    df["when"] = _parse_mixed_datetime(df["when"])
    df["duration_td"] = df["duration"].apply(_parse_duration)
    df["end"] = df["when"] + df["duration_td"]
    df["duration_hours"] = df["duration_td"].dt.total_seconds() / 3600

    kinds, lines, stations, reasons = [], [], [], []
    for desc in df["description"]:
        line_m = _LINE_CLOSURE_RE.match(desc)
        station_m = _STATION_CLOSURE_RE.match(desc)
        if line_m:
            kinds.append("Line suspension")
            lines.append(line_m.group(1))
            stations.append(f"{line_m.group(2)} ↔ {line_m.group(3)}")
            reasons.append(line_m.group(4))
        elif station_m:
            kinds.append("Station closure")
            lines.append(None)
            stations.append(station_m.group(1))
            reasons.append(station_m.group(2))
        else:
            kinds.append("Other")
            lines.append(None)
            stations.append(None)
            reasons.append(desc)
    df["closure_type"] = kinds
    df["affected_line"] = lines
    df["affected_segment"] = stations
    df["reason"] = reasons
    return df


@st.cache_data(show_spinner=False)
def load_energy(folder: str) -> pd.DataFrame:
    df = _read_merged(Path(folder), "energy_consumption*.csv")
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = _parse_mixed_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived / feature-engineered tables
# ---------------------------------------------------------------------------

def station_cols(flows: pd.DataFrame) -> list[str]:
    return [c for c in flows.columns if c != "timestamp"]


@st.cache_data(show_spinner=False)
def flows_long(folder: str) -> pd.DataFrame:
    """Melted flows: timestamp, station_name, passengers — plus calendar parts."""
    flows = load_flows(folder)
    cols = station_cols(flows)
    long = flows.melt(id_vars="timestamp", value_vars=cols,
                       var_name="station_name", value_name="passengers")
    long["date"] = long["timestamp"].dt.date
    long["hour"] = long["timestamp"].dt.hour
    long["dow"] = long["timestamp"].dt.dayofweek
    long["dow_name"] = long["timestamp"].dt.day_name()
    long["is_weekend"] = long["dow"] >= 5
    return long


@st.cache_data(show_spinner=False)
def network_total_flow(folder: str) -> pd.DataFrame:
    flows = load_flows(folder)
    cols = station_cols(flows)
    out = pd.DataFrame({
        "timestamp": flows["timestamp"],
        "total_passengers": flows[cols].sum(axis=1),
    })
    return out


@st.cache_data(show_spinner=False)
def station_avg_flow(folder: str) -> pd.DataFrame:
    flows = load_flows(folder)
    cols = station_cols(flows)
    n_days = flows["timestamp"].dt.date.nunique()
    avg_15min = flows[cols].mean().rename("avg_15min_passengers")
    total = flows[cols].sum().rename("total_passengers")
    daily_avg = (total / max(n_days, 1)).rename("avg_daily_passengers")
    peak = flows[cols].max().rename("peak_15min_passengers")
    out = pd.concat([avg_15min, total, daily_avg, peak], axis=1)
    out.index.name = "station_name"
    return out.reset_index()


@st.cache_data(show_spinner=False)
def line_flow_series(folder: str) -> pd.DataFrame:
    """Approximate per-line passenger flow by summing flows of every station
    that serves that line. Interchange stations are counted on every line
    they serve, so this is a coverage proxy, not an exact per-line count.
    """
    flows = load_flows(folder)
    stations = load_stations(folder)
    cols = station_cols(flows)
    name_to_lines = dict(zip(stations["station_name"], stations["u_bahn_lines"]))

    line_series = {}
    for line in sorted(load_lines(folder)["line_name"].unique()):
        member_cols = [c for c in cols
                       if c in name_to_lines and line in [x.strip() for x in name_to_lines[c].split(",")]]
        if member_cols:
            line_series[line] = flows[member_cols].sum(axis=1)
    out = pd.DataFrame(line_series)
    out.insert(0, "timestamp", flows["timestamp"])
    return out


@st.cache_data(show_spinner=False)
def build_graph(folder: str) -> nx.Graph:
    conn = load_connections(folder)
    g = nx.Graph()
    g.add_edges_from(conn[["station_id_1", "station_id_2"]].itertuples(index=False, name=None))
    return g


@st.cache_data(show_spinner=False)
def network_resilience(folder: str) -> pd.DataFrame:
    """Rank stations by fragmentation impact if removed: articulation-point
    status plus betweenness centrality, joined with average daily ridership.
    """
    g = build_graph(folder)
    stations = load_stations(folder)
    avg_flow = station_avg_flow(folder)

    articulation = set(nx.articulation_points(g))
    betweenness = nx.betweenness_centrality(g)

    id_to_name = dict(zip(stations["station_id"], stations["station_name"]))
    rows = []
    for node in g.nodes():
        # Component sizes if this node were removed.
        h = g.copy()
        h.remove_node(node)
        components = list(nx.connected_components(h)) if h.number_of_nodes() else []
        largest = max((len(c) for c in components), default=0)
        n_components = len(components)
        rows.append({
            "station_id": node,
            "station_name": id_to_name.get(node, node),
            "is_articulation_point": node in articulation,
            "betweenness_centrality": betweenness.get(node, 0.0),
            "resulting_components": n_components,
            "largest_fragment_size": largest,
        })
    out = pd.DataFrame(rows)
    out = out.merge(avg_flow[["station_name", "avg_daily_passengers"]],
                     on="station_name", how="left")
    out["fragmentation_score"] = out["betweenness_centrality"] * out["avg_daily_passengers"].fillna(0)
    return out.sort_values("fragmentation_score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def hourly_profile(folder: str, station_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    long = flows_long(folder)
    if station_names:
        long = long[long["station_name"].isin(station_names)]
    prof = long.groupby(["hour", "is_weekend"], as_index=False)["passengers"].mean()
    return prof


@st.cache_data(show_spinner=False)
def dow_hour_heatmap(folder: str, station_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    long = flows_long(folder)
    if station_names:
        long = long[long["station_name"].isin(station_names)]
    pivot = long.pivot_table(index="dow_name", columns="hour", values="passengers", aggfunc="mean")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in order if d in pivot.index])
    return pivot
