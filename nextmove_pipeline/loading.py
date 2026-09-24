"""
Loading of the raw training data (CSV files in one folder).

Expected files (the suffix, e.g. '_pre_innotrans', is optional, see config.DATA_SUFFIX):
    flows*.csv                    passengers per station and 15-minute slot
    weather_data*.csv             temp, prcp, wspd, ... per 15-minute slot
    stations_with_ubahn.csv       station_id, station_name, longitude, latitude, u_bahn_lines
    berlin_ubahn_connections.csv  station_id_1, station_id_2 (undirected network edges)
    berlin_events_summer_2026*.csv
    closures*.csv                 when, duration, description
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import config


def find_file(data_dir, stem: str) -> Path:
    """'<stem><DATA_SUFFIX>.csv' if it exists, otherwise the first CSV starting with `stem`."""
    d = Path(data_dir)
    exact = d / f"{stem}{config.DATA_SUFFIX}.csv"
    if exact.exists():
        return exact
    hits = sorted(p for p in d.glob(f"{stem}*.csv"))
    if not hits:
        raise FileNotFoundError(f"no file '{stem}*.csv' in {d}")
    if len(hits) > 1:
        print(f"[warn] several files for '{stem}': {[h.name for h in hits]} -> using {hits[0].name}")
    return hits[0]


def load_stations(data_dir) -> pd.DataFrame:
    return pd.read_csv(find_file(data_dir, "stations_with_ubahn"), dtype={"station_id": str})


def load_flow_station_names(data_dir):
    """Only the header of the flows file: the station columns that have passenger data."""
    try:
        header = pd.read_csv(find_file(data_dir, "flows"), nrows=0).columns
    except FileNotFoundError:
        return None
    return [c for c in header if c != "timestamp"]


def load_network(data_dir, stations: pd.DataFrame) -> dict:
    """Adjacency {station_name: set(neighbour names)} from the connections file."""
    con = pd.read_csv(find_file(data_dir, "berlin_ubahn_connections"), dtype=str)
    id2name = dict(zip(stations.station_id, stations.station_name))
    adj = defaultdict(set)
    for a, b in zip(con.station_id_1, con.station_id_2):
        if a in id2name and b in id2name:
            adj[id2name[a]].add(id2name[b])
            adj[id2name[b]].add(id2name[a])
    return dict(adj)


def load_flows(data_dir, station_names=None) -> pd.DataFrame:
    """Flows as DataFrame (index: timestamp, columns: station names), optionally restricted
    to `station_names` (stations without coordinates cannot be placed in the network)."""
    flows = pd.read_csv(find_file(data_dir, "flows"))
    flows["timestamp"] = pd.to_datetime(flows["timestamp"])
    flows = flows.set_index("timestamp").sort_index()
    if station_names is not None:
        keep = set(station_names)
        dropped = [c for c in flows.columns if c not in keep]
        if dropped:
            print(f"[info] {len(dropped)} flow column(s) without station metadata dropped: {dropped}")
        flows = flows[[c for c in flows.columns if c in keep]]
    return flows.astype(float)


def load_weather(data_dir) -> pd.DataFrame:
    w = pd.read_csv(find_file(data_dir, "weather_data"), index_col=0)
    w.index = pd.to_datetime(w.index)
    return w.sort_index()


def load_events(data_dir) -> pd.DataFrame:
    return pd.read_csv(find_file(data_dir, "berlin_events_summer_2026"))


def load_closures(data_dir) -> pd.DataFrame:
    cl = pd.read_csv(find_file(data_dir, "closures"))
    cl["when"] = pd.to_datetime(cl["when"])
    return cl


def resample(flows: pd.DataFrame, weather: pd.DataFrame, freq: str):
    """Coarser time grid: passengers are summed, weather is averaged.
    min_count=1 keeps slots without any data (night pause 1-4 am) as NaN instead of 0."""
    if freq in ("15min", "15T"):
        return flows, weather
    return flows.resample(freq).sum(min_count=1), weather.resample(freq).mean()
