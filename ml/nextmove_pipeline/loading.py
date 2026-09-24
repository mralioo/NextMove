"""
Loading of the raw training data (CSV files in one folder).

Expected files (training split: *_pre_innotrans.csv, test split: *_rest.csv; the test split has no header row in the events file and mojibake in column names — both handled here):
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


def _fix_mojibake(text):
    """'KurfÃ¼rstenstr.' (UTF-8 read as cp1252 by the exporter of the test split) -> 'Kurfürstenstr.'; text without the artefact is returned unchanged."""
    if not isinstance(text, str):
        return text
    for enc in ("cp1252", "latin-1"):
        try:
            return text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def _read_csv(path, **kw) -> pd.DataFrame:
    """utf-8 with BOM, mojibake column names repaired."""
    df = pd.read_csv(path, encoding="utf-8-sig", **kw)
    return df.rename(columns={c: _fix_mojibake(c) for c in df.columns})


def find_file(data_dir, stem: str, fallback_dir=None) -> Path:
    """The raw file of `stem` in `data_dir`: '<stem>_pre_innotrans.csv' (training split), then '<stem>_rest.csv' (test split), then the first CSV starting with `stem`.
    Files the test split does not ship (stations, network) are taken from `fallback_dir` (default: the training folder)."""
    d = Path(data_dir)
    for suffix in (config.DATA_SUFFIX, config.TEST_SUFFIX):
        exact = d / f"{stem}{suffix}.csv"
        if exact.exists():
            return exact
    hits = sorted(p for p in d.glob(f"{stem}*.csv"))
    if not hits and d.resolve() != config.DEFAULT_DATA_DIR.resolve():
        return find_file(fallback_dir or config.DEFAULT_DATA_DIR, stem)
    if not hits:
        raise FileNotFoundError(f"no file '{stem}*.csv' in {d}")
    if len(hits) > 1:
        print(f"[warn] several files for '{stem}': {[h.name for h in hits]} -> using {hits[0].name}")
    return hits[0]


def load_stations(data_dir) -> pd.DataFrame:
    return _read_csv(find_file(data_dir, "stations_with_ubahn"), dtype={"station_id": str})


def load_flow_station_names(data_dir):
    """Only the header of the flows file: the station columns that have passenger data."""
    try:
        header = _read_csv(find_file(data_dir, "flows"), nrows=0).columns
    except FileNotFoundError:
        return None
    return [c for c in header if c != "timestamp"]


def load_network(data_dir, stations: pd.DataFrame) -> dict:
    """Adjacency {station_name: set(neighbour names)} from the connections file."""
    con = _read_csv(find_file(data_dir, "berlin_ubahn_connections"), dtype=str)
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
    flows = _read_csv(find_file(data_dir, "flows"))
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
    w = _read_csv(find_file(data_dir, "weather_data"), index_col=0)
    w.index = pd.to_datetime(w.index)
    return w.sort_index()


def load_events(data_dir) -> pd.DataFrame:
    """The events file. The test split ships it WITHOUT a header row (and mojibake in the text): the training file names the columns."""
    path = find_file(data_dir, "berlin_events_summer_2026")
    df = _read_csv(path)
    if "event_name" not in df.columns:
        names = list(_read_csv(find_file(config.DEFAULT_DATA_DIR, "berlin_events_summer_2026"), nrows=0).columns)
        df = _read_csv(path, header=None, names=names)
        for c in df.select_dtypes("object").columns:
            df[c] = df[c].map(_fix_mojibake)
    return df


def load_closures(data_dir) -> pd.DataFrame:
    cl = _read_csv(find_file(data_dir, "closures"))
    cl["when"] = pd.to_datetime(cl["when"])
    return cl


def resample(flows: pd.DataFrame, weather: pd.DataFrame, freq: str):
    """Coarser time grid: passengers are summed, weather is averaged.
    min_count=1 keeps slots without any data (night pause 1-4 am) as NaN instead of 0."""
    if freq in ("15min", "15T"):
        return flows, weather
    return flows.resample(freq).sum(min_count=1), weather.resample(freq).mean()
