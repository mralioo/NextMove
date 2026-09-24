"""GOLDEN DATA — direct, read-only access to the pre-processed high-quality files: `data/normalized/*` (the normalization pipeline's tables and model) and `data/processed/*` (geocode cache).

Schema and meaning of every file: `data/data_schema_high_quality.md`. The quality database (`ml/quality_db.py`, boundaries and checks) is DERIVED from these files; this module gives agents the files
themselves: catalog, schema and statistics, station time series across all four tables, slices at a time, episodes, coefficients, the fitted normal-flow MODEL (normal flow for ANY timestamp,
also outside the data window) and the venue → station lookup (geocode cache + nearest stations).

    from golden_data import Golden;  g = Golden();  g.catalog();  g.series("Rudow", "2026-07-14 16:00", "2026-07-14 20:00")
"""
from __future__ import annotations

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent
REPO = ML.parent
sys.path[:0] = [str(ML)]

NORM = REPO / "data" / "normalized"
PROC = REPO / "data" / "processed"

# name -> (file, kind, description). Wide tables: index timestamp, one column per station (167); *_test = the held-out days scored with the SAME model.
WIDE = {
    "normalized_flows": ("total", "total = log1p(actual) − normal  (log scale, 0 = normal, +0.1 ≈ +10 %, +0.69 ≈ twice as many)"),
    "normalized_weather": ("weather", "weather part = expected (actual weather) − normal: what rain / heat / wind explain (log scale)"),
    "normalized_rest": ("rest", "rest = total − weather: events, closures, anomalies (log scale); total = weather + rest exactly"),
    "normal_flow_passengers": ("normal", "normal flow in PASSENGERS per 15 minutes (typical weather); the reference for '+1 200 people'"),
}
TABLES = {"episodes": "events (≥ 2 000 visitors) and closures with their stations and windows, the cells the model excluded from its fit",
          "normal_flow_coefficients": "rain / heat / school-holiday effect per station in %"}


def _fmt(t) -> str:
    return str(pd.Timestamp(t))


@lru_cache(maxsize=8)
def _wide(name: str) -> pd.DataFrame:
    """A normalized table, training + test split concatenated (float32)."""
    parts = []
    for suffix in ("", "_test"):
        f = NORM / f"{name}{suffix}.csv"
        if f.exists():
            parts.append(pd.read_csv(f, index_col=0, parse_dates=True).astype("float32").assign(_split="train" if not suffix else "test"))
    if not parts:
        raise FileNotFoundError(f"{NORM}/{name}.csv not found (build: ./.venv/bin/python ml/quality_db.py build)")
    return pd.concat(parts).sort_index()


@lru_cache(maxsize=1)
def _model() -> dict:
    with open(NORM / "normal_flow_model.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _names():
    from nextmove_pipeline.names import StationNameIndex
    return StationNameIndex(list(_wide("normalized_rest").drop(columns="_split").columns))


class Golden:
    def __init__(self) -> None:
        self.norm, self.proc = NORM, PROC

    @property
    def available(self) -> bool:
        return (NORM / "normalized_rest.csv").exists()

    # ------------------------------------------------------------------ catalog / schema
    def catalog(self) -> list[dict]:
        out = []
        for name, (short, desc) in WIDE.items():
            for suffix, split in (("", "train"), ("_test", "test")):
                f = NORM / f"{name}{suffix}.csv"
                out.append({"name": f"{name}{suffix}", "path": str(f.relative_to(REPO)), "kind": "wide table (timestamp × 167 stations)", "split": split, "exists": f.exists(),
                            "size_mb": round(f.stat().st_size / 1e6, 1) if f.exists() else None, "description": desc, "unit": "passengers / 15 min" if short == "normal" else "log ratio"})
        for name, desc in TABLES.items():
            for suffix in ("", "_test") if name == "episodes" else ("",):
                f = NORM / f"{name}{suffix}.csv"
                out.append({"name": f"{name}{suffix}", "path": str(f.relative_to(REPO)), "kind": "table", "split": "test" if suffix else "train", "exists": f.exists(), "size_mb": round(f.stat().st_size / 1e6, 3) if f.exists() else None, "description": desc})
        m = NORM / "normal_flow_model.pkl"
        out.append({"name": "normal_flow_model", "path": str(m.relative_to(REPO)), "kind": "fitted model (pickle)", "exists": m.exists(), "size_mb": round(m.stat().st_size / 1e6, 2) if m.exists() else None,
                    "description": "ridge normal-flow model per station (B), typical weather per slot, median re-centring shift, training statistics; gives the normal flow for ANY timestamp (golden_normal_flow)"})
        g = PROC / "geocode_cache_pipeline.json"
        out.append({"name": "geocode_cache_pipeline", "path": str(g.relative_to(REPO)), "kind": "json cache", "exists": g.exists(), "size_mb": round(g.stat().st_size / 1e6, 3) if g.exists() else None,
                    "description": "venue|address → coordinates (OpenStreetMap Nominatim, cached; 'not_found' entries have null lat/lon); used to place events at stations"})
        return out

    def describe(self, name: str) -> dict:
        base = name.replace("_test", "")
        if base in WIDE:
            df = _wide(base)
            sub = df[df._split == ("test" if name.endswith("_test") else "train")].drop(columns="_split")
            v = sub.to_numpy(dtype="float64")
            return {"name": name, "rows": len(sub), "columns": sub.shape[1], "index": "timestamp (15-minute slots, Berlin local time)", "from": _fmt(sub.index[0]), "to": _fmt(sub.index[-1]),
                    "station_columns_sample": list(sub.columns[:5]), "unit": "passengers / 15 min" if base == "normal_flow_passengers" else "log ratio (0 = normal)",
                    "nan_share": round(float(np.isnan(v).mean()), 4), "min": round(float(np.nanmin(v)), 3), "p05": round(float(np.nanquantile(v, .05)), 3), "median": round(float(np.nanmedian(v)), 3),
                    "p95": round(float(np.nanquantile(v, .95)), 3), "max": round(float(np.nanmax(v)), 3), "description": WIDE[base][1]}
        if base in TABLES:
            df = pd.read_csv(NORM / f"{name}.csv")
            return {"name": name, "rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()}, "sample": df.head(2).to_dict("records"), "description": TABLES[base]}
        if name == "normal_flow_model":
            m = _model()
            return {"name": name, "grid": m["freq"], "stations": len(m["stations"]), "design_matrix_columns": int(m["B"].shape[0]), "level_dummies": int(m["n_level"]), "extra_terms": m["extra_terms"],
                    "training_window": [str(m["stats"]["t0"]), str(m["stats"]["t1"])], "mean_temperature_c": round(m["stats"]["temp"][0], 1), "typical_weather_slots": int(len(m["typical"])),
                    "note": "level dummies = 80 slots × 4 day types (Mon-Thu, Fri, Sat, Sun); the trend term is clipped to [-1, 1] outside the training window"}
        if name == "geocode_cache_pipeline":
            c = json.loads((PROC / "geocode_cache_pipeline.json").read_text())
            return {"name": name, "entries": len(c), "found": sum(1 for v in c.values() if v.get("lat") is not None), "not_found": sum(1 for v in c.values() if v.get("lat") is None),
                    "sources": sorted({v.get("source") for v in c.values()}), "key_format": "'venue|address' lower-case; empty venue = '|address'"}
        return {"error": f"unknown dataset '{name}'; call golden_datasets"}

    # ------------------------------------------------------------------ station-level access
    def resolve(self, station: str) -> str | None:
        cols = list(_wide("normalized_rest").drop(columns="_split").columns)
        if station in cols:
            return station
        return _names().resolve(station, cutoff=0.8) or next((c for c in cols if station.lower() in c.lower()), None)

    def series(self, station: str, start: str, end: str, max_points: int = 200) -> dict:
        """All four tables for one station in [start, end], plus the ACTUAL passengers reconstructed from them (actual = expm1(total + log1p(normal))), which equals the raw reading within rounding (tables are rounded to 4 / 1 decimals)."""
        s = self.resolve(station)
        if s is None:
            return {"error": f"unknown station '{station}'"}
        d = {k: _wide(k)[s] for k in WIDE}
        df = pd.DataFrame({WIDE[k][0]: v for k, v in d.items()})
        split = _wide("normalized_rest")["_split"]
        df = df.loc[start:end]
        if df.empty:
            return {"error": f"no rows between {start} and {end}"}
        df["actual"] = np.expm1(df["total"] + np.log1p(df["normal"]))
        df["split"] = split.loc[df.index]
        step = max(1, int(np.ceil(len(df) / max_points)))
        df = df.iloc[::step]
        rows = [{"ts": str(t), "actual": None if np.isnan(r.actual) else round(float(r.actual), 1), "normal": None if np.isnan(r.normal) else round(float(r.normal), 1), "total": round(float(r.total), 3),
                 "weather": round(float(r.weather), 3), "rest": round(float(r.rest), 3), "split": r.split} for t, r in df.iterrows()]
        return {"station": s, "n_rows": len(rows), "every_n_slots": step, "unit": {"actual/normal": "passengers per 15 min", "total/weather/rest": "log ratio, 0 = normal"}, "rows": rows}

    def slice(self, at: str, table: str = "normalized_rest", top_n: int = 10, order: str = "desc") -> dict:
        """All stations at one 15-minute slot from one table, largest (or smallest) first."""
        if table not in WIDE:
            return {"error": f"table must be one of {list(WIDE)}"}
        df = _wide(table).drop(columns="_split")
        i = df.index.searchsorted(pd.Timestamp(at))
        if i >= len(df) or abs((df.index[i] - pd.Timestamp(at)).total_seconds()) > 900:
            return {"error": f"no slot at {at} (window {df.index[0]} .. {df.index[-1]})"}
        r = df.iloc[i].dropna().sort_values(ascending=(order == "asc")).head(top_n)
        return {"table": table, "at": str(df.index[i]), "order": order, "values": {k: round(float(v), 3) for k, v in r.items()}}

    def episodes(self, date: str = "", station: str = "", kind: str = "", limit: int = 30) -> list[dict]:
        e = pd.concat([pd.read_csv(NORM / "episodes.csv").assign(split="train"), pd.read_csv(NORM / "episodes_test.csv").assign(split="test")], ignore_index=True)
        if date:
            e = e[e.start.astype(str).str.startswith(date) | e.end.astype(str).str.startswith(date)]
        if station:
            s = self.resolve(station) or station
            e = e[e.anchors.astype(str).str.contains(s, regex=False)]
        if kind:
            e = e[e.kind == kind]
        return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in e.head(limit).to_dict("records")]

    def coefficients(self, station: str = "") -> dict:
        c = pd.read_csv(NORM / "normal_flow_coefficients.csv")
        if station:
            s = self.resolve(station)
            return c[c.station == s].iloc[0].to_dict() if s and (c.station == s).any() else {"error": f"unknown station '{station}'"}
        return {"network_median_pct": c.drop(columns="station").median().round(2).to_dict(), "stations": len(c), "extremes": {col: {"max": c.loc[c[col].idxmax(), "station"], "min": c.loc[c[col].idxmin(), "station"]}
                                                                                                                       for col in c.columns if col != "station"}}

    # ------------------------------------------------------------------ the model
    def normal_flow(self, station: str, at: str, periods: int = 1) -> dict:
        """The fitted model's NORMAL flow (typical weather) for `periods` consecutive 15-minute slots from `at` — for ANY timestamp, also outside the data window (e.g. a date after 2026-10-01).
        Outside the training window the trend term is clipped (the model does not extrapolate a trend); the shift and level terms carry the time-of-day / day-type pattern."""
        from nextmove_pipeline.baseline import design_matrix, set_typical_weather, slot_daytype_index
        s = self.resolve(station)
        if s is None:
            return {"error": f"unknown station '{station}'"}
        m = _model()
        idx = pd.date_range(pd.Timestamp(at), periods=max(1, min(periods, 96)), freq="15min")
        w = pd.DataFrame({"temp": np.nan, "prcp": 0.0, "wspd": np.nan}, index=idx)
        X, n_level = design_matrix(idx, w, m["stats"])
        set_typical_weather(X, n_level, m["typical"], idx)
        j = m["stations"].index(s)
        shift = m["shift"].reindex(slot_daytype_index(idx))[s].fillna(0).to_numpy()
        normal = np.expm1(X @ m["B"][:, j] + shift)
        inside = (idx >= m["stats"]["t0"]) & (idx <= m["stats"]["t1"])
        return {"station": s, "unit": "passengers per 15 minutes, typical weather", "outside_training_window": bool((~inside).any()),
                "rows": [{"ts": str(t), "normal": round(float(v), 1)} for t, v in zip(idx, normal)]}

    def model_info(self) -> dict:
        return self.describe("normal_flow_model")

    # ------------------------------------------------------------------ processed: geocode cache
    def geocode_cache(self, query: str = "", limit: int = 20) -> list[dict]:
        c = json.loads((PROC / "geocode_cache_pipeline.json").read_text())
        q = query.lower()
        rows = [{"key": k, **v} for k, v in c.items() if not q or q in k]
        return rows[:limit]

    def venue_station(self, venue: str = "", address: str = "", k: int = 3) -> dict:
        """Which U-Bahn stations are nearest to a venue / address? Manual overrides first (config.VENUE_TO_STATION), then the geocode cache (never a network request), then station-name text."""
        from nextmove_pipeline import config, loading
        from nextmove_pipeline.geo import Geocoder, StationIndex, cache_key
        from nextmove_pipeline.names import normalize_text
        text = normalize_text(f"{venue} | {address}")
        for key, station in config.VENUE_TO_STATION.items():
            if normalize_text(key) in text:
                return {"method": "manual override (domain knowledge)", "stations": [{"station_name": station, "distance_m": None}]}
        gc = Geocoder(offline=True)
        loc = gc.locate(address=address, venue=venue)
        if loc is None:
            hits = _names().find_in_text(text)
            return {"method": "station name in text" if hits else "not found", "stations": [{"station_name": h} for h in hits[:k]], "note": None if hits else "not in the geocode cache (offline lookup only)"}
        idx = StationIndex(loading.load_stations(config.DEFAULT_DATA_DIR), loading.load_flow_station_names(config.DEFAULT_DATA_DIR))
        return {"method": "geocode cache + nearest station", "location": {"lat": loc[0], "lon": loc[1]}, "cache_key": cache_key(venue, address), "stations": idx.nearest(loc[0], loc[1], k=k)}
