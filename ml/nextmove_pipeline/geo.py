"""
Location -> nearest U-Bahn stations.

Two building blocks:

  Geocoder       turns an address / venue into (lat, lon) via OpenStreetMap Nominatim.
                 Every answer (also "not found") is stored in a JSON cache, so each place
                 is queried at most once. Older caches of the project are read as seeds.
  StationIndex   finds the k nearest stations to a coordinate (straight-line distance).

`nearest_stations_for_address()` combines both, and this module doubles as a CLI:

    python -m nextmove_pipeline.geo "Friedrichstraße 101"
    python -m nextmove_pipeline.geo "Am Glockenturm 1" --venue "Waldbühne" -k 5
    python -m nextmove_pipeline.geo "Uber-Platz 1" --offline      # cache only, no network
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .loading import load_flow_station_names, load_stations


# =========================================================================== distances
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. lat2/lon2 may be arrays."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(np.asarray(lat2, float)), np.radians(np.asarray(lon2, float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * config.EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def in_berlin(lat, lon) -> bool:
    b = config.BERLIN_BBOX
    return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


# =========================================================================== geocoding
def _clean(x) -> str:
    """None / NaN / 'nan' -> '', otherwise stripped string with single spaces."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = " ".join(str(x).split())
    return "" if s.lower() == "nan" else s


def cache_key(venue="", address="") -> str:
    """Canonical cache key 'venue|address' (lowercase). Both parts may be empty."""
    return f"{_clean(venue).lower()}|{_clean(address).lower()}"


def _parse_cached_value(value):
    """Accepts every cache format used in this project and returns (lat, lon) or None:
    [lat, lon]  |  {"lat":.., "lon":..}  |  null  |  {"lat": null, ...}."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lat, lon = value
    elif isinstance(value, dict):
        lat, lon = value.get("lat"), value.get("lon")
    else:
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(lat) or math.isnan(lon) else (lat, lon)


class Geocoder:
    """Cached Nominatim geocoder.

    Cache entries: {"venue|address": {"lat", "lon", "source", "query"}}; misses are stored
    with lat/lon = null so they are not re-queried. Set "source": "manual" to fix an entry by hand.
    """

    def __init__(self, cache_path=config.DEFAULT_GEOCODE_CACHE, seed_paths=config.SEED_GEOCODE_CACHES,
                 offline=False):
        self.cache_path = Path(cache_path)
        self.offline = offline
        self.cache = {}
        self._nominatim = None
        self._last_request = 0.0
        self._dirty = False
        # Seeds first, then our own cache on top (own entries win).
        for p in list(seed_paths) + [self.cache_path]:
            self._load(Path(p))

    # ---------------------------------------------------------------- cache I/O
    def _load(self, path: Path):
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key, value in raw.items():
            venue, _, address = str(key).partition("|")
            hit = _parse_cached_value(value)
            query = value.get("query") if isinstance(value, dict) else None
            # Older caches contain hits for the bare query 'Berlin, Germany' = city centre, not the place.
            if query and query.strip().lower() in ("berlin, germany", "berlin"):
                hit = None
            entry = {"lat": hit[0] if hit else None, "lon": hit[1] if hit else None,
                     "source": (value.get("source") if isinstance(value, dict) else None) or "cache",
                     "query": query}
            k = cache_key(venue, address)
            # A known coordinate is never overwritten by a miss from another cache.
            if hit or k not in self.cache:
                self.cache[k] = entry

    def save(self):
        """Write the merged cache (only if something new was geocoded)."""
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        self._dirty = False

    # ---------------------------------------------------------------- lookup
    def _query(self, query: str):
        """One Nominatim request, rate-limited; returns (lat, lon) inside Berlin or None."""
        if self._nominatim is None:
            from geopy.geocoders import Nominatim   # imported lazily: offline use needs no geopy
            self._nominatim = Nominatim(user_agent=config.NOMINATIM_USER_AGENT)
        wait = config.NOMINATIM_MIN_INTERVAL_S - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()
        try:
            loc = self._nominatim.geocode(query, country_codes="de", timeout=10)
        except Exception as exc:   # network error, timeout, rate limit ...
            print(f"[warn] geocoding failed for '{query}': {exc}")
            return None
        if loc is None or not in_berlin(loc.latitude, loc.longitude):
            return None
        return float(loc.latitude), float(loc.longitude)

    def locate(self, address="", venue="", city="Berlin", country="Germany"):
        """(lat, lon) of a place, or None. Lookup order:
        1) cache 'venue|address'  2) cache '|address' (same address, any venue)
        3) Nominatim, from the most to the least specific query string."""
        venue, address = _clean(venue), _clean(address)
        keys = [cache_key(venue, address)]
        if venue and address:
            keys.append(cache_key("", address))
        for k in keys:
            e = self.cache.get(k)
            if e and e["lat"] is not None:
                return e["lat"], e["lon"]
        # Known miss (already tried before) or no network allowed -> give up.
        if keys[0] in self.cache or self.offline:
            return None

        queries = [", ".join(p for p in parts if p) for parts in
                   ([venue, address, city, country], [address, city, country], [venue, city, country])]
        hit, used = None, None
        for q in dict.fromkeys(queries):            # dedupe, keep order
            if q and q != f"{city}, {country}":
                hit = self._query(q)
                if hit:
                    used = q
                    break
        self.cache[keys[0]] = {"lat": hit[0] if hit else None, "lon": hit[1] if hit else None,
                               "source": "nominatim" if hit else "not_found", "query": used}
        self._dirty = True
        return hit


# =========================================================================== nearest stations
class StationIndex:
    """Nearest-station search over stations_with_ubahn.csv."""

    def __init__(self, stations: pd.DataFrame, flow_station_names=None):
        st = stations.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        if flow_station_names is not None:
            st["has_flow_data"] = st.station_name.isin(set(flow_station_names))
        self.stations = st
        self._lat = st.latitude.to_numpy(float)
        self._lon = st.longitude.to_numpy(float)

    def nearest(self, lat, lon, k=3, max_distance_m=None, only_with_flow=False):
        """k nearest stations to (lat, lon), closest first, as a list of dicts with
        station_id, station_name, u_bahn_lines, distance_m (and has_flow_data if known)."""
        d = haversine_m(lat, lon, self._lat, self._lon)
        if only_with_flow and "has_flow_data" in self.stations:
            d = np.where(self.stations.has_flow_data.to_numpy(), d, np.inf)
        out = []
        for j in np.argsort(d)[:k]:
            if not np.isfinite(d[j]) or (max_distance_m is not None and d[j] > max_distance_m):
                break
            s = self.stations.iloc[j]
            item = {"station_id": s.station_id, "station_name": s.station_name,
                    "u_bahn_lines": [x.strip() for x in str(s.u_bahn_lines).split(",")],
                    "distance_m": round(float(d[j]), 1)}
            if "has_flow_data" in s:
                item["has_flow_data"] = bool(s.has_flow_data)
            out.append(item)
        return out


def nearest_stations_for_address(address, venue="", k=3, max_distance_m=None,
                                 data_dir=config.DEFAULT_DATA_DIR, geocoder=None, station_index=None):
    """Address (street + number, optionally a venue name) -> dict with the geocoded location
    and the k nearest U-Bahn stations. Pass `geocoder` / `station_index` to reuse them in loops."""
    geocoder = geocoder or Geocoder()
    if station_index is None:
        station_index = StationIndex(load_stations(data_dir), load_flow_station_names(data_dir))
    loc = geocoder.locate(address=address, venue=venue)
    geocoder.save()
    if loc is None:
        return {"address": address, "venue": venue, "location": None, "stations": [],
                "note": "address could not be geocoded (not found or outside Berlin)"}
    return {"address": address, "venue": venue, "location": {"lat": loc[0], "lon": loc[1]},
            "stations": station_index.nearest(loc[0], loc[1], k=k, max_distance_m=max_distance_m)}


# =========================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description="Find the nearest U-Bahn stations for a Berlin address.")
    ap.add_argument("address", help="street and number, e.g. 'Friedrichstraße 101'")
    ap.add_argument("--venue", default="", help="optional venue name (improves geocoding)")
    ap.add_argument("-k", type=int, default=3, help="number of stations to return")
    ap.add_argument("--max-distance", type=float, help="ignore stations farther away (metres)")
    ap.add_argument("--data", default=str(config.DEFAULT_DATA_DIR), help="folder with stations_with_ubahn.csv")
    ap.add_argument("--cache", default=str(config.DEFAULT_GEOCODE_CACHE), help="geocode cache JSON")
    ap.add_argument("--offline", action="store_true", help="use the cache only, never query Nominatim")
    a = ap.parse_args()
    res = nearest_stations_for_address(a.address, a.venue, a.k, a.max_distance, a.data,
                                       geocoder=Geocoder(a.cache, offline=a.offline))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
