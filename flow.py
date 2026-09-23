"""
Schritt 2: Jedes Event der naechstgelegenen U-Bahn-Station zuordnen.

Eingabe (laut dataset_schema.md):
  - berlin_events_summer_2026.csv   (venue_name, address, city, country, ...)
  - stations_with_ubahn.csv         (station_id, station_name, longitude, latitude, u_bahn_lines)
  - flows.csv (optional)            nur Header, um zu pruefen, ob die Station Flow-Daten hat

Ausgabe:
  - JSON-Datei mit einem Eintrag pro Event inkl. naechster Station und Distanz
  - geocode_cache.json: Cache der Geocoding-Ergebnisse. Kann manuell editiert
    werden, um falsche oder fehlende Koordinaten zu korrigieren.

Aufruf:
  pip install pandas numpy geopy
  python event_station_mapping.py --data-dir data --out data/processed/event_nearest_station.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Grobe Bounding Box Berlin, um offensichtlich falsche Geocoding-Treffer zu erkennen
BERLIN_BBOX = {"lat_min": 52.33, "lat_max": 52.68, "lon_min": 13.08, "lon_max": 13.77}

# Ab dieser Distanz gilt ein Event als "keine U-Bahn-Station in der Naehe"
# (Netz ist partiell, eine Linie fehlt komplett)
MAX_DISTANCE_M = 1500

EARTH_RADIUS_M = 6_371_000


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2_arr, lon2_arr):
    """Distanz in Metern von einem Punkt zu vielen Punkten (vektorisiert)."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def in_berlin(lat, lon):
    return (BERLIN_BBOX["lat_min"] <= lat <= BERLIN_BBOX["lat_max"]
            and BERLIN_BBOX["lon_min"] <= lon <= BERLIN_BBOX["lon_max"])


def clean(value):
    """NaN / leere Strings zu None, sonst getrimmter String."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    return s or None


def venue_key(venue, address):
    """Eindeutiger Schluessel pro Veranstaltungsort (fuer den Cache)."""
    return f"{venue or ''} | {address or ''}"


# --------------------------------------------------------------------------
# Geocoding mit Cache
# --------------------------------------------------------------------------
class CachedGeocoder:
    """
    Geocodiert jeden Veranstaltungsort nur einmal (Nominatim / OpenStreetMap,
    max. 1 Anfrage pro Sekunde). Ergebnisse werden in einer JSON-Datei gecacht.
    Manuelle Korrekturen: im Cache lat/lon eintragen und "source": "manual" setzen.
    """

    def __init__(self, cache_path: Path, offline: bool = False):
        self.cache_path = cache_path
        self.offline = offline
        self.cache = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self._geocoder = None

    def _get_geocoder(self):
        if self._geocoder is None:
            from geopy.geocoders import Nominatim
            self._geocoder = Nominatim(user_agent="innotrans2026-hackathon-event-mapping")
        return self._geocoder

    def _query_nominatim(self, query):
        time.sleep(1.1)  # Nominatim-Nutzungsbedingungen: max. 1 Request/Sekunde
        try:
            loc = self._get_geocoder().geocode(query, country_codes="de", timeout=10)
        except Exception as exc:  # Netzwerkfehler, Timeout etc.
            print(f"  ! Geocoding-Fehler fuer '{query}': {exc}")
            return None
        if loc is None:
            return None
        return float(loc.latitude), float(loc.longitude)

    def geocode(self, venue, address, city, country):
        key = venue_key(venue, address)
        if key in self.cache:
            return self.cache[key]

        result = {"lat": None, "lon": None, "source": None, "query": None}
        if not self.offline:
            # Vom genauesten zum groebsten Suchstring
            candidates = [
                ", ".join(p for p in [venue, address, city, country] if p),
                ", ".join(p for p in [address, city, country] if p),
                ", ".join(p for p in [venue, city, country] if p),
            ]
            seen = set()
            for query in candidates:
                if not query or query in seen:
                    continue
                seen.add(query)
                hit = self._query_nominatim(query)
                if hit and in_berlin(*hit):
                    result = {"lat": hit[0], "lon": hit[1], "source": "nominatim", "query": query}
                    break

        self.cache[key] = result
        return result

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


# --------------------------------------------------------------------------
# Hauptlogik
# --------------------------------------------------------------------------
def load_stations(path: Path) -> pd.DataFrame:
    stations = pd.read_csv(path, dtype={"station_id": str})
    stations = stations.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return stations


def load_flow_station_names(path: Path):
    """Liest nur den Header von flows.csv, um die Stationsspalten zu kennen."""
    if not path.exists():
        return None
    header = pd.read_csv(path, nrows=0).columns
    return set(c for c in header if c != "timestamp")


def map_events(events: pd.DataFrame, stations: pd.DataFrame, geocoder: CachedGeocoder,
               flow_stations, n_alternatives: int):
    st_lat = stations["latitude"].to_numpy(dtype=float)
    st_lon = stations["longitude"].to_numpy(dtype=float)

    results = []
    for i, row in events.iterrows():
        venue = clean(row.get("venue_name"))
        address = clean(row.get("address"))
        city = clean(row.get("city"))
        country = clean(row.get("country"))

        attendance = row.get("estimated_attendance")
        entry = {
            "event_id": f"evt_{i:04d}",
            "event_name": clean(row.get("event_name")),
            "began_local": clean(row.get("began_local")),
            "estimated_end_local": clean(row.get("estimated_end_local")),
            "venue_name": venue,
            "address": address,
            "city": city,
            "segment": clean(row.get("segment")),
            "genre": clean(row.get("genre")),
            "estimated_attendance": None if pd.isna(attendance) else int(attendance),
            "event_location": None,
            "nearest_station": None,
            "alternative_stations": [],
            "flags": [],
        }

        if city and city.lower() != "berlin":
            entry["flags"].append("city_not_berlin")

        geo = geocoder.geocode(venue, address, city, country)
        if geo["lat"] is None:
            entry["flags"].append("geocoding_failed")
            results.append(entry)
            continue

        entry["event_location"] = {"lat": geo["lat"], "lon": geo["lon"],
                                   "geocode_source": geo["source"]}

        dist = haversine_m(geo["lat"], geo["lon"], st_lat, st_lon)
        order = np.argsort(dist)[: 1 + n_alternatives]

        def station_info(idx):
            s = stations.iloc[idx]
            info = {
                "station_id": s["station_id"],
                "station_name": s["station_name"],
                "u_bahn_lines": [l.strip() for l in str(s["u_bahn_lines"]).split(",")],
                "distance_m": round(float(dist[idx]), 1),
            }
            if flow_stations is not None:
                info["has_flow_data"] = s["station_name"] in flow_stations
            return info

        entry["nearest_station"] = station_info(order[0])
        entry["alternative_stations"] = [station_info(j) for j in order[1:]]

        if dist[order[0]] > MAX_DISTANCE_M:
            entry["flags"].append("no_station_within_max_distance")
        if flow_stations is not None and not entry["nearest_station"]["has_flow_data"]:
            entry["flags"].append("nearest_station_not_in_flows")

        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description="Events der naechsten U-Bahn-Station zuordnen")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out", default="data/processed/event_nearest_station.json", type=Path)
    parser.add_argument("--cache", default="data/processed/geocode_cache.json", type=Path)
    parser.add_argument("--alternatives", default=2, type=int,
                        help="Anzahl weiterer naher Stationen (0 = nur die naechste)")
    parser.add_argument("--offline", action="store_true",
                        help="Kein Geocoding-Request, nur Cache verwenden")
    args = parser.parse_args()

    events = pd.read_csv(args.data_dir / "berlin_events_summer_2026_pre_innotrans.csv")
    stations = load_stations(args.data_dir / "stations_with_ubahn.csv")
    flow_stations = load_flow_station_names(args.data_dir / "flows_pre_innotrans.csv")

    n_venues = events[["venue_name", "address"]].drop_duplicates().shape[0]
    print(f"{len(events)} Events, {n_venues} eindeutige Orte, {len(stations)} Stationen")

    geocoder = CachedGeocoder(args.cache, offline=args.offline)
    try:
        results = map_events(events, stations, geocoder, flow_stations, args.alternatives)
    finally:
        geocoder.save()  # Cache auch bei Abbruch sichern

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Kurze Zusammenfassung
    mapped = sum(1 for r in results if r["nearest_station"])
    flagged = sum(1 for r in results if r["flags"])
    print(f"Zugeordnet: {mapped}/{len(results)} | mit Flags: {flagged}")
    print(f"JSON geschrieben: {args.out}")
    print(f"Geocode-Cache:    {args.cache}")


if __name__ == "__main__":
    main()