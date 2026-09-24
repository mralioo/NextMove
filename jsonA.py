"""
Baut aus den Ergebnissen von Schritt 2 (event_nearest_station.json) und
Schritt 3+4 (event_windows_stations.parquet / event_windows_meta.json) ein
kompaktes JSON: pro Event, welche Nachbarstationen betroffen sind und welche
Anomalien dort auftreten (Zeit, Phase, Staerke).

Laeuft zusaetzlich NACH event_baseline.py, veraendert dessen Ausgaben nicht.

Eingaben:
  data/processed/event_nearest_station.json   (Schritt 2: nearest_station + alternative_stations)
  data/processed/event_windows_stations.parquet (oder .csv.gz)  (Schritt 3+4)
  data/processed/event_windows_meta.json      (Schritt 3+4: begin/end/flags)

Ausgabe:
  data/processed/event_neighbor_anomalies.json

Aufruf:
  python event_neighbor_anomalies.py
  python event_neighbor_anomalies.py --z-threshold 2.5
"""

import argparse
import json
from pathlib import Path

import pandas as pd

PHASE_TO_DIRECTION = {"pre": "arrival", "during": "during_event", "post": "departure"}


def load_table(base_path: Path) -> pd.DataFrame:
    path = base_path if base_path.exists() else base_path.with_suffix(".csv.gz")
    if not path.exists():
        raise SystemExit(f"Nicht gefunden: {base_path} (auch nicht als .csv.gz)")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_neighbor_lists(path: Path) -> dict:
    """event_id -> Liste von {station_name, distance_m, within_radius, u_bahn_lines, ...},
    naechste Station zuerst, ohne Duplikate."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for e in raw:
        near = e.get("nearest_station")
        alts = e.get("alternative_stations") or []
        seen, neighbors = set(), []
        for s in ([near] if near else []) + alts:
            if not s or s["station_name"] in seen:
                continue
            seen.add(s["station_name"])
            neighbors.append({
                "station_name": s["station_name"],
                "distance_m": s.get("distance_m"),
                "within_radius": s.get("within_radius", True),
                "u_bahn_lines": s.get("u_bahn_lines"),
                "is_nearest": s is near,
                "has_flow_data": s.get("has_flow_data"),
            })
        out[e["event_id"]] = neighbors
    return out


def summarize_phase(sub: pd.DataFrame):
    """Kompakte Kennzahlen fuer eine Phase (pre/during/post) an einer Station."""
    if sub.empty or sub["z_score"].notna().sum() == 0:
        # leer, oder alle Werte NaN (z. B. Slot ausserhalb des Flow-Zeitraums)
        excess_sum = float(sub["anomaly"].clip(lower=0).fillna(0).sum())
        return {"n_slots": int(len(sub)), "excess_sum": round(excess_sum, 1), "max_abs_z": None}
    excess_sum = float(sub["anomaly"].clip(lower=0).sum())
    peak = sub.loc[sub["z_score"].abs().idxmax()]
    return {
        "n_slots": int(len(sub)),
        "excess_sum": round(excess_sum, 1),
        "max_abs_z": round(float(peak["z_score"]), 2),
    }


def significant_slots(sub: pd.DataFrame, z_threshold: float, max_slots: int):
    """Liste der auffaelligen Zeitpunkte (|z| >= threshold), staerkste zuerst,
    auf max_slots gekappt, damit die JSON-Datei nicht ausufert."""
    sig = sub[sub["z_score"].notna() & (sub["z_score"].abs() >= z_threshold)].copy()
    sig = sig.reindex(sig["z_score"].abs().sort_values(ascending=False).index)
    rows = []
    for r in sig.head(max_slots).itertuples():
        rows.append({
            "timestamp": r.timestamp.isoformat(),
            "rel_slot": int(r.rel_slot),
            "phase": r.phase,
            "observed": round(float(r.observed), 1),
            "baseline": round(float(r.baseline), 1),
            "anomaly": round(float(r.anomaly), 1),
            "anomaly_pct": None if pd.isna(r.anomaly_pct) else round(float(r.anomaly_pct), 1),
            "z_score": round(float(r.z_score), 2),
        })
    return rows, int(len(sig))


def build_neighbor_entry(sub: pd.DataFrame, neighbor: dict, z_threshold: float, max_slots: int):
    sub = sub.sort_values("timestamp")
    by_phase = {ph: summarize_phase(sub[sub["phase"] == ph]) for ph in ["pre", "during", "post"]}
    sig_rows, n_sig = significant_slots(sub, z_threshold, max_slots)

    affected = n_sig > 0
    direction = None
    if affected:
        top_phase = sig_rows[0]["phase"]
        direction = PHASE_TO_DIRECTION.get(top_phase)

    hop = sub["hop_from_venue"].dropna()
    return {
        "station_name": neighbor["station_name"],
        "distance_m": neighbor["distance_m"],
        "within_radius": neighbor["within_radius"],
        "is_nearest_station": neighbor["is_nearest"],
        "u_bahn_lines": neighbor["u_bahn_lines"],
        "hop_from_venue": None if hop.empty else int(hop.iloc[0]),
        "affected": affected,
        "dominant_direction": direction,
        "n_significant_slots": n_sig,
        "by_phase": by_phase,
        "significant_anomalies": sig_rows,
    }


def main():
    p = argparse.ArgumentParser(description="Event -> betroffene Nachbarstationen -> Anomalien, als JSON")
    p.add_argument("--stations-parquet", type=Path,
                    default=Path("data/processed/event_windows_stations.parquet"))
    p.add_argument("--events-json", type=Path,
                    default=Path("data/processed/event_nearest_station.json"))
    p.add_argument("--meta", type=Path, default=Path("data/processed/event_windows_meta.json"))
    p.add_argument("--out", type=Path, default=Path("data/processed/event_neighbor_anomalies.json"))
    p.add_argument("--z-threshold", type=float, default=2.0,
                    help="|z-score| ab dem ein Slot als Anomalie gilt")
    p.add_argument("--max-slots-per-neighbor", type=int, default=20,
                    help="Deckelung der aufgelisteten Anomalie-Slots pro Nachbarstation")
    p.add_argument("--only-affected", action="store_true",
                    help="Nachbarn ohne jede Anomalie aus der Ausgabe weglassen")
    args = p.parse_args()

    df = load_table(args.stations_parquet)
    neighbor_lists = load_neighbor_lists(args.events_json)
    meta = {m["event_id"]: m for m in json.loads(args.meta.read_text(encoding="utf-8"))}

    results = []
    for event_id, neighbors in neighbor_lists.items():
        m = meta.get(event_id, {})
        if "skipped" in m or not neighbors:
            continue

        event_df = df[df["event_id"] == event_id]
        if event_df.empty:
            continue

        neighbor_entries = []
        for nb in neighbors:
            sub = event_df[event_df["station_name"] == nb["station_name"]]
            if sub.empty:
                continue  # z. B. Station nicht in flows.csv (has_flow_data=False)
            entry = build_neighbor_entry(sub, nb, args.z_threshold, args.max_slots_per_neighbor)
            if args.only_affected and not entry["affected"]:
                continue
            neighbor_entries.append(entry)

        neighbor_entries.sort(key=lambda e: (not e["affected"], e["distance_m"] or 1e9))

        results.append({
            "event_id": event_id,
            "event_name": m.get("event_name"),
            "attendance": m.get("attendance"),
            "begin": m.get("begin"),
            "end": m.get("end"),
            "end_source": m.get("end_source"),
            "event_flags": m.get("flags", []),
            "n_neighbors_checked": len(neighbor_entries),
            "n_neighbors_affected": sum(e["affected"] for e in neighbor_entries),
            "neighbors": neighbor_entries,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ev = len(results)
    n_aff = sum(1 for r in results if r["n_neighbors_affected"] > 0)
    print(f"{n_ev} Events verarbeitet, {n_aff} mit mindestens einer betroffenen Nachbarstation")
    print(f"JSON geschrieben: {args.out}")


if __name__ == "__main__":
    main()