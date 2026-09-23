"""
Ein Event aus event_windows_stations.parquet auswaehlen, die naechste Station
(aus event_windows_meta.json) bestimmen und den Effekt auf den Flow zeigen:
beobachteter Flow vs. Baseline, plus Anomalie, ueber das gesamte -3h/+3h-Fenster.

Aufruf:
  pip install pandas matplotlib pyarrow
  python show_event_effect.py                      # nimmt automatisch das Event
                                                     # mit dem hoechsten |Z-Score|
  python show_event_effect.py --random              # zufaelliges auswertbares Event
  python show_event_effect.py --random --seed 42    # reproduzierbar zufaellig
  python show_event_effect.py --event-id evt_0000   # bestimmtes Event
  python show_event_effect.py --list                # alle Events mit Kennzahlen auflisten
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def load_meta(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {m["event_id"]: m for m in meta}


def evaluable_event_ids(df: pd.DataFrame, meta: dict):
    """Alle Event-IDs, fuer die es tatsaechlich Zeilen an der naechsten Station gibt."""
    ids = []
    for eid, m in meta.items():
        if "skipped" in m or not m.get("nearest_station"):
            continue
        if ((df["event_id"] == eid) & (df["station_name"] == m["nearest_station"])).any():
            ids.append(eid)
    return ids


def pick_event_id(df: pd.DataFrame, meta: dict, requested, mode: str = "max_z"):
    if requested:
        if requested not in set(df["event_id"]):
            raise SystemExit(f"Event-ID '{requested}' nicht in den Daten gefunden.")
        return requested

    ids = evaluable_event_ids(df, meta)
    if not ids:
        raise SystemExit("Kein auswertbares Event gefunden.")

    if mode == "random":
        return random.choice(ids)

    # Default: das Event mit dem staerksten Ausschlag an der naechsten Station
    scored = []
    for eid in ids:
        m = meta[eid]
        sub = df[(df["event_id"] == eid) & (df["station_name"] == m["nearest_station"])]
        scored.append((sub["z_score"].abs().max(), eid))
    return max(scored)[1]


def list_events(df: pd.DataFrame, meta: dict):
    rows = []
    for eid, m in meta.items():
        if "skipped" in m or not m.get("nearest_station"):
            continue
        sub = df[(df["event_id"] == eid) & (df["station_name"] == m["nearest_station"])]
        if sub.empty:
            continue
        rows.append({
            "event_id": eid,
            "event_name": m.get("event_name"),
            "nearest_station": m["nearest_station"],
            "attendance": m.get("attendance"),
            "max_abs_z": round(sub["z_score"].abs().max(), 1),
            "max_excess": round(sub["anomaly"].max(), 1),
        })
    out = pd.DataFrame(rows).sort_values("max_abs_z", ascending=False)
    print(out.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stations-parquet", type=Path,
                    default=Path("data/processed/event_windows_stations.parquet"))
    p.add_argument("--meta", type=Path, default=Path("data/processed/event_windows_meta.json"))
    p.add_argument("--event-id", default=None)
    p.add_argument("--random", action="store_true",
                    help="zufaelliges auswertbares Event statt des staerksten Ausschlags")
    p.add_argument("--seed", type=int, default=None, help="Seed fuer --random (reproduzierbar)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--out", type=Path, default=Path("data/processed/event_effect.png"))
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    path = args.stations_parquet
    if not path.exists():  # Fallback, falls ohne pyarrow als .csv.gz geschrieben wurde
        path = path.with_suffix(".csv.gz")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    meta = load_meta(args.meta)

    if args.list:
        list_events(df, meta)
        return

    event_id = pick_event_id(df, meta, args.event_id, mode="random" if args.random else "max_z")
    m = meta[event_id]
    station = m["nearest_station"]
    print(f"Event:            {m.get('event_name')} ({event_id})")
    print(f"Naechste Station: {station}")
    print(f"Fenster:          {m['window_start']}  bis  {m['window_end']}")
    print(f"Event:            {m['begin']}  bis  {m['end']}  (end_source={m.get('end_source')})")
    if m.get("flags"):
        print(f"Flags:            {m['flags']}")

    sub = (df[(df["event_id"] == event_id) & (df["station_name"] == station)]
           .sort_values("timestamp").reset_index(drop=True))
    if sub.empty:
        raise SystemExit(f"Keine Zeilen fuer Event '{event_id}' an Station '{station}'.")

    cols = ["timestamp", "rel_slot", "phase", "observed", "baseline",
            "anomaly", "anomaly_pct", "z_score", "baseline_source", "n_ref_days"]
    print("\n" + sub[cols].to_string(index=False))

    pre = sub[sub["phase"] == "pre"]["anomaly"].clip(lower=0).sum()
    post = sub[sub["phase"] == "post"]["anomaly"].clip(lower=0).sum()
    print("\nZusaetzliche Passagiere (Summe positiver Anomalie):")
    print(f"  vorher (Anreise):  {pre:,.0f}")
    print(f"  nachher (Abreise): {post:,.0f}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib nicht installiert (pip install matplotlib) - kein Plot erzeugt.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(sub["timestamp"], sub["observed"], label="beobachteter Flow", color="tab:blue")
    ax1.plot(sub["timestamp"], sub["baseline"], label="Baseline (erwartet)",
              color="tab:gray", linestyle="--")
    ax1.fill_between(sub["timestamp"], sub["baseline"], sub["observed"],
                      where=sub["observed"] >= sub["baseline"],
                      color="tab:red", alpha=0.25, label="Ueberschuss")
    ax1.axvline(pd.Timestamp(m["begin"]), color="green", linestyle=":", label="Event-Start")
    ax1.axvline(pd.Timestamp(m["end"]), color="darkred", linestyle=":", label="Event-Ende")
    ax1.set_ylabel("Passagiere / 15 min")
    ax1.set_title(f"{m.get('event_name')} @ {station}")
    ax1.legend(loc="upper left", fontsize=8)

    colors = sub["anomaly"].apply(lambda v: "tab:red" if v >= 0 else "tab:blue")
    ax2.bar(sub["timestamp"], sub["anomaly"], width=0.008, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Anomalie\n(beobachtet - Baseline)")
    ax2.set_xlabel("Zeit")

    fig.autofmt_xdate()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nPlot gespeichert: {args.out}")


if __name__ == "__main__":
    main()