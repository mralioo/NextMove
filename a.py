
"""
check_normalized.py - prueft die Tabellen aus normalize_flows.py

  python check_normalized.py --data ./training_dataset --freq 1h --geocode-cache analysis_output/geocode_cache.json

Checks
  1) Form: gleiche Zeitstempel und Stationen wie der (aggregierte) Roh-Flow, wenig NaN
  2) Zerlegung: gesamt = wetter + rest
  3) Normal sitzt: Median ~0 insgesamt, je Station, je Uhrzeit und je Wochentag
     (bleibt ein Tagesmuster uebrig, ist die Pendlerwelle nicht sauber raus)
  4) Wetter: bei Regen Wetteranteil > 0, bei Hitze < 0, bei trockenem Normalwetter ~0
  5) Events/Sperrungen: Rest am Bahnhof nach Eventende > 0, waehrend Sperrung < 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import event_effects_mcp as core  # noqa: E402

def check(ok, text, hint=""):
    print(f"  {'OK ' if ok else '!! '} {text}" + ("" if ok or not hint else f"\n       -> {hint}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--freq", default="15min", help="dasselbe Raster wie bei normalize_flows.py")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--geocode-cache")
    ap.add_argument("--venue-map")
    a = ap.parse_args()
    d, sfx = a.data, (f"_{a.suffix}" if a.suffix else "")

    read = lambda n: pd.read_csv(os.path.join(d, f"normalized_{n}{sfx}.csv"), index_col=0, parse_dates=True)
    T, W, R = read("flows"), read("weather"), read("rest")

    print("\n1) Form")
    flows = pd.read_csv(core.pick(d, "flows"), parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    weather = pd.read_csv(core.pick(d, "weather_data"), index_col=0, parse_dates=True).sort_index()
    if a.freq != "15min":
        flows, weather = core.aggregate(flows, weather, a.freq)
    check(T.index.equals(flows.index), f"{len(T)} Zeitstempel wie im Roh-Flow ({len(flows)})",
          "anderes --freq als beim Erzeugen?")
    missing = [c for c in flows.columns if c not in T.columns]
    check(len(missing) <= 1, f"{T.shape[1]} Stationen, im Roh-Flow aber nicht bereinigt: {len(missing)}")
    check(T.shape == W.shape == R.shape, "alle drei Tabellen gleich gross")
    empty = T.isna().all(axis=1)
    hrs = sorted(set(T.index[empty].hour))
    print(f"       {empty.mean():.1%} Zeitstempel ganz ohne Daten (Betriebspause), Stunden: {hrs}")
    nan = T[~empty].isna().to_numpy().mean()
    check(nan < 0.01, f"NaN-Anteil in Betriebszeiten {nan:.2%}")

    print("\n2) Zerlegung")
    diff = np.nanmax(np.abs((T - W - R).to_numpy()))
    check(diff < 2e-3, f"gesamt = wetter + rest, max. Abweichung {diff:.4f} (nur Rundung)")

    print("\n3) Normal sitzt (Werte auf Log-Skala, 0.05 ~ 5 %)")
    med = np.nanmedian(T.to_numpy())
    check(abs(med) < 0.05, f"Median gesamt {med:+.3f}")
    st = T.median()
    check((st.abs() > 0.1).mean() < 0.05, f"Stationen mit |Median| > 0.1: {(st.abs() > 0.1).sum()} von {len(st)}",
          f"z.B. {st.abs().sort_values().index[-1]} ({st.abs().max():.2f})")
    hr = R.groupby(R.index.hour).median().median(axis=1)
    check(hr.abs().max() < 0.1, f"Rest je Uhrzeit: max |Median| {hr.abs().max():.3f} um {hr.abs().idxmax()} Uhr",
          "Tagesmuster uebrig -> Pendlerwelle nicht sauber raus")
    wd = R.groupby(R.index.dayofweek).median().median(axis=1)
    check(wd.abs().max() < 0.1, f"Rest je Wochentag: max |Median| {wd.abs().max():.3f} (Tag {wd.abs().idxmax()}, 0 = Mo)")

    print("\n4) Wetter")
    w = weather.reindex(T.index)
    rain = (w["prcp"] > 0.1).to_numpy()
    hot = (w["temp"] > w["temp"].quantile(0.9)).to_numpy()
    calm = ~rain & ~hot
    m = lambda df, mask: np.nanmean(df.to_numpy()[mask]) if mask.any() else np.nan
    print(f"       Stunden: Regen {rain.sum()}, heiss (> {w['temp'].quantile(0.9):.1f} C) {hot.sum()}")
    # Rest wird gegen den Rest bei Normalwetter verglichen: der Median ist 0, der Mittelwert wegen
    # der Nullen im Flow leicht negativ -> entscheidend ist, dass Regen/Hitze nicht davon abweichen
    rc = m(R, calm)
    check(m(W, rain) > 0.02, f"Regen: Wetteranteil {m(W, rain):+.3f}, Rest {m(R, rain) - rc:+.3f} ggue. Normalwetter",
          "Regen wird nicht erkannt")
    check(m(W, hot) < -0.02, f"Hitze: Wetteranteil {m(W, hot):+.3f}, Rest {m(R, hot) - rc:+.3f} ggue. Normalwetter",
          "Hitze wird nicht erkannt")
    check(abs(m(W, calm)) < 0.03, f"Normalwetter: Wetteranteil {m(W, calm):+.3f}")
    check(abs(m(R, rain) - rc) < 0.05 and abs(m(R, hot) - rc) < 0.05,
          "Rest bei Regen/Hitze wie bei Normalwetter (Wetter steckt im Wetteranteil)")

    print("\n5) Events und Sperrungen (Rest an der Ankerstation)")
    episodes = core.load_real(d, a.venue_map, a.geocode_cache)[3]
    res = {"closure": [], "event": []}
    for ep in episodes:
        cl = ep["type"] == "closure"
        lo, hi = (ep["start"], ep["end"]) if cl else (ep["end"], ep["end"] + pd.Timedelta(hours=1))
        rows = (R.index >= lo) & (R.index < hi)
        cols = [s for s in ep["anchors"] if s in R.columns]
        vals = R.loc[rows, cols].to_numpy() if rows.any() and cols else np.array([np.nan])
        if not np.isnan(vals).all():   # Fenster in der Betriebspause -> keine Daten
            res["closure" if cl else "event"].append(np.nanmean(R.loc[rows, cols].to_numpy()))
    for k, want, label in [("closure", -1, "waehrend Sperrung"), ("event", 1, "1 h nach Eventende")]:
        v = np.array(res[k])
        if not len(v):
            print(f"       keine {k}-Episoden mit Daten")
            continue
        share = np.mean(np.sign(v) == want)
        check(share >= 0.6, f"{k}: {len(v)} Episoden, Rest {label} im Median {np.median(v):+.3f}, "
                            f"{share:.0%} mit erwartetem Vorzeichen",
              "Effekt kaum sichtbar - Zuordnung Event -> Station pruefen")
    print()


if __name__ == "__main__":
    main()