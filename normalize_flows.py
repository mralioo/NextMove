"""
normalize_flows.py - bereinigter Flow per Regression, zerlegt in Wetteranteil und Rest

Modell = Schritt 1 aus event_effects.py (identischer Code aus event_effects_mcp.py), pro Station:
  log1p(flow) ~ Viertelstunde x Tagtyp + Ferien + Regen + Temperatur + Temperatur^2 + Wind + Trend
  Ridge, gefittet nur auf Zeiten ohne Event/Sperrung (3 h davor bis 2 h danach, bis 2 Stationen Abstand)

Zerlegung je Station und Viertelstunde, alles auf Log-Skala, es gilt: gesamt = wetter + rest
  normalized_flows.csv    gesamt = log1p(Ist) - Normal
  normalized_weather.csv  wetter = Anteil, den das Modell dem Wetter zuschreibt
                                   (Regen, Hitze/Kaelte, Wind gegenueber dem fuer die Uhrzeit typischen Wetter)
  normalized_rest.csv     rest   = gesamt - wetter -> Events, Sperrungen, echte Anomalien
  Normal = Viertelstunde x Tagtyp + Ferien + Trend + typisches Wetter fuer diese Uhrzeit,
           danach je Station x Viertelstunde x Tagtyp auf den Median nachzentriert
           (die Regression trifft den Mittelwert; Nullen im Flow ziehen den nach unten,
            so heisst 0 wieder: typischer Slot)
  0 normal, < 0 weniger, > 0 mehr;  0.1 ~ +10 %, -0.2 ~ -18 %, 0.69 ~ doppelt so viel

Dazu
  normal_flow_coefficients.csv   Regen-, Hitze- und Ferieneffekt je Station in %
  normal_flow_model.pkl          gefittetes Modell, um neue Daten mit demselben Normal zu bereinigen

Eingabe: flows_pre_innotrans.csv, weather_data_pre_innotrans.csv, ... (Endung per Umgebungsvariable
DATA_SUFFIX aenderbar; fehlt die Datei, wird die erste CSV mit passendem Namensanfang genommen)

Aufruf (event_effects_mcp.py muss im selben Ordner liegen)
  python normalize_flows.py --data ./data
  python normalize_flows.py --data ./data_neu --model ./data/normal_flow_model.pkl --suffix innotrans
  --suffix    Namensendung der Ausgabedateien (Standard: keine)
  --freq 1h   Stundensummen statt Viertelstunden (deutlich weniger Rauschen)
  optional: --geocode-cache analysis_output/geocode_cache.json  --venue-map venues.csv  --out ORDNER
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import event_effects_mcp as core  # noqa: E402

# Reihenfolge der Zusatzterme in core.baseline_design (hinter den Level-Dummies)
HOL, HOL_WK, RAIN, RAIN_1H, TEMP, TEMP2, WIND, TREND = range(8)


def pct(b):
    return (np.exp(b) - 1) * 100


def slot_of(idx):
    return np.asarray(idx.hour * 60 + idx.minute)


def group_keys(idx):
    """(Tageszeit, Tagtyp) wie die Level-Dummies der Baseline: Mo-Do, Fr, Sa, So."""
    dow = np.asarray(idx.dayofweek)
    return pd.MultiIndex.from_arrays([slot_of(idx), np.select([dow <= 3, dow == 4, dow == 5], [0, 1, 2], 3)])


# ----------------------------------------------------------------------------- Fit
def fit(data, venue_map=None, geocode=None, freq="15min"):
    """Baseline wie im Original: Event-/Sperrfenster raus, Ridge pro Station, dann Median-Nachzentrierung."""
    flows, weather, adj, episodes = core.load_real(data, venue_map, geocode)
    if freq != "15min":
        flows, weather = core.aggregate(flows, weather, freq)
    idx, stations = flows.index, list(flows.columns)
    sidx = {s: i for i, s in enumerate(stations)}
    Y = np.log1p(flows.to_numpy(float))

    excluded = np.zeros_like(Y, bool)
    for ep in episodes:
        hops = core.bfs_hops(adj, [a for a in ep["anchors"] if a in sidx], core.MAX_HOP)
        cols = [sidx[s] for s in hops if s in sidx]
        for _, i0, i1 in core.ep_segments(ep, idx):
            excluded[i0:i1, cols] = True
    print(f"[info] Fit ohne {excluded.mean():.1%} der Zellen (Event-/Sperrfenster)")

    bstats = core.baseline_stats(idx, weather)
    X, n_level = core.baseline_design(idx, weather, bstats)
    assert X.shape[1] == n_level + 8, "Spalten von baseline_design haben sich geaendert"
    _, B = core.fit_baseline(Y, X, excluded, n_level)

    # typisches Wetter je Uhrzeit (mittlere standardisierte Temperatur / Wind) -> gehoert zum Normal
    typical = pd.DataFrame({"temp": X[:, n_level + TEMP], "wind": X[:, n_level + WIND]},
                           index=slot_of(idx)).groupby(level=0).mean()

    # Median-Nachzentrierung: Median des Rests je Station x (Tageszeit, Tagtyp), nur saubere Zellen
    R = pd.DataFrame(np.where(excluded, np.nan, Y - X @ B), index=group_keys(idx), columns=stations)
    shift = R.groupby(level=[0, 1]).median()
    print(f"[info] Median-Nachzentrierung: im Mittel {np.nanmean(shift.to_numpy()):+.3f} (log)")

    model = dict(B=B, bstats=bstats, n_level=n_level, stations=stations, typical=typical,
                 shift=shift, freq=freq)
    return model, flows, weather


# ----------------------------------------------------------------------------- Zerlegung
def set_typical_weather(X, n, t_typ, w_typ, rows=slice(None)):
    X[rows, n + RAIN] = 0.0
    X[rows, n + RAIN_1H] = 0.0
    X[rows, n + TEMP] = t_typ[rows]
    X[rows, n + TEMP2] = t_typ[rows] ** 2
    X[rows, n + WIND] = w_typ[rows]


def decompose(model, flows, weather):
    stations = model["stations"]
    cols = [c for c in flows.columns if c in stations]
    lost = [c for c in flows.columns if c not in stations]
    if lost:
        print(f"[warn] {len(lost)} Stationen nicht im Modell, werden weggelassen: {lost[:5]}")
    B = model["B"][:, [stations.index(c) for c in cols]]
    n = model["n_level"]
    idx = flows.index

    X, _ = core.baseline_design(idx, weather, model["bstats"])
    typ = model["typical"].reindex(slot_of(idx))
    t_typ, w_typ = typ["temp"].fillna(0).to_numpy(), typ["wind"].fillna(0).to_numpy()

    miss = np.asarray((idx < weather.index.min()) | (idx > weather.index.max()))
    if miss.any():   # keine Wetterdaten -> typisches Wetter annehmen, Wetteranteil = 0
        set_typical_weather(X, n, t_typ, w_typ, miss)
        print(f"[warn] {int(miss.sum())} Zeitstempel ohne Wetterdaten - typisches Wetter angenommen")

    pred = X @ B                                   # Normal + tatsaechliches Wetter
    Xn = X.copy()
    set_typical_weather(Xn, n, t_typ, w_typ)
    normal = Xn @ B                                # Normal mit typischem Wetter fuer die Uhrzeit

    if "shift" in model:        # Median-Nachzentrierung gehoert zum Normal
        sh = model["shift"].reindex(group_keys(idx))[cols].fillna(0).to_numpy()
        pred, normal = pred + sh, normal + sh

    Y = np.log1p(flows[cols].to_numpy(float))
    frame = lambda a: pd.DataFrame(a.round(4), index=idx, columns=cols)
    return frame(Y - normal), frame(pred - normal), frame(Y - pred)


def coefficients(model):
    """Lesbare Effekte je Station in % (fuer Erklaerungen: 'Regen bringt hier +8 %')."""
    B, n, bs = model["B"], model["n_level"], model["bstats"]
    b = lambda k: B[n + k]
    z5 = 5 / (bs["temp"][1] + 1e-9)            # +5 Grad in Standardabweichungen
    return pd.DataFrame({
        "station": model["stations"],
        "rain_now_pct": pct(b(RAIN)),
        "rain_last_hour_pct": pct(b(RAIN_1H)),
        "rain_both_pct": pct(b(RAIN) + b(RAIN_1H)),
        "temp_plus5C_vs_mean_pct": pct(b(TEMP) * z5 + b(TEMP2) * z5 ** 2),
        "school_holiday_weekend_pct": pct(b(HOL)),
        "school_holiday_weekday_pct": pct(b(HOL) + b(HOL_WK)),
    }).round(2)


# ----------------------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ordner mit flows*.csv und weather_data*.csv (beim Fit alle CSVs)")
    ap.add_argument("--model", help="vorhandenes normal_flow_model.pkl verwenden statt neu zu fitten")
    ap.add_argument("--out", help="Zielordner (Standard: --data)")
    ap.add_argument("--freq", default="15min", help="Zeitraster, z.B. 1h (beim --model wird dessen Raster genommen)")
    ap.add_argument("--suffix", default="", help="Endung der Ausgabedateien, z.B. innotrans fuer neue Daten")
    ap.add_argument("--venue-map")
    ap.add_argument("--geocode-cache")
    a = ap.parse_args()
    out = a.out or a.data
    os.makedirs(out, exist_ok=True)
    sfx = f"_{a.suffix}" if a.suffix else ""

    if a.model:
        with open(a.model, "rb") as f:
            model = pickle.load(f)
        flows = pd.read_csv(core.pick(a.data, "flows"), parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        weather = pd.read_csv(core.pick(a.data, "weather_data"), index_col=0, parse_dates=True).sort_index()
        if model.get("freq", "15min") != "15min":
            flows, weather = core.aggregate(flows, weather, model["freq"])
        print(f"[info] Modell geladen: {a.model} (Raster {model.get('freq', '15min')})")
    else:
        model, flows, weather = fit(a.data, a.venue_map, a.geocode_cache, a.freq)
        with open(os.path.join(out, f"normal_flow_model{sfx}.pkl"), "wb") as f:
            pickle.dump(model, f)
        coef = coefficients(model)
        coef.to_csv(os.path.join(out, f"normal_flow_coefficients{sfx}.csv"), index=False)
        print(f"[info] Mitteltemperatur {model['bstats']['temp'][0]:.1f} C. Effekte im Netz (Median der Stationen):")
        print(coef.drop(columns="station").median().round(1).to_string())

    total, wpart, rest = decompose(model, flows, weather)
    for name, df in [("normalized_flows", total), ("normalized_weather", wpart), ("normalized_rest", rest)]:
        df.to_csv(os.path.join(out, f"{name}{sfx}.csv"))
    t, r = total.to_numpy(), rest.to_numpy()
    print(f"[info] {total.shape[0]} Zeitstempel x {total.shape[1]} Stationen -> {out}")
    print(f"[info] Median gesamt {np.nanmedian(t):.3f}, Median Rest {np.nanmedian(r):.3f}, Wetter erklaert {1 - np.nanvar(r) / np.nanvar(t):.1%} "
          f"der Varianz des bereinigten Flows")


if __name__ == "__main__":
    main()