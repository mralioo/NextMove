"""scenario_flow — turn a closure + the TabPFN demand baseline into per-station
pressure estimates. Last step of Category C:

    resolve_closure -> apply_closure -> alternate_paths -> scenario_flow

For each 15-min slot in the closure window:
  1. baseline demand per involved station = TabPFN predictive distribution (mean + quantiles)
     (ml/demand_baseline.py)
  2. displaced demand = baseline expected demand (mean) of unserved stations (100%) — and, for a
     line-section suspension, `diversion_share` x half the flow of the section
     endpoints / still-served interchange stations (crossing passengers)
  3. unserved stations' demand spills to the nearest open stations
     (ml/disruption.py::spill_targets); crossing passengers load the transfer
     stations of the alternate paths (path_weights)
  4. pressure = P(baseline + added > the station's own p95 for that hour/day-type),
     read off the TabPFN quantile grid

`diversion_share` is an ASSUMPTION, so the result is reported for low/base/high
values, never as a single confident number. See disruption.ASSUMPTIONS.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from disruption import (
    ASSUMPTIONS,
    DEFAULT_DIVERSION_SHARES,
    ClosureEffect,
    Network,
    exceed_probability,
    redistribution_plan,
)
from demand_baseline import MEAN_COL, MEDIAN_IDX, Q_COLS, QUANTILES, DemandBaseline


def _f(x) -> float:
    return round(float(x), 1)


def run_scenario(net: Network, baseline: DemandBaseline, effect: ClosureEffect,
                 shares: dict[str, float] | None = None, top_n: int = 10) -> dict:
    shares = shares or DEFAULT_DIVERSION_SHARES
    spec = effect.spec
    start, end = pd.to_datetime(spec["start"]), pd.to_datetime(spec["end"])
    plan = redistribution_plan(net, effect)

    involved = set(plan["sources"]) | {r for m in plan["spill"].values() for r in m} | set(plan["transfer"])
    t = baseline.table
    rows = t[t["station_name"].isin(involved) & (t["timestamp"] >= start) & (t["timestamp"] < end)]
    if rows.empty:
        return {"error": "No dataset rows for the involved stations in this window "
                         f"({start} – {end}); scenario_flow only supports windows inside the "
                         "dataset coverage (see describe_dataset)."}

    quant = baseline.predict_quantiles(rows)
    df = rows[["timestamp", "station_name", "passengers", "station_hour_p95"]].join(quant)
    med = df.pivot(index="timestamp", columns="station_name", values=MEAN_COL)  # expected demand
    slots = list(med.index)

    crossing_stations = [s for s, k in plan["sources"].items() if k in ("endpoint", "partial")]
    unserved = [s for s, k in plan["sources"].items() if k == "unserved"]

    # ---- displaced demand (share-independent part: unserved stations) ----
    displaced = {}
    for s in unserved:
        v = med[s] if s in med else pd.Series(0.0, index=slots)
        displaced[s] = {
            "avg_baseline_per_15min": _f(v.mean()),
            "peak_baseline_per_15min": _f(v.max()),
            "total_displaced_over_window": _f(v.sum()),
            "note": "baseline (counterfactual) demand at a station with no service",
        }

    # ---- added load per (station, slot) for every assumption level ----
    def added_load(share: float) -> dict[tuple[str, pd.Timestamp], float]:
        add: dict = defaultdict(float)
        for ts in slots:
            for s in unserved:
                d = float(med.loc[ts, s]) if s in med else 0.0
                for recv, w in plan["spill"].get(s, {}).items():
                    add[(recv, ts)] += w * d
            if plan["transfer"] and crossing_stations:
                crossing = share * 0.5 * sum(float(med.loc[ts, s]) for s in crossing_stations if s in med)
                for stn, w in plan["transfer"].items():
                    add[(stn, ts)] += w * crossing
        return add

    added = {label: added_load(sh) for label, sh in shares.items()}
    receivers = sorted({k[0] for a in added.values() for k in a})

    def station_table(label: str) -> dict[str, dict]:
        out = {}
        for r in receivers:
            g = df[df["station_name"] == r].set_index("timestamp")
            best = None
            for ts in slots:
                if ts not in g.index:
                    continue
                add = added[label].get((r, ts), 0.0)
                qv = g.loc[ts, Q_COLS].to_numpy(float)
                p95 = float(g.loc[ts, "station_hour_p95"])
                p_exc = exceed_probability(p95 - add, QUANTILES, list(qv))
                p_base = exceed_probability(p95, QUANTILES, list(qv))
                cand = (p_exc, add, ts, float(qv[MEDIAN_IDX]), p95, p_base, float(g.loc[ts, MEAN_COL]))
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if best is None:
                continue
            p_exc, add, ts, q50, p95, p_base, mean_ = best
            out[r] = {
                "peak_slot": str(ts), "baseline_expected": _f(mean_), "baseline_median": _f(q50), "added_load": _f(add),
                "expected_total": _f(mean_ + add), "own_p95_reference": _f(p95),
                "pct_of_own_p95": round((mean_ + add) / p95 * 100, 0) if p95 else None,
                "prob_exceed_own_p95": p_exc, "prob_exceed_without_closure": p_base,
                "avg_added_over_window": _f(np.mean([added[label].get((r, x), 0.0) for x in slots])),
            }
        return out

    tables = {label: station_table(label) for label in shares}
    base_label = "base" if "base" in shares else next(iter(shares))
    ranked = sorted(tables[base_label].items(),
                    key=lambda kv: (kv[1]["prob_exceed_own_p95"], kv[1]["added_load"]), reverse=True)

    ranked_out = []
    for name, v in ranked[:top_n]:
        row = {"station_name": name, "lines": net.lines_of.get(name, []), **v}
        for label in shares:
            if label != base_label and name in tables[label]:
                row[f"prob_exceed_own_p95_{label}_assumption"] = tables[label][name]["prob_exceed_own_p95"]
                row[f"added_load_{label}_assumption"] = tables[label][name]["added_load"]
        ranked_out.append(row)

    out = {
        "closure": {k: spec.get(k) for k in ("closure_id", "source", "kind", "line", "station",
                                              "from_station", "to_station", "start", "end", "reason")},
        "window_slots_15min": len(slots),
        "assumed_diversion_shares": shares,
        "displaced_unserved_stations": displaced,
        "ranked_pressure_stations": ranked_out,
        "n_receiver_stations": len(receivers),
        "data_mode": "model-based estimate, NOT a measurement",
        "assumptions": ASSUMPTIONS,
        "model": "TabPFNRegressor quantile baseline (see ml/demand_baseline.py)",
    }

    # ---- reality check against the historical rows, when the closure is in the dataset ----
    if spec.get("source") == "closures.csv":
        chk = []
        for name, v in ranked[:top_n]:
            g = df[df["station_name"] == name]
            if len(g):
                chk.append({
                    "station_name": name,
                    "observed_avg_per_15min": _f(g["passengers"].mean()),
                    "baseline_expected_avg_per_15min": _f(g[MEAN_COL].mean()),
                    "assumed_avg_with_redistribution": _f(
                        g[MEAN_COL].mean() + v["avg_added_over_window"]),
                })
        out["historical_reality_check"] = {
            "rows": chk,
            "reading": "Compare observed vs baseline: if observed ≈ baseline, the dataset shows NO "
                       "measurable redistribution at that station during this closure, so the "
                       "assumed_with_redistribution figure is scenario planning, not a replay.",
        }
    return out
