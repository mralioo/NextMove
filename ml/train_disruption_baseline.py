"""Train + evaluate the TabPFN demand baseline behind Category C (disruption
response), then run the 26-closure case study.

Usage:
    make train-disruption
    # or: ./.venv/bin/python ml/train_disruption_baseline.py

Requires TABPFN_API_TOKEN (same as ml/train_overcrowding_classifier.py).

What it does
  1. Builds the station x 15-min feature table, excludes every closure window,
     splits chronologically (first 80% of days = train pool).
  2. Fits TabPFNRegressor on a 10k-row sample; scores the held-out last 20% of
     days: MAE/RMSE/R2 vs naive baselines + prediction-interval calibration.
  3. CASE STUDY — for each closure in closures.csv, compares OBSERVED flow
     during the window with the model's counterfactual distribution, by role:
       closed/unserved | section endpoints + interchanges | 1 hop | 2 hops.
     If redistribution existed in the data, neighbour stations would sit ABOVE
     their interval. If they sit inside it at the nominal rate, the dataset
     contains no measurable redistribution — the reason the solver labels its
     redistribution as an assumption.
  4. Writes to ml/output/: disruption_baseline_report.json, disruption_case_study.csv,
     disruption_heldout_predictions.csv, disruption_showcase_series.csv — all read by
     the dashboard's "ML Engine" page (predictions vs ground truth).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

from env_loader import load_all_dotenvs  # noqa: E402
from demand_baseline import MEAN_COL, Q_COLS, DemandBaseline, q_col  # noqa: E402
from disruption import Network, _closure_to_spec, apply_closure  # noqa: E402
from features import build_feature_table  # noqa: E402
from table_cache import cached_feature_table  # noqa: E402
from utils.data_loader import (  # noqa: E402
    DEFAULT_DATA_DIR,
    discover_dataset_dirs,
    load_closures,
    load_connections,
    load_flows,
    load_stations,
    station_cols,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build_baseline(folder: str) -> tuple[DemandBaseline, Network, pd.DataFrame]:
    """Shared with the MCP server: prepared (unfitted) baseline + network + closures."""
    closures = load_closures(folder)
    net = Network.from_frames(load_stations(folder), load_connections(folder),
                              station_cols(load_flows(folder)))
    return DemandBaseline.prepare(cached_feature_table(folder), closures, dataset_folder=folder), net, closures


def case_study(base: DemandBaseline, net: Network, closures: pd.DataFrame) -> pd.DataFrame:
    """One row per (closure, station, role) aggregated over the closure window."""
    t = base.table
    recs = []
    for i in range(len(closures)):
        spec = _closure_to_spec(net, i, closures.iloc[i])
        if spec.get("warnings"):
            continue
        eff = apply_closure(net, spec)
        start, end = pd.to_datetime(spec["start"]), pd.to_datetime(spec["end"])

        role: dict[str, str] = {s: "closed_unserved" for s in eff.unserved}
        for s in eff.affected_stations:
            role.setdefault(s, "endpoint_or_interchange")
        # hop rings around the closed footprint, in the ORIGINAL graph
        dist: dict[str, int] = {}
        for src in eff.section_path:
            for n, d in nx.single_source_shortest_path_length(net.g, src, cutoff=2).items():
                dist[n] = min(dist.get(n, 9), d)
        for n, d in dist.items():
            if n not in role and d in (1, 2):
                role[n] = f"hop{d}"

        rows = t[t["station_name"].isin(role) & (t["timestamp"] >= start) & (t["timestamp"] < end)]
        if rows.empty:
            continue
        pred = base.predict_quantiles(rows, exact_mean=True)
        bl = base.baseline_columns(rows)
        df = rows[["timestamp", "station_name", "passengers"]].join(pred).join(bl)
        df["closure_id"] = i
        df["role"] = df["station_name"].map(role)
        df["kind"] = spec["kind"]
        df["in80"] = (df["passengers"] >= df[q_col(0.1)]) & (df["passengers"] <= df[q_col(0.9)])
        df["above_q90"] = df["passengers"] > df[q_col(0.9)]
        df["below_q10"] = df["passengers"] < df[q_col(0.1)]
        df["expected"] = df[MEAN_COL]
        df["q10"], df["q90"] = df[q_col(0.1)], df[q_col(0.9)]
        df["slot_mean"] = df["baseline_mean"]                       # naive baseline: station x slot mean
        df["b_q90"] = df[f"b_{q_col(0.9)}"]                         # naive baseline: empirical station x hour q90
        df["baseline_above_q90"] = df["passengers"] > df["b_q90"]
        recs.append(df[["closure_id", "kind", "station_name", "role", "timestamp",
                        "passengers", "expected", "q10", "q90", "slot_mean", "b_q90", "in80",
                        "above_q90", "below_q10", "baseline_above_q90"]])
    return pd.concat(recs, ignore_index=True)


def summarise_case_study(cs: pd.DataFrame) -> dict:
    out = {}
    for (kind, role), g in cs.groupby(["kind", "role"]):
        out[f"{kind}/{role}"] = {
            "rows": int(len(g)),
            "observed_over_expected": round(float(g["passengers"].sum() / max(g["expected"].sum(), 1e-9)), 3),
            "observed_over_slot_mean": round(float(g["passengers"].sum() / max(g["slot_mean"].sum(), 1e-9)), 3),
            "baseline_share_above_q90": round(float(g["baseline_above_q90"].mean()), 3),
            "share_inside_80pct_interval": round(float(g["in80"].mean()), 3),
            "share_above_q90": round(float(g["above_q90"].mean()), 3),
            "share_below_q10": round(float(g["below_q10"].mean()), 3),
        }
    return out


def main() -> None:
    load_all_dotenvs()
    folder = str(next(iter(discover_dataset_dirs(DEFAULT_DATA_DIR).values())))
    print(f"Dataset: {folder}")
    base, net, closures = build_baseline(folder)
    print(f"Feature table: {len(base.table):,} rows | train pool {int(base.train_pool_mask.sum()):,} | "
          f"held-out pool {int(base.test_pool_mask.sum()):,} | cutoff {base.cutoff_date}")

    base.fit()
    metrics = base.evaluate()
    ckpt = base.save_checkpoint(metrics=metrics)
    print(f"Checkpoint saved: {ckpt}")
    print("\n== Held-out evaluation (last 20% of days, no closure windows) ==")
    print(json.dumps(metrics, indent=2))

    cs = case_study(base, net, closures)
    summary = summarise_case_study(cs)
    print("\n== Case study: 26 historical closures — observed vs counterfactual baseline ==")
    print("(nominal: 80% inside interval, 10% above q90, 10% below q10; 'observed/expected' ~ 1 = no effect)")
    for k, v in summary.items():
        print(f"  {k:42s} {v}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    cs.to_csv(OUTPUT_DIR / "disruption_case_study.csv", index=False)
    base.heldout_frame.round(2).to_csv(OUTPUT_DIR / "disruption_heldout_predictions.csv", index=False)
    print("Building contiguous showcase series (busiest/median/quiet stations, last 3 held-out days)...")
    base.showcase().round(2).to_csv(OUTPUT_DIR / "disruption_showcase_series.csv", index=False)
    (OUTPUT_DIR / "disruption_baseline_report.json").write_text(
        json.dumps({"heldout_metrics": metrics, "case_study_summary": summary,
                    "n_closures": int(cs["closure_id"].nunique())}, indent=2))
    print(f"\nWrote {OUTPUT_DIR / 'disruption_baseline_report.json'}")


if __name__ == "__main__":
    main()
