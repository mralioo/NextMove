"""Does the pressure ranking (Category P) have skill? Replay days only, where the observed answer is known.

    ./.venv/bin/python evaluation/validate_pressure.py [--days 8]

For each replay day, rank ALL stations (screen=200) and compare with what was observed at the same 8 peak slots:
  * load ranking      predicted 90th-percentile load  vs  observed peak load           (what `rank_pressure` returns)
  * p95 ranking       mean probability of exceeding the station's own p95  vs  observed number of slots above it
Reports Spearman correlation, the observed peak of the predicted top-3 against the all-station mean, and the top-3 overlap
(chance level is about 3*3/167 = 0.05). Writes knowledge/validation.json["pressure"], which the knowledge base cites.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "agent"), str(REPO)]

DAYS = ["2026-09-16", "2026-09-15", "2026-09-10", "2026-09-05", "2026-08-20", "2026-07-14", "2026-08-05", "2026-07-29"]


async def main() -> None:
    import numpy as np
    from scipy.stats import spearmanr

    from mcp_runtime import get_runtime

    n = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else len(DAYS)
    rt = get_runtime()
    rows = []
    for d in DAYS[:n]:
        r = await rt.call("rank_pressure", date=d, top_n=3, screen=200)
        if "error" in r or not r.get("predicted_all"):
            print(d, "skipped", r.get("error"))
            continue
        pa, oa, pp, oe = r["predicted_all"], r["observed_all"], r["predicted_p_all"], r["observed_exceed_all"]
        ks = [k for k in pa if k in oa and k in pp and k in oe]
        rho_l = spearmanr([pa[k] for k in ks], [oa[k] for k in ks])[0]
        rho_p = spearmanr([pp[k] for k in ks], [oe[k] for k in ks])[0]
        top = [x["s"] for x in r["top"]]
        rows.append({"date": d, "spearman_load": float(rho_l), "spearman_p95": float(rho_p), "obs_peak_top3": float(np.mean([oa[k] for k in top])),
                     "obs_peak_all": float(np.mean(list(oa.values()))), "overlap_top3": int(r["observed_overlap"])})
        print(d, {k: round(v, 2) if isinstance(v, float) else v for k, v in rows[-1].items()})
    out = {"n_days": len(rows), "spearman_load": float(np.mean([x["spearman_load"] for x in rows])), "spearman_p95": float(np.mean([x["spearman_p95"] for x in rows])),
           "obs_peak_of_pred_top3": float(np.mean([x["obs_peak_top3"] for x in rows])), "obs_peak_mean_all": float(np.mean([x["obs_peak_all"] for x in rows])),
           "mean_overlap_top3": float(np.mean([x["overlap_top3"] for x in rows])), "chance_overlap_top3": 3 * 3 / 167, "days": rows}
    p = REPO / "knowledge" / "validation.json"
    p.parent.mkdir(exist_ok=True)
    v = json.loads(p.read_text()) if p.exists() else {}
    v["pressure"] = out
    p.write_text(json.dumps(v, indent=1))
    print("\nload ranking: Spearman %.2f, predicted top-3 observed peak %.0f vs all-station mean %.0f, overlap %.2f/3 (chance %.2f)" %
          (out["spearman_load"], out["obs_peak_of_pred_top3"], out["obs_peak_mean_all"], out["mean_overlap_top3"], out["chance_overlap_top3"]))
    print("p95-exceedance ranking: Spearman %.2f (no skill)" % out["spearman_p95"])


if __name__ == "__main__":
    asyncio.run(main())
