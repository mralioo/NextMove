"""Build (or verify) the checkpoints of every TabPFN model used at inference time.

Usage:
    make checkpoints                 # reuse valid checkpoints, fit + save only what is missing/stale
    make checkpoints FORCE=1         # refit everything and overwrite

Checkpoints (ml/checkpoints/<name>/, see ml/checkpoints.py for the format):
    demand_baseline           TabPFNRegressor  quantile demand baseline  (Category C, scenario_flow)
    overcrowding_classifier   TabPFNClassifier overcrowding risk         (predict_overcrowding_risk)
    expected_flow_regressor   TabPFNRegressor  expected flow              (predict_expected_flow)

The MCP server restores these on start-up, so inference never pays for a fresh fit.
Requires TABPFN_API_TOKEN (the same account must load the checkpoints later).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from env_loader import load_all_dotenvs  # noqa: E402
from checkpoints import CHECKPOINT_DIR, write_manifest  # noqa: E402
from features import build_feature_table  # noqa: E402
from inference_models import load_or_fit_point_models  # noqa: E402
from train_disruption_baseline import build_baseline  # noqa: E402
from utils.data_loader import DEFAULT_DATA_DIR, discover_dataset_dirs  # noqa: E402


def main() -> None:
    force = "--force" in sys.argv
    load_all_dotenvs()
    folder = str(next(iter(discover_dataset_dirs(DEFAULT_DATA_DIR).values())))
    print(f"Dataset: {folder} | force={force}")

    t0 = time.time()
    base, _net, _closures = build_baseline(folder)
    status = {"demand_baseline": base.restore_or_fit(force=force)}
    # The point models reuse the same feature table the baseline was prepared from.
    point = load_or_fit_point_models(base.table, dataset_folder=folder, force=force)
    status.update(point["status"])

    write_manifest()
    print(f"\nCheckpoints in {CHECKPOINT_DIR} ({time.time() - t0:.0f}s):")
    for m in json.loads((CHECKPOINT_DIR / "manifest.json").read_text()):
        print(f"  {m['name']:26s} {status.get(m['name'], '?'):22s} {m['task']:15s} "
              f"rows={m['n_train_rows']:<6} model_id={m['model_id']}")
    print("\nstatus: loaded = server still had the fit | refit-from-checkpoint = server had "
          "forgotten it, refitted from the saved sample | fitted = new")


if __name__ == "__main__":
    main()
