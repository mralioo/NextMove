"""Checkpoints for the TabPFN models, so inference never needs a fresh `fit()`.

TabPFN is an in-context foundation model served by an API: there are no local
weights to pickle. "Fitting" uploads the training rows and the server returns a
`model_id`; `predict()` then references that fit. What `estimator.save_model()`
stores is just that reference plus hyperparameters (no data), and it only stays
valid while the server keeps the fit.

So a checkpoint here is a directory, `ml/checkpoints/<name>/`, holding:

    model.json          tabpfn-client record (server model_id, hyperparameters)
    train_sample.csv.gz the exact rows the model was fitted on (deterministic refit
                        + encoder reference + the `seen_in_training_sample` flag)
    meta.json           task, feature list, fingerprint, metrics, when/how it was saved

`restore_checkpoint()` tries `load_model()` and probes it with one prediction. If
the server has forgotten the fit it silently REFITS from `train_sample.csv.gz` and
rewrites `model.json` — same data, same hyperparameters, same seed — and reports
`status="refit-from-checkpoint"`. A checkpoint whose fingerprint no longer matches
(different dataset, features, split or model version) is treated as stale and
ignored, so a new dataset can never be served by an old model.

Loading needs the same TabPFN account/token that fitted the model.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", REPO_ROOT / "ml" / "checkpoints"))


def make_fingerprint(**parts) -> str:
    """Stable hash of everything that must match for a checkpoint to be reusable."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def authenticate() -> None:
    token = os.environ.get("TABPFN_API_TOKEN")
    if not token:
        raise RuntimeError("TABPFN_API_TOKEN not set (put it in a .env file above this repo).")
    import tabpfn_client

    tabpfn_client.set_access_token(token)


def _dir(name: str) -> Path:
    return CHECKPOINT_DIR / name


def save_checkpoint(name: str, estimator, train_sample: pd.DataFrame, meta: dict) -> Path:
    d = _dir(name)
    d.mkdir(parents=True, exist_ok=True)
    record = estimator.save_model(d / "model.json")
    train_sample.to_csv(d / "train_sample.csv.gz", index=False)
    import tabpfn_client

    full_meta = {
        **meta,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tabpfn_client_version": getattr(tabpfn_client, "__version__", record.get("tabpfn_client_version")),
        "model_id": record.get("model_id"),
        "n_train_rows": int(len(train_sample)),
    }
    (d / "meta.json").write_text(json.dumps(full_meta, indent=2, default=str) + "\n")
    write_manifest()
    return d


def restore_checkpoint(
    name: str, estimator_cls, fingerprint: str, encode: Callable[[pd.DataFrame], pd.DataFrame],
    target_col: str,
):
    """Returns (estimator, train_sample, meta, status) or None if there is no usable checkpoint.
    status is 'loaded' (server still had the fit) or 'refit-from-checkpoint'."""
    d = _dir(name)
    if not (d / "meta.json").exists() or not (d / "model.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("fingerprint") != fingerprint:
        return None  # stale: dataset / features / split / model version changed
    sample = pd.read_csv(d / "train_sample.csv.gz")
    if "timestamp" in sample.columns:
        sample["timestamp"] = pd.to_datetime(sample["timestamp"])

    authenticate()
    est = estimator_cls.load_model(d / "model.json")
    try:
        est.predict(encode(sample.head(1)))          # probe: does the server still hold the fit?
        return est, sample, meta, "loaded"
    except Exception:
        est.fit(encode(sample), sample[target_col])   # deterministic refit from the saved rows
        est.save_model(d / "model.json")
        meta["refit_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["model_id"] = est.model_id_
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        write_manifest()
        return est, sample, meta, "refit-from-checkpoint"


def write_manifest() -> Path:
    """ml/checkpoints/manifest.json — one entry per checkpoint, read by the dashboard."""
    entries = []
    for meta_path in sorted(CHECKPOINT_DIR.glob("*/meta.json")):
        m = json.loads(meta_path.read_text())
        entries.append({k: m.get(k) for k in (
            "name", "task", "description", "target", "n_features", "n_train_rows", "model_id",
            "saved_at", "refit_at", "tabpfn_model", "tabpfn_client_version", "fingerprint", "metrics",
            "dataset_folder", "train_days_before")})
    path = CHECKPOINT_DIR / "manifest.json"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, default=str) + "\n")
    return path
