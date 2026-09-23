"""Train a TabPFN classifier to predict station-timeslot overcrowding risk.

Usage:
    make train-overcrowding
    # or directly:
    ./.venv/bin/python ml/train_overcrowding_classifier.py

Requires a TabPFN API token. Put it in a `.env` file anywhere above this
repo's working directory (python-dotenv walks up looking for one) as:

    TABPFN_API_TOKEN=tabpfn_sk_...

Get a token at https://platform.priorlabs.ai/account/api-keys

TabPFN is an in-context tabular foundation model, not a big-data model — it's
designed for at most a few thousand to ~10k training rows. Our melted
station x 15-min table has ~1.4M rows, so this script takes a stratified,
chronologically-split sample rather than trying to feed the whole thing in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_loader import load_all_dotenvs  # noqa: E402
from features import FEATURE_COLUMNS, build_feature_table, encode_categoricals  # noqa: E402
from utils.data_loader import DEFAULT_DATA_DIR, discover_dataset_dirs  # noqa: E402

TRAIN_SAMPLE_SIZE = 8_000
TEST_SAMPLE_SIZE = 2_000
TEST_HOLDOUT_FRACTION = 0.2  # last 20% of days, by date, held out before any sampling
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_api_token() -> str:
    loaded = load_all_dotenvs()
    token = os.environ.get("TABPFN_API_TOKEN")
    if not token:
        raise SystemExit(
            "No TABPFN_API_TOKEN found. Add it to a .env file "
            f"(searched from {Path.cwd()} upward"
            f"{', found: ' + ', '.join(str(p) for p in loaded) if loaded else ', none found'}) "
            "as: TABPFN_API_TOKEN=tabpfn_sk_..."
        )
    return token


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["timestamp"].dt.date.unique())
    cutoff_idx = int(len(dates) * (1 - TEST_HOLDOUT_FRACTION))
    cutoff_date = dates[cutoff_idx]
    train_pool = df[df["timestamp"].dt.date < cutoff_date]
    test_pool = df[df["timestamp"].dt.date >= cutoff_date]
    return train_pool, test_pool


def stratified_subsample(df: pd.DataFrame, n: int, label_col: str, seed: int = 0) -> pd.DataFrame:
    n = min(n, len(df))
    frac_positive = df[label_col].mean()
    n_pos = int(round(n * max(frac_positive, 0.15)))  # oversample the minority a bit for a cleaner signal
    n_pos = min(n_pos, (df[label_col] == 1).sum())
    n_neg = min(n - n_pos, (df[label_col] == 0).sum())
    pos = df[df[label_col] == 1].sample(n_pos, random_state=seed)
    neg = df[df[label_col] == 0].sample(n_neg, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    token = load_api_token()
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier

    tabpfn_client.set_access_token(token)

    folder = str(next(iter(discover_dataset_dirs(DEFAULT_DATA_DIR).values())))
    print(f"Building feature table from: {folder}")
    table = build_feature_table(folder)
    print(f"Feature table: {table.shape[0]:,} rows, overcrowded rate = {table['overcrowded'].mean():.1%}")

    train_pool, test_pool = chronological_split(table)
    train_df = stratified_subsample(train_pool, TRAIN_SAMPLE_SIZE, "overcrowded", seed=0)
    test_df = stratified_subsample(test_pool, TEST_SAMPLE_SIZE, "overcrowded", seed=1)
    print(f"Train sample: {len(train_df):,} rows ({train_df['overcrowded'].mean():.1%} positive)")
    print(f"Test sample:  {len(test_df):,} rows ({test_df['overcrowded'].mean():.1%} positive), "
          f"chronologically after train (no date overlap)")

    combined = pd.concat([train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]], keys=["train", "test"])
    combined = encode_categoricals(combined)
    X_train = combined.loc["train"].reset_index(drop=True)
    X_test = combined.loc["test"].reset_index(drop=True)
    y_train = train_df["overcrowded"].reset_index(drop=True)
    y_test = test_df["overcrowded"].reset_index(drop=True)
    categorical_idx = [FEATURE_COLUMNS.index("primary_line")]

    print("Fitting TabPFNClassifier...")
    model = TabPFNClassifier(model_path="v3.5_default", categorical_features_indices=categorical_idx)
    model.fit(X_train, y_train)

    print("Predicting on held-out test sample...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print()
    print("=== Evaluation (chronologically held-out test sample) ===")
    print(classification_report(y_test, y_pred, target_names=["normal", "overcrowded"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.3f}")
    print("Confusion matrix [rows=true, cols=pred] (normal, overcrowded):")
    print(confusion_matrix(y_test, y_pred))

    # Naive baseline for the dashboard's comparison: historical overcrowding rate of the same
    # station, day-type and hour, learned from the train pool only (no model, no weather).
    rate = (train_pool.groupby(["station_name", "is_weekend", "hour"])["overcrowded"].mean()
            .rename("baseline_probability").reset_index())
    test_keys = test_df[["station_name", "is_weekend", "hour"]].merge(
        rate, on=["station_name", "is_weekend", "hour"], how="left")
    baseline_p = test_keys["baseline_probability"].fillna(float(train_pool["overcrowded"].mean())).to_numpy()
    print(f"Baseline (station x day-type x hour rate): ROC-AUC {roc_auc_score(y_test, baseline_p):.3f}, "
          f"PR-AUC {average_precision_score(y_test, baseline_p):.3f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = test_df[["timestamp", "station_name", "passengers", "overcrowded"]].reset_index(drop=True)
    out["baseline_probability"] = baseline_p
    out["predicted_overcrowded"] = y_pred
    out["overcrowd_probability"] = y_proba
    out_path = OUTPUT_DIR / "overcrowding_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nPredictions written to {out_path}")


if __name__ == "__main__":
    main()
