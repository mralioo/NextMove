"""The two TabPFN point-prediction models the MCP server exposes
(`predict_overcrowding_risk` classifier, `predict_expected_flow` regressor),
with checkpointing so a server restart doesn't re-fit them.

Both are fitted on the same stratified 8,000-row sample of the chronological
train pool (first 80% of days), exactly as `ml/train_overcrowding_classifier.py`
evaluates. `load_or_fit_point_models` restores them from `ml/checkpoints/` when a
matching checkpoint exists (see checkpoints.py), otherwise fits and saves them.
"""
from __future__ import annotations

import pandas as pd

from checkpoints import authenticate, make_fingerprint, restore_checkpoint, save_checkpoint
from features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, encode_categoricals

SAMPLE_SIZE = 8_000
MODEL_PATH = "v3.5_default"
CLASSIFIER_NAME = "overcrowding_classifier"
REGRESSOR_NAME = "expected_flow_regressor"
TRAIN_FRACTION = 0.8


def train_pool_dates(table: pd.DataFrame) -> tuple[set, object]:
    """Chronological train-pool dates (first 80% of days) and the cutoff date."""
    dates = sorted(table["timestamp"].dt.date.unique())
    cutoff = dates[int(len(dates) * TRAIN_FRACTION)]
    return {d for d in dates if d < cutoff}, cutoff


def _sample(table: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    train_dates, cutoff = train_pool_dates(table)
    pool = table[table["timestamp"].dt.date.isin(train_dates)]
    n = min(SAMPLE_SIZE, len(pool))
    n_pos = min(int(round(n * max(pool["overcrowded"].mean(), 0.15))), int((pool["overcrowded"] == 1).sum()))
    n_neg = min(n - n_pos, int((pool["overcrowded"] == 0).sum()))
    sample = pd.concat([
        pool[pool["overcrowded"] == 1].sample(n_pos, random_state=0),
        pool[pool["overcrowded"] == 0].sample(n_neg, random_state=0),
    ]).sample(frac=1, random_state=0).reset_index(drop=True)
    return sample, cutoff


def _fingerprint(table: pd.DataFrame, cutoff, dataset_folder: str) -> str:
    return make_fingerprint(
        features=FEATURE_COLUMNS, model=MODEL_PATH, sample=SAMPLE_SIZE, cutoff=cutoff,
        n_rows=len(table), n_stations=table["station_name"].nunique(), dataset=dataset_folder)


def load_or_fit_point_models(table: pd.DataFrame, dataset_folder: str = "", force: bool = False) -> dict:
    """Returns {classifier, regressor, train_sample, cutoff_date, status:{name: status}}.
    status values: 'loaded' | 'refit-from-checkpoint' | 'fitted'."""
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

    train_dates, cutoff = train_pool_dates(table)
    fp = _fingerprint(table, cutoff, dataset_folder)

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        # categories fixed from the (checkpointed) train sample so codes never drift between runs
        return encode_categoricals(df[FEATURE_COLUMNS], reference=reference[FEATURE_COLUMNS])

    reference = None
    status: dict[str, str] = {}
    restored = {}
    if not force:
        # first restore needs a reference for encoding: read the saved sample itself
        from checkpoints import CHECKPOINT_DIR

        sample_path = CHECKPOINT_DIR / CLASSIFIER_NAME / "train_sample.csv.gz"
        if sample_path.exists():
            reference = pd.read_csv(sample_path)
            for name, cls, target in ((CLASSIFIER_NAME, TabPFNClassifier, "overcrowded"),
                                      (REGRESSOR_NAME, TabPFNRegressor, "passengers")):
                got = restore_checkpoint(name, cls, fp, encode, target)
                if got:
                    restored[name] = got

    if len(restored) == 2:
        clf, sample, _, s1 = restored[CLASSIFIER_NAME]
        reg, _, _, s2 = restored[REGRESSOR_NAME]
        return {"classifier": clf, "regressor": reg, "train_sample": sample, "cutoff_date": cutoff,
                "status": {CLASSIFIER_NAME: s1, REGRESSOR_NAME: s2}}

    authenticate()
    sample, cutoff = _sample(table)
    reference = sample
    X = encode(sample)
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]

    clf = TabPFNClassifier(model_path=MODEL_PATH, categorical_features_indices=cat_idx)
    clf.fit(X, sample["overcrowded"])
    reg = TabPFNRegressor(model_path=MODEL_PATH, categorical_features_indices=cat_idx)
    reg.fit(X, sample["passengers"])

    common = {"n_features": len(FEATURE_COLUMNS), "features": FEATURE_COLUMNS, "tabpfn_model": MODEL_PATH,
              "fingerprint": fp, "dataset_folder": dataset_folder, "train_days_before": str(cutoff)}
    save_checkpoint(CLASSIFIER_NAME, clf, sample, {
        **common, "task": "classification", "target": "overcrowded (flow >= station's own p90)",
        "description": "Overcrowding-risk classifier behind predict_overcrowding_risk"})
    save_checkpoint(REGRESSOR_NAME, reg, sample, {
        **common, "task": "regression", "target": "passengers (per station per 15 min)",
        "description": "Expected-flow regressor behind predict_expected_flow"})
    return {"classifier": clf, "regressor": reg, "train_sample": sample, "cutoff_date": cutoff,
            "status": {CLASSIFIER_NAME: "fitted", REGRESSOR_NAME: "fitted"}}
