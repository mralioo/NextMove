"""TabPFN counterfactual demand baseline — the ML engine behind Category C.

Question it answers: "what would this station's passenger flow have been in
this 15-min slot if nothing were wrong?" — with an honest prediction interval.

Why a *baseline* model and not a "closure uplift" model: the dataset has only
26 closures, and the simulated flows show no measurable redistribution around
them (only the closed station itself drops to exactly 0; neighbour flows are
statistically indistinguishable from noise — see docs/disruption_case_study.md).
A supervised uplift model would fit noise. What TabPFN regression is genuinely
good at here is the other half of the problem: a calibrated *distribution* of
normal demand per station/slot, learned from ~10k in-context examples. The
Category C solver then adds an explicit, labelled ASSUMPTION for how displaced
demand redistributes, and asks the model how likely the result is to exceed the
station's own p95.

Regression capability used: `TabPFNRegressor.predict(output_type="quantiles")`
(https://docs.priorlabs.ai/capabilities/regression) — one call returns the
predictive distribution's quantiles, not just a point estimate.

Leakage rules (enforced in `prepare`):
  * every row inside ANY closure window is excluded from fitting, profile
    features and the p95 reference (closed stations read 0 by simulation);
  * chronological split: the first 80% of days are the train pool, the last 20%
    are held out for evaluation;
  * target-encoded profile features (`station_slot_mean`, `station_hour_p95`)
    are computed from the train pool only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from checkpoints import authenticate, make_fingerprint, restore_checkpoint, save_checkpoint

QUANTILES = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
MEDIAN_IDX = QUANTILES.index(0.5)
TRAIN_SAMPLE_SIZE = 10_000
TRAIN_FRACTION = 0.8
PREDICT_CHUNK = 2_000
MODEL_PATH = "v3.5_default"

BASELINE_FEATURES = [
    "hour", "slot", "dow", "is_weekend", "month",
    "station_avg_passengers", "station_slot_mean", "n_lines", "is_interchange", "primary_line",
    "temp", "prcp", "wspd", "cldc", "coco",
    "daily_event_count", "daily_event_attendance",
]
CATEGORICAL = ["primary_line"]


def q_col(q: float) -> str:
    return f"q{int(round(q * 1000)):04d}"


Q_COLS = [q_col(q) for q in QUANTILES]
MEAN_COL = "mean"


def _regressor_cls():
    from tabpfn_client import TabPFNRegressor

    return TabPFNRegressor


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


@dataclass
class DemandBaseline:
    table: pd.DataFrame                      # prepared feature table (all rows)
    cutoff_date: object
    train_pool_mask: pd.Series
    test_pool_mask: pd.Series
    _categories: list[str] = field(default_factory=list)
    _model: object = None
    train_sample: pd.DataFrame | None = None
    heldout_frame: pd.DataFrame | None = None   # set by evaluate(): per-row prediction vs truth
    dataset_folder: str = ""
    checkpoint_status: str = "not-fitted"       # 'fitted' | 'loaded' | 'refit-from-checkpoint'
    _slot_median: pd.DataFrame | None = None    # naive baseline: station x day-type x slot median
    _hour_q: pd.DataFrame | None = None         # naive baseline: station x day-type x hour empirical quantiles

    # ------------------------------------------------------------------ prepare
    @classmethod
    def prepare(cls, table: pd.DataFrame, closures: pd.DataFrame, dataset_folder: str = "") -> "DemandBaseline":
        t = table.copy()
        ts = t["timestamp"]
        t["slot"] = ts.dt.hour * 4 + ts.dt.minute // 15
        in_window = pd.Series(False, index=t.index)
        for _, c in closures.iterrows():
            in_window |= (ts >= c["when"]) & (ts < c["end"])
        t["in_closure_window"] = in_window

        dates = np.sort(ts.dt.date.unique())
        cutoff = dates[int(len(dates) * TRAIN_FRACTION)]
        is_train_day = ts.dt.date < cutoff
        train_pool = is_train_day & ~in_window
        test_pool = ~is_train_day & ~in_window

        pool = t[train_pool]
        prof = (pool.groupby(["station_name", "is_weekend", "slot"])["passengers"].mean()
                .rename("station_slot_mean").reset_index())
        p95 = (pool.groupby(["station_name", "is_weekend", "hour"])["passengers"].quantile(0.95)
               .rename("station_hour_p95").reset_index())
        t = t.merge(prof, on=["station_name", "is_weekend", "slot"], how="left")
        t = t.merge(p95, on=["station_name", "is_weekend", "hour"], how="left")
        t["station_slot_mean"] = t["station_slot_mean"].fillna(t["station_avg_passengers"])
        t["station_hour_p95"] = t["station_hour_p95"].fillna(t["station_avg_passengers"] * 2)
        # merge() reset the index; realign the masks to the new (0..n-1) index
        train_pool = train_pool.reset_index(drop=True)
        test_pool = test_pool.reset_index(drop=True)
        t = t.reset_index(drop=True)

        obj = cls(table=t, cutoff_date=cutoff, train_pool_mask=train_pool, test_pool_mask=test_pool,
                  dataset_folder=str(dataset_folder))
        obj._categories = sorted(t["primary_line"].unique())

        # Naive baselines the model must beat, built from the SAME train pool (no leakage):
        #  * station x day-type x slot median   (point forecast, the MAE-optimal naive choice)
        #  * station x day-type x hour empirical quantiles (interval forecast, same 13 levels)
        obj._slot_median = (pool.groupby(["station_name", "is_weekend", "slot"])["passengers"].median()
                            .rename("baseline_median").reset_index())
        hq = pool.groupby(["station_name", "is_weekend", "hour"])["passengers"].quantile(QUANTILES).unstack()
        hq.columns = [f"b_{q_col(q)}" for q in QUANTILES]
        obj._hour_q = hq.reset_index()
        return obj

    # ------------------------------------------------------------------ naive baselines
    def baseline_columns(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Naive-baseline predictions for `rows`, indexed like `rows`: `baseline_mean` (station x
        day-type x slot mean), `baseline_median`, and `b_q****` empirical station x day-type x hour
        quantiles. These are what the TabPFN numbers are judged against."""
        keys = rows[["station_name", "is_weekend", "slot", "hour"]].copy()
        keys["_i"] = rows.index
        m = keys.merge(self._slot_median, on=["station_name", "is_weekend", "slot"], how="left") \
                .merge(self._hour_q, on=["station_name", "is_weekend", "hour"], how="left") \
                .set_index("_i")
        out = m[["baseline_median"] + [f"b_{c}" for c in Q_COLS]].reindex(rows.index)
        out.insert(0, "baseline_mean", rows["station_slot_mean"].to_numpy())
        out["baseline_median"] = out["baseline_median"].fillna(out["baseline_mean"])
        return out

    # ------------------------------------------------------------------ helpers
    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[BASELINE_FEATURES].copy()
        for col in CATEGORICAL:
            X[col] = pd.Categorical(X[col], categories=self._categories).codes
        return X.astype(float)

    # ------------------------------------------------------------------ fit
    def fit(self, seed: int = 0) -> "DemandBaseline":
        authenticate()
        from tabpfn_client import TabPFNRegressor

        pool = self.table[self.train_pool_mask]
        self.train_sample = pool.sample(min(TRAIN_SAMPLE_SIZE, len(pool)), random_state=seed)
        cat_idx = [BASELINE_FEATURES.index(c) for c in CATEGORICAL]
        self._model = TabPFNRegressor(model_path=MODEL_PATH, categorical_features_indices=cat_idx)
        self._model.fit(self._encode(self.train_sample), self.train_sample["passengers"])
        self.checkpoint_status = "fitted"
        return self

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ checkpoint
    CHECKPOINT_NAME = "demand_baseline"

    def fingerprint(self) -> str:
        """Everything that must be identical for a saved model to be reusable."""
        return make_fingerprint(
            features=BASELINE_FEATURES, quantiles=QUANTILES, model=MODEL_PATH, sample=TRAIN_SAMPLE_SIZE,
            cutoff=self.cutoff_date, n_rows=len(self.table), n_stations=self.table["station_name"].nunique(),
            dataset=self.dataset_folder)

    def save_checkpoint(self, metrics: dict | None = None):
        """Persist the fitted model's server reference + its exact training sample (see checkpoints.py)."""
        return save_checkpoint(self.CHECKPOINT_NAME, self._model, self.train_sample, {
            "task": "regression", "target": "passengers (per station per 15 min)",
            "description": "Counterfactual demand baseline (predictive distribution) for Category C",
            "n_features": len(BASELINE_FEATURES), "features": BASELINE_FEATURES, "tabpfn_model": MODEL_PATH,
            "fingerprint": self.fingerprint(), "dataset_folder": self.dataset_folder,
            "train_days_before": str(self.cutoff_date), "metrics": metrics,
        })

    def restore_or_fit(self, force: bool = False) -> str:
        """Use the saved checkpoint if it matches this dataset, else fit and save a new one.
        Returns 'loaded' | 'refit-from-checkpoint' | 'fitted'."""
        if not force:
            got = restore_checkpoint(self.CHECKPOINT_NAME, _regressor_cls(), self.fingerprint(),
                                     self._encode, "passengers")
            if got:
                self._model, self.train_sample, _meta, status = got
                self.checkpoint_status = status
                return status
        self.fit()
        self.save_checkpoint()
        self.checkpoint_status = "fitted"
        return "fitted"

    # ------------------------------------------------------------------ predict
    def predict_quantiles(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: subset of `self.table` (needs BASELINE_FEATURES). Returns a
        frame indexed like `rows`: one column per level in QUANTILES plus the
        predictive `mean` (demand is right-skewed, so mean > median; volumes
        to redistribute should use the mean, exceedance uses the quantiles)."""
        if not self.is_fitted:
            raise RuntimeError("call fit() first")
        X = self._encode(rows)
        parts, means = [], []
        for i in range(0, len(X), PREDICT_CHUNK):
            chunk = X.iloc[i:i + PREDICT_CHUNK]
            parts.append(np.column_stack(
                self._model.predict(chunk, output_type="quantiles", quantiles=QUANTILES)))
            means.append(np.asarray(self._model.predict(chunk, output_type="mean"), float))
        arr = np.maximum(np.vstack(parts), 0.0)           # counts can't be negative
        arr = np.maximum.accumulate(arr, axis=1)            # enforce monotone quantiles
        out = pd.DataFrame(arr, index=rows.index, columns=Q_COLS)
        out[MEAN_COL] = np.maximum(np.concatenate(means), 0.0)
        return out

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, n_test: int = 4_000, seed: int = 1) -> dict:
        """Held-out (last 20% of days, non-closure) accuracy + interval calibration,
        against naive baselines (slot mean/median, global mean, empirical station x hour quantiles)."""
        test = self.table[self.test_pool_mask]
        test = test.sample(min(n_test, len(test)), random_state=seed)
        pred = self.predict_quantiles(test)
        y = test["passengers"].to_numpy(float)
        med = pred[Q_COLS[MEDIAN_IDX]].to_numpy()
        mean = pred[MEAN_COL].to_numpy()
        glob = float(self.table.loc[self.train_pool_mask, "passengers"].mean())

        base = self.baseline_columns(test)
        self.heldout_frame = test[["timestamp", "station_name", "hour", "is_weekend", "passengers",
                                   "station_hour_p95"]].join(pred).join(base)
        ss_tot = np.sum((y - y.mean()) ** 2)
        b_med, b_mean = base["baseline_median"].to_numpy(), base["baseline_mean"].to_numpy()

        def cov(lo: float, hi: float, prefix: str = "") -> float:
            return float(np.mean((y >= pred_or_base(prefix, lo)) & (y <= pred_or_base(prefix, hi))))

        def pred_or_base(prefix: str, q: float) -> np.ndarray:
            return (base[f"b_{q_col(q)}"] if prefix else pred[q_col(q)]).to_numpy()

        def mean_pinball(prefix: str) -> float:
            return float(np.mean([pinball_loss(y, pred_or_base(prefix, q), q) for q in QUANTILES]))

        pin_m, pin_b = mean_pinball(""), mean_pinball("b")
        return {
            "n_train_sample": int(len(self.train_sample)),
            "n_test_rows": int(len(test)),
            "train_days_before": str(self.cutoff_date),
            "tabpfn_median": {"mae": round(float(np.mean(np.abs(y - med))), 2)},
            "tabpfn_mean": {
                "rmse": round(float(np.sqrt(np.mean((y - mean) ** 2))), 2),
                "r2": round(float(1 - np.sum((y - mean) ** 2) / ss_tot), 4),
            },
            "baseline_station_slot_mean": {
                "mae": round(float(np.mean(np.abs(y - b_mean))), 2),
                "rmse": round(float(np.sqrt(np.mean((y - b_mean) ** 2))), 2),
                "r2": round(float(1 - np.sum((y - b_mean) ** 2) / ss_tot), 4),
            },
            "baseline_station_slot_median": {"mae": round(float(np.mean(np.abs(y - b_med))), 2)},
            "baseline_global_mean": {"mae": round(float(np.mean(np.abs(y - glob))), 2)},
            "interval_coverage": {          # nominal -> observed share of held-out rows inside
                "80% (q10-q90)": round(cov(0.1, 0.9), 3),
                "90% (q05-q95)": round(cov(0.05, 0.95), 3),
                "95% (q025-q975)": round(cov(0.025, 0.975), 3),
            },
            "baseline_interval_coverage": {
                "80% (q10-q90)": round(cov(0.1, 0.9, "b"), 3),
                "90% (q05-q95)": round(cov(0.05, 0.95, "b"), 3),
                "95% (q025-q975)": round(cov(0.025, 0.975, "b"), 3),
            },
            "mean_interval_width_80": round(float(np.mean(pred[q_col(0.9)].to_numpy() - pred[q_col(0.1)].to_numpy())), 2),
            "baseline_mean_interval_width_80": round(float(np.mean(
                base[f"b_{q_col(0.9)}"].to_numpy() - base[f"b_{q_col(0.1)}"].to_numpy())), 2),
            "mean_pinball_loss": {"tabpfn": round(pin_m, 3), "baseline_empirical_quantiles": round(pin_b, 3),
                                  "skill_vs_baseline": round(1 - pin_m / pin_b, 4)},
        }

    # ------------------------------------------------------------------ showcase
    def showcase(self, n_days: int = 3, n_stations: int = 6) -> pd.DataFrame:
        """Contiguous held-out series for a handful of stations (busiest, median,
        quiet) over the last `n_days` closure-free held-out days — for plotting the
        prediction band against the real series."""
        t = self.table
        test = t[self.test_pool_mask]
        by_day = test.groupby(test["timestamp"].dt.date).size()
        days = [d for d, n in by_day.items() if n >= by_day.max() * 0.9][-n_days:]
        avg = t.groupby("station_name")["station_avg_passengers"].first().sort_values(ascending=False)
        picks = [avg.index[i] for i in np.linspace(0, len(avg) - 1, n_stations).astype(int)]
        rows = test[test["station_name"].isin(picks) & test["timestamp"].dt.date.isin(days)]
        pred = self.predict_quantiles(rows)
        return rows[["timestamp", "station_name", "hour", "passengers", "station_hour_p95"]] \
            .join(pred).join(self.baseline_columns(rows)) \
            .sort_values(["station_name", "timestamp"]).reset_index(drop=True)
