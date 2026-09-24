"""
The "normal flow" model and the decomposition into weather part and rest.

Model, fitted separately for every station (ridge regression on log scale):

    log1p(flow) ~ time-of-day slot x day type          ("level" dummies: commuter waves)
                + school holiday (+ extra on weekdays)
                + rain now + rain in the previous hour
                + temperature + temperature^2 + wind     (standardized)
                + linear trend over the training period

Only "clean" cells are used for fitting: every station within MAX_HOP hops of an event or
closure is excluded from 3 h before to 2 h after the episode. So events and closures do not
leak into what we call normal.

Decomposition (all on log scale, per station and slot):

    normal   = model prediction with TYPICAL weather for that time of day
    expected = model prediction with the ACTUAL weather
    total    = log1p(actual) - normal        -> normalized_flows.csv
    weather  = expected      - normal        -> normalized_weather.csv
    rest     = log1p(actual) - expected      -> normalized_rest.csv   (events, closures, anomalies)

    total = weather + rest  holds exactly.

Finally the normal is re-centred per station x slot x day type on the median of the clean rest:
the regression fits the mean, which zeros in the flow pull down, so without it a typical slot
would not be 0. Reading the values: 0 = normal, 0.1 ~ +10 %, -0.2 ~ -18 %, 0.69 ~ twice as many.
"""
import numpy as np
import pandas as pd

from . import config
from .episodes import hops_from

# Names of the non-level columns of the design matrix, in order.
EXTRA_TERMS = ["holiday", "holiday_weekday", "rain_now", "rain_prev_hour",
               "temp_z", "temp_z2", "wind_z", "trend"]
T = {name: i for i, name in enumerate(EXTRA_TERMS)}


# =========================================================================== time helpers
def slot_of(idx: pd.DatetimeIndex) -> np.ndarray:
    """Minute of day (0..1439)."""
    return np.asarray(idx.hour * 60 + idx.minute)


def day_type(idx: pd.DatetimeIndex) -> np.ndarray:
    """0 = Mon-Thu, 1 = Fri, 2 = Sat, 3 = Sun."""
    dow = np.asarray(idx.dayofweek)
    return np.select([dow <= 3, dow == 4, dow == 5], [0, 1, 2], 3)


def slot_daytype_index(idx) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([slot_of(idx), day_type(idx)], names=["slot", "day_type"])


# =========================================================================== design matrix
def fit_stats(idx: pd.DatetimeIndex, weather: pd.DataFrame, freq: str) -> dict:
    """Everything the design matrix needs from the TRAINING data (slots, time range,
    weather means/stds), so new data is encoded exactly like the training data."""
    w = weather.reindex(idx).interpolate(limit_direction="both")
    mean_std = lambda c: (float(w[c].mean()), float(w[c].std())) if c in w else (0.0, 1.0)
    return dict(slots=np.unique(slot_of(idx)), t0=idx[0], t1=idx[-1], freq=freq,
                temp=mean_std("temp"), wspd=mean_std("wspd"), has_wspd="wspd" in w)


def design_matrix(idx: pd.DatetimeIndex, weather: pd.DataFrame, stats: dict):
    """Returns (X, n_level). The first n_level columns are one-hot slot x day-type dummies,
    followed by the EXTRA_TERMS. Missing weather -> mean temperature/wind, no rain."""
    n = len(idx)
    slot = np.clip(np.searchsorted(stats["slots"], slot_of(idx)), 0, len(stats["slots"]) - 1)
    dtype = day_type(idx)
    n_level = len(stats["slots"]) * 4
    levels = np.zeros((n, n_level))
    levels[np.arange(n), slot * 4 + dtype] = 1.0

    holiday = np.zeros(n, bool)
    for a, b in config.SCHOOL_HOLIDAYS:
        holiday |= (idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b) + pd.Timedelta(days=1))

    w = weather.reindex(idx).interpolate(limit_direction="both")
    prcp = w["prcp"].fillna(0)
    # Rain in the hour BEFORE this slot (time-based, so it means the same at 15 min and 1 h).
    prev_hour = prcp.shift(1, freq=stats["freq"]).rolling("1h").mean().reindex(idx).fillna(0)
    z = lambda c: ((w[c].fillna(stats[c][0]) - stats[c][0]) / (stats[c][1] + 1e-9)).to_numpy()
    temp_z = z("temp")
    wind_z = z("wspd") if stats["has_wspd"] else np.zeros(n)
    span = (stats["t1"] - stats["t0"]) or pd.Timedelta(1)
    trend = np.clip(-1 + 2 * np.asarray((idx - stats["t0"]) / span), -1, 1)

    extra = np.column_stack([
        holiday, holiday & (dtype < 2),              # holiday, extra effect Mon-Fri
        prcp > config.RAIN_THRESHOLD_MM,             # rain now
        prev_hour > config.RAIN_THRESHOLD_MM,        # rain in the previous hour
        temp_z, temp_z ** 2, wind_z, trend,
    ]).astype(float)
    return np.hstack([levels, extra]), n_level


def set_typical_weather(X, n_level, typical: pd.DataFrame, idx, rows=slice(None)):
    """Overwrite the weather columns of X (in place) with 'typical weather for the time of day':
    no rain, and the average standardized temperature / wind of that slot in the training data."""
    typ = typical.reindex(slot_of(idx)).fillna(0)
    t, wd = typ["temp_z"].to_numpy()[rows], typ["wind_z"].to_numpy()[rows]
    X[rows, n_level + T["rain_now"]] = 0.0
    X[rows, n_level + T["rain_prev_hour"]] = 0.0
    X[rows, n_level + T["temp_z"]] = t
    X[rows, n_level + T["temp_z2"]] = t ** 2
    X[rows, n_level + T["wind_z"]] = wd


# =========================================================================== fitting
def exclusion_mask(idx, stations, adj, episodes) -> np.ndarray:
    """Boolean (time x station): True where an event/closure may influence the flow."""
    col = {s: j for j, s in enumerate(stations)}
    mask = np.zeros((len(idx), len(stations)), bool)
    before = pd.Timedelta(hours=config.EXCLUDE_HOURS_BEFORE)
    after = pd.Timedelta(hours=config.EXCLUDE_HOURS_AFTER)
    for ep in episodes:
        near = hops_from(adj, [a for a in ep["anchors"] if a in col], config.MAX_HOP)
        cols = [col[s] for s in near if s in col]
        i0, i1 = idx.searchsorted(ep["start"] - before), idx.searchsorted(ep["end"] + after)
        mask[i0:i1, cols] = True
    return mask


def ridge_per_station(Y, X, excluded, n_level, alpha=config.RIDGE_ALPHA):
    """Ridge regression for every column of Y, skipping NaN and excluded cells.
    Uses one shared X'X and subtracts the rows each station has to skip (much faster than
    building X per station). Level dummies are (almost) not penalized."""
    bad = np.isnan(Y) | excluded
    Y0 = np.where(np.isnan(Y), 0.0, Y)
    XtX, XtY = X.T @ X, X.T @ Y0
    penalty = np.diag(np.r_[np.full(n_level, 1e-6), np.full(X.shape[1] - n_level, alpha)])
    B = np.empty((X.shape[1], Y.shape[1]))
    for s in range(Y.shape[1]):
        Xb = X[bad[:, s]]
        B[:, s] = np.linalg.solve(XtX - Xb.T @ Xb + penalty, XtY[:, s] - Xb.T @ Y0[bad[:, s], s])
    return B


def fit_normal_model(flows: pd.DataFrame, weather: pd.DataFrame, adj: dict, episodes, freq: str) -> dict:
    """Fit the normal-flow model. Returns a picklable dict (see keys at the end)."""
    idx, stations = flows.index, list(flows.columns)
    Y = np.log1p(flows.to_numpy(float))

    excluded = exclusion_mask(idx, stations, adj, episodes)
    print(f"[info] fit without {excluded.mean():.1%} of the cells (event / closure windows)")

    stats = fit_stats(idx, weather, freq)
    X, n_level = design_matrix(idx, weather, stats)
    B = ridge_per_station(Y, X, excluded, n_level)

    # Typical weather per time of day (belongs to "normal", not to the weather effect).
    typical = pd.DataFrame({"temp_z": X[:, n_level + T["temp_z"]], "wind_z": X[:, n_level + T["wind_z"]]},
                           index=slot_of(idx)).groupby(level=0).mean()

    # Median re-centring on clean cells, per station x slot x day type.
    rest = np.where(excluded, np.nan, Y - X @ B)
    shift = pd.DataFrame(rest, index=slot_daytype_index(idx), columns=stations).groupby(level=[0, 1]).median()
    print(f"[info] median re-centring: mean shift {np.nanmean(shift.to_numpy()):+.3f} (log)")

    return dict(B=B, n_level=n_level, stats=stats, typical=typical, shift=shift,
                stations=stations, freq=freq, extra_terms=EXTRA_TERMS)


# =========================================================================== decomposition
def decompose(model: dict, flows: pd.DataFrame, weather: pd.DataFrame) -> dict:
    """Apply a fitted model to (possibly new) flows. Returns DataFrames
    'total', 'weather', 'rest' (log scale) and 'normal_passengers' (expected passengers
    with typical weather, i.e. the absolute reference for 'x % more than normal')."""
    stations = model["stations"]
    cols = [c for c in flows.columns if c in stations]
    missing = [c for c in flows.columns if c not in stations]
    if missing:
        print(f"[warn] {len(missing)} station(s) not in the model, skipped: {missing[:5]}")
    B = model["B"][:, [stations.index(c) for c in cols]]
    n_level, idx = model["n_level"], flows.index

    X, _ = design_matrix(idx, weather, model["stats"])
    no_weather = np.asarray((idx < weather.index.min()) | (idx > weather.index.max()))
    if no_weather.any():   # outside the weather data: assume typical weather -> weather part 0
        set_typical_weather(X, n_level, model["typical"], idx, no_weather)
        print(f"[warn] {int(no_weather.sum())} timestamps without weather data - typical weather assumed")

    X_normal = X.copy()
    set_typical_weather(X_normal, n_level, model["typical"], idx)
    shift = model["shift"].reindex(slot_daytype_index(idx))[cols].fillna(0).to_numpy()

    expected = X @ B + shift          # normal + actual weather
    normal = X_normal @ B + shift     # normal with typical weather
    actual = np.log1p(flows[cols].to_numpy(float))

    frame = lambda a, r=4: pd.DataFrame(np.round(a, r), index=idx, columns=cols)
    return {
        "total": frame(actual - normal),
        "weather": frame(expected - normal),
        "rest": frame(actual - expected),
        # NaN where the actual flow is NaN (night pause), so all tables share the same gaps.
        "normal_passengers": frame(np.where(np.isnan(actual), np.nan, np.expm1(normal)), 1),
    }


def coefficient_table(model: dict) -> pd.DataFrame:
    """Readable per-station effects in % (for explanations like 'rain adds +8 % here')."""
    B, n, stats = model["B"], model["n_level"], model["stats"]
    b = lambda term: B[n + T[term]]
    pct = lambda x: (np.exp(x) - 1) * 100
    z5 = 5 / (stats["temp"][1] + 1e-9)        # +5 degC expressed in standard deviations
    return pd.DataFrame({
        "station": model["stations"],
        "rain_now_pct": pct(b("rain_now")),
        "rain_prev_hour_pct": pct(b("rain_prev_hour")),
        "rain_both_pct": pct(b("rain_now") + b("rain_prev_hour")),
        "temp_plus5C_vs_mean_pct": pct(b("temp_z") * z5 + b("temp_z2") * z5 ** 2),
        "school_holiday_weekend_pct": pct(b("holiday")),
        "school_holiday_weekday_pct": pct(b("holiday") + b("holiday_weekday")),
    }).round(2)
