"""
Volatility forecasting: the Phase 1 gate.

The question is narrow and pre-committed: does a gradient-boosted model forecast next
week's realized volatility better than cheap, well-established baselines? If it does
not, the baseline ships and the model is deleted. The failure mode this guards against
is the one the previous branch fell into — keeping a model because it is a model.

Four forecasters, evaluated on identical rows:

  rw     random walk. Today's 21-day realized vol, carried forward. Zero parameters.
         The "do nothing" bar; anything that cannot beat this is not a forecast.
  ewma   RiskMetrics exponentially weighted variance, lambda fixed at 0.94. Zero
         fitted parameters. The industry default.
  har    Corsi's HAR-RV: log forward vol regressed on log realized vol at daily,
         weekly and monthly horizons. Three coefficients, refit each fold. Captures
         volatility's long-memory cascade and is notoriously hard to beat.
  xgb    Gradient boosting on the full feature set, same log target as HAR.

Scoring uses both QLIKE and RMSE because they disagree in an informative way. RMSE is
symmetric and dominated by high-vol days; QLIKE is the loss implied by a Gaussian
likelihood, penalises under-forecasting far more than over-forecasting, and is robust
to noise in the volatility proxy. A model that wins on one and loses on the other has
not earned the gate.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, FORWARD_DAYS
from src.storage import atomic_path

LABEL_COL = "forward_vol"
# The date each row's label window closes on. Carried from `labels.target` so the
# walk-forward purge can ask the row itself rather than counting calendar dates.
LABEL_END_COL = "label_end"

# Long-horizon returns (126d, 252d) are deliberately absent: they belong to the momentum
# formation window, carry no volatility information, and their year-long warmup would
# throw away the first year of every ticker once features are dropped for NaNs.
VOL_FEATURES = [
    "rv_1d", "rv_5d", "rv_21d", "rv_63d",
    "park_5d", "park_21d", "park_63d",
    "gk_5d", "gk_21d", "gk_63d",
    "ewma_vol",
    "vol_ratio_5_21", "vol_ratio_21_63", "vol_of_vol",
    "atr_pct",
    "return_1d", "return_5d", "return_10d", "return_21d", "return_63d",
    "volume_ratio_5d", "volume_ratio_21d",
    "gap", "daily_range", "close_position",
]

# Corsi's daily / weekly / monthly cascade.
HAR_FEATURES = ["rv_1d", "rv_5d", "rv_21d"]

VOL_MODEL_PATH = MODELS_DIR / "vol_forecast.joblib"

# Everything a frame must carry to be trainable or scoreable. Stated as one list
# because a *partial* frame is the dangerous case: narrowing the feature set to
# whatever happened to be present fits a different model against the same gate,
# with no error and no record of what changed.
REQUIRED_COLUMNS = list(dict.fromkeys(
    VOL_FEATURES + HAR_FEATURES + [LABEL_COL, LABEL_END_COL]))


def _build_model() -> XGBRegressor:
    """The single XGBoost configuration shared by validation and the saved model."""
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42, verbosity=0)


def _date_windows(dates, n_splits: int = 5):
    """
    Yield (fold, train_dates, test_dates) for expanding-window splits over the sorted
    unique dates. Time order is never shuffled.

    Splitting on dates rather than row position is what makes this valid for a pooled
    multi-ticker frame: "train on the past, test on the next block" then holds for every
    ticker at once, even though many rows share each date.

    No embargo is applied here, deliberately. The blocks are handed over whole and the
    caller purges them against each row's own `label_end`, because a fixed number of
    calendar dates is the wrong unit for a ragged panel: a label looks ahead `horizon`
    *observed bars of one ticker*, and for a ticker with missing sessions that lands
    well past the `horizon`-th pooled date. Such a row clears a date-counted embargo
    while its label is measured from inside the test block — reproduced on a 20-day
    calendar, where a ticker seen on days 0-4 and 10-19 kept a training row whose
    label ended on day 14, four days into the test block.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}")

    dates = np.sort(pd.Index(dates).unique().values)
    fold_size = len(dates) // (n_splits + 1)
    if fold_size == 0:
        return
    for i in range(1, n_splits + 1):
        yield i, dates[:i * fold_size], dates[i * fold_size:(i + 1) * fold_size]


def split_frames(clean: pd.DataFrame, n_splits: int = 5):
    """
    Yield (fold, train, test) frames, with training rows purged of label overlap.

    Separated from `validate_vol_forecast` because this is where the walk-forward
    validity actually lives, and a property only observable through a model score is
    a property nobody checks. The invariant is one line and worth stating: every
    training row's `label_end` falls strictly before the first date of the test
    block, so no row is ever fitted on an outcome measured inside the block it is
    about to be graded on.

    Purging on the row's own label end rather than on a count of calendar dates is
    the part that matters for a ragged panel — see `_date_windows`.
    """
    for fold, train_dates, test_dates in _date_windows(clean.index, n_splits):
        test = clean[clean.index.isin(test_dates)]
        if test.empty:
            continue

        test_start = test.index.min()
        train = clean[clean.index.isin(train_dates) & (clean[LABEL_END_COL] < test_start)]
        if train.empty:
            continue

        yield fold, train, test


# --------------------------------------------------------------------------- #
# Forecasters. Each takes (train, test, feature_cols) and returns a volatility
# forecast per test row, in the same annualized units as the label.
# --------------------------------------------------------------------------- #

def _forecast_rw(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Random walk: next week's vol equals the trailing 21-day realized vol."""
    return test["rv_21d"].to_numpy()


def _forecast_ewma(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """EWMA variance carried forward. Computed as a feature so one definition serves both uses."""
    return test["ewma_vol"].to_numpy()


def _smearing_factor(log_actual: np.ndarray, log_predicted: np.ndarray) -> float:
    """
    Duan's smearing estimator, for undoing the bias introduced by fitting in logs.

    Fitting log(vol) under squared-error loss and exponentiating produces a forecast of
    the conditional MEDIAN, not the mean, and for a right-skewed quantity like
    volatility the median sits below the mean. Measured here the correction factor is
    about 1.10, so an uncorrected forecast reads roughly 9% *low* — which QLIKE, being
    deliberately asymmetric against under-forecasting, punishes hard. Left uncorrected
    it makes a log-target model look worse than a direct-variance one for reasons that
    have nothing to do with forecast skill.

    The correction is mean(exp(residual)), estimated on TRAINING residuals only so no
    test-block information leaks into the forecast. It is applied identically to HAR
    and XGBoost, both of which fit in logs, keeping their comparison fair.
    """
    return float(np.mean(np.exp(log_actual - log_predicted)))


def _forecast_har(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    HAR-RV in logs. Working in logs keeps forecasts positive without clipping and makes
    the residuals roughly homoskedastic, which ordinary least squares assumes.
    """
    log_train_x = np.log(train[HAR_FEATURES])
    log_train_y = np.log(train[LABEL_COL].to_numpy())

    model = LinearRegression()
    model.fit(log_train_x, log_train_y)
    smearing = _smearing_factor(log_train_y, model.predict(log_train_x))
    return np.exp(model.predict(np.log(test[HAR_FEATURES]))) * smearing


def _forecast_xgb(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Gradient boosting on the same log target as HAR, so the two are directly comparable."""
    log_train_y = np.log(train[LABEL_COL].to_numpy())

    model = _build_model()
    model.fit(train[feature_cols], log_train_y)
    smearing = _smearing_factor(log_train_y, model.predict(train[feature_cols]))
    return np.exp(model.predict(test[feature_cols])) * smearing


FORECASTERS = {
    "rw": _forecast_rw,
    "ewma": _forecast_ewma,
    "har": _forecast_har,
    "xgb": _forecast_xgb,
}

# The gate: xgb must beat both of these, on both metrics.
GATE_BASELINES = ("ewma", "har")

_EPS = 1e-12


def _qlike(actual_vol: np.ndarray, forecast_vol: np.ndarray) -> float:
    """
    QLIKE on variances: mean(a/f - log(a/f) - 1), which is zero for a perfect forecast
    and grows sharply when the forecast is too low. Asymmetric on purpose — under-
    forecasting risk is the expensive error for a position sizer.
    """
    a = np.maximum(actual_vol.astype(float) ** 2, _EPS)
    f = np.maximum(forecast_vol.astype(float) ** 2, _EPS)
    ratio = a / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _rmse(actual_vol: np.ndarray, forecast_vol: np.ndarray) -> float:
    """RMSE in annualized volatility points. Monotone in MSE, so the ranking is identical."""
    return float(np.sqrt(np.mean((actual_vol.astype(float) - forecast_vol.astype(float)) ** 2)))


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop rows that cannot be honestly trained or scored on, and return the features.

    Three checks, because `dropna` on its own passes all three of the values that
    actually cause trouble here:

      schema      every required column must be present. Silently narrowing the
                  feature set to whatever the frame happened to carry fits a
                  different model and compares it against the same gate, with no
                  error and nothing in the output saying the inputs changed.
      finiteness  NaN is not the only missing value. An infinity survives `dropna`
                  and reaches the estimator — `ewma_vol` can produce one off a
                  zero-variance stretch, and the mandatory EWMA baseline is then
                  scored against it.
      positivity  HAR and the log target take logs. A zero realised volatility is a
                  legitimate reading for a halted week and becomes -inf under the
                  log, poisoning a fitted coefficient instead of raising.

    Rows removed are counted and printed. A preparation step that quietly discards
    most of the data looks exactly like one that worked.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Frame is missing required columns {missing}. Rebuild the feature cache "
            "(`pipeline.py --vol-validate --rebuild`) instead of fitting on a subset — "
            "a narrowed feature set is a different model wearing the same name.")

    feature_cols = list(VOL_FEATURES)
    numeric = list(dict.fromkeys(feature_cols + HAR_FEATURES + [LABEL_COL]))

    before = len(df)
    finite = np.isfinite(df[numeric].to_numpy(dtype=float)).all(axis=1)
    clean = df[finite]
    non_finite = before - len(clean)

    usable = ((clean[LABEL_COL] > 0)
              & (clean[HAR_FEATURES] > 0).all(axis=1)
              & clean[LABEL_END_COL].notna())
    kept = clean[usable]
    non_positive = len(clean) - len(kept)

    if before and len(kept) < before:
        print(f"  prepare: kept {len(kept):,}/{before:,} rows "
              f"({non_finite:,} non-finite, {non_positive:,} non-positive or unlabelled)")
    return kept, feature_cols


def validate_vol_forecast(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Walk-forward comparison of every forecaster on identical out-of-sample rows.

    Reports QLIKE and RMSE per fold and pooled across all folds, then states plainly
    whether XGBoost cleared the gate against both EWMA and HAR-RV on both metrics.

    Read the fold-level numbers, not just the pooled ones. A model that wins overall
    while losing on half the folds has found a regime, not a forecast.

    One caveat on precision: consecutive rows share overlapping forward windows, so the
    errors are autocorrelated and the effective sample is smaller than the row count
    suggests. That affects how confident to be in a narrow margin; it does not bias the
    comparison between models, which all score the same rows.
    """
    clean, feature_cols = prepare(df)
    if clean.empty:
        raise ValueError("No rows left after dropping NaN features/labels.")

    print(f"\nEvaluating on {len(clean):,} rows, {clean['ticker'].nunique()} tickers, "
          f"{clean.index.nunique():,} dates ({clean.index.min().date()} to {clean.index.max().date()})")
    print(f"Features: {len(feature_cols)}   horizon: {FORWARD_DAYS}d   splits: {n_splits}")

    fold_rows = []
    pooled = {name: {"actual": [], "forecast": []} for name in FORECASTERS}

    for fold, train, test in split_frames(clean, n_splits):
        actual = test[LABEL_COL].to_numpy()
        row = {
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_from": str(test.index.min().date()),
            "test_to": str(test.index.max().date()),
        }
        for name, forecaster in FORECASTERS.items():
            forecast = forecaster(train, test, feature_cols)
            row[f"qlike_{name}"] = _qlike(actual, forecast)
            row[f"rmse_{name}"] = _rmse(actual, forecast)
            pooled[name]["actual"].append(actual)
            pooled[name]["forecast"].append(forecast)
        fold_rows.append(row)

    if not fold_rows:
        raise ValueError("No evaluable walk-forward folds — try fewer splits.")

    folds = pd.DataFrame(fold_rows)
    print("\n=== Per-fold QLIKE (lower is better) ===")
    print(folds[["fold", "test_from", "test_to", "test_rows"]
                + [f"qlike_{n}" for n in FORECASTERS]].to_string(index=False, float_format="%.4f"))
    print("\n=== Per-fold RMSE, annualized vol points (lower is better) ===")
    print(folds[["fold"] + [f"rmse_{n}" for n in FORECASTERS]]
          .to_string(index=False, float_format="%.4f"))

    summary = {}
    for name in FORECASTERS:
        actual = np.concatenate(pooled[name]["actual"])
        forecast = np.concatenate(pooled[name]["forecast"])
        summary[name] = {
            "qlike": _qlike(actual, forecast),
            "rmse": _rmse(actual, forecast),
            "fold_wins_qlike": 0,
            "fold_wins_rmse": 0,
        }

    for _, row in folds.iterrows():
        summary[min(FORECASTERS, key=lambda n: row[f"qlike_{n}"])]["fold_wins_qlike"] += 1
        summary[min(FORECASTERS, key=lambda n: row[f"rmse_{n}"])]["fold_wins_rmse"] += 1

    table = pd.DataFrame(summary).T[["qlike", "rmse", "fold_wins_qlike", "fold_wins_rmse"]]
    print("\n=== Pooled across all out-of-sample folds ===")
    print(table.to_string(float_format="%.4f"))

    verdict = _report_gate(summary, len(folds))
    return {"folds": folds.to_dict("records"), "summary": summary, **verdict}


def _report_gate(summary: dict, n_folds: int) -> dict:
    """Print the pre-committed pass/fail and return it."""
    print("\n=== Phase 1 gate ===")
    print("  XGBoost must beat BOTH EWMA and HAR-RV on BOTH QLIKE and RMSE.")

    xgb = summary["xgb"]
    checks = []
    for baseline in GATE_BASELINES:
        for metric in ("qlike", "rmse"):
            baseline_score = summary[baseline][metric]
            better = xgb[metric] < baseline_score
            checks.append(better)
            # A zero baseline loss is a perfect baseline forecast — unlikely, but a
            # legitimate result, and there is no percentage improvement over zero to
            # report. Dividing anyway crashes the gate on a run that should simply
            # have failed it, so the absolute gap is printed instead.
            margin = (f"{(baseline_score - xgb[metric]) / baseline_score:+.1%}"
                      if baseline_score > 0 else f"{baseline_score - xgb[metric]:+.4f} abs")
            print(f"  {'PASS' if better else 'FAIL'}  xgb {metric:<5} {xgb[metric]:.4f} "
                  f"vs {baseline:<4} {baseline_score:.4f}   ({margin})")

    passed = all(checks)
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} — "
          f"{'ship the model' if passed else 'ship the baseline and delete the model'}")
    if passed and xgb["fold_wins_qlike"] < n_folds / 2:
        print("  CAUTION: xgb wins pooled QLIKE but loses most individual folds — "
              "check for a single regime carrying the result.")

    best_qlike = min(summary, key=lambda n: summary[n]["qlike"])
    print(f"  Best QLIKE overall: {best_qlike}")
    return {"gate_passed": passed, "best_by_qlike": best_qlike}


def train_vol_model(df: pd.DataFrame, kind: str = "xgb") -> Path:
    """
    Fit the chosen forecaster on the full history and save it for downstream sizing.

    `kind` is a deliberate argument rather than a constant: if the gate fails, the
    baseline is what gets saved, and the sizing layer neither knows nor cares which
    one it loaded.
    """
    if kind not in ("xgb", "har"):
        raise ValueError(f"kind must be 'xgb' or 'har' (parameter-free baselines need no fitting), got {kind!r}")

    clean, feature_cols = prepare(df)

    log_y = np.log(clean[LABEL_COL].to_numpy())

    if kind == "xgb":
        model = _build_model()
        x = clean[feature_cols]
        model.fit(x, log_y)
        payload = {"kind": "xgb", "model": model, "features": feature_cols,
                   "log_input": False}
    else:
        model = LinearRegression()
        x = np.log(clean[HAR_FEATURES])
        model.fit(x, log_y)
        payload = {"kind": "har", "model": model, "features": HAR_FEATURES,
                   "log_input": True}

    # Saved alongside the model so production forecasts carry the same bias correction
    # validation measured them with. Without it every forecast reads about 9% low, and
    # an inverse-volatility sizer reading a calmer market than the one it is in takes
    # roughly 10% *more* exposure per name than intended, before the cap binds.
    payload["smearing"] = _smearing_factor(log_y, model.predict(x))
    payload["log_target"] = True
    payload["horizon"] = FORWARD_DAYS
    payload["trained_rows"] = len(clean)
    payload["trained_through"] = str(clean.index.max().date())

    # Written through a temporary file: this is the only saved model, and an
    # interrupted dump over the live path destroys the one that was working.
    with atomic_path(VOL_MODEL_PATH) as tmp:
        joblib.dump(payload, tmp)
    print(f"Saved {kind} vol forecaster -> {VOL_MODEL_PATH}  "
          f"({len(clean):,} rows, {len(payload['features'])} features, "
          f"smearing {payload['smearing']:.4f})")
    return VOL_MODEL_PATH


def predict_vol(payload: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Apply a saved forecaster to a feature frame, returning annualized volatility.

    The sizing layer should always come through here rather than calling the estimator
    directly — the log transform and the smearing correction are part of the forecast,
    and applying one without the other silently biases every position size.

    The inputs are checked the way `prepare` checks them at training time, against the
    feature list stored *in the payload* rather than the module constant. A model has
    to be scored on the columns it was fitted on: reading the constant instead would
    silently reorder or extend the matrix the day a feature is added, and the estimator
    would accept it and return numbers.
    """
    features = list(payload["features"])
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot score: frame is missing {missing}, which this model was fitted on.")

    x = df[features]
    values = x.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot score: the feature matrix holds NaN or infinity. Filter the "
                         "rows first — a position size is about to be computed from this.")
    if payload.get("log_input"):
        if (values <= 0).any():
            raise ValueError("Cannot score: this model takes logs of its inputs and the frame "
                             "holds non-positive values.")
        x = np.log(x)
    return np.exp(payload["model"].predict(x)) * payload.get("smearing", 1.0)
