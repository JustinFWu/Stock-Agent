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

# A pre-committed gate: xgb must forecast next week's realized vol better than cheap,
# well-established baselines, or the baseline ships and the model is deleted. The failure
# mode this guards against is the previous branch's — keeping a model because it is a model.

# Both QLIKE and RMSE, because they disagree informatively. RMSE is symmetric and dominated
# by high-vol days; QLIKE is the Gaussian-likelihood loss, punishes under-forecasting far
# harder, and is robust to proxy noise. Winning one and losing the other is not a pass.

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

# One list, because a *partial* frame is the dangerous case: narrowing the feature set to
# whatever happened to be present fits a different model against the same gate, with no
# error and no record of what changed.
REQUIRED_COLUMNS = list(dict.fromkeys(
    VOL_FEATURES + HAR_FEATURES + [LABEL_COL, LABEL_END_COL]))


def _build_model() -> XGBRegressor:
    # One configuration shared by validation and the saved model, so the gate scores the
    # thing that ships.
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42, verbosity=0)


def _date_windows(dates, n_splits: int = 5):
    # Expanding windows split on dates, not row position — that is what makes it valid for
    # a pooled multi-ticker frame, where "train on the past, test on the next block" must
    # hold for every ticker at once even though many rows share each date.

    # No embargo here, deliberately: the caller purges against each row's own `label_end`.
    # A calendar-date count is the wrong unit for a ragged panel — reproduced on a 20-day
    # calendar, a ticker seen on days 0-4 and 10-19 kept a training row labelled to day 14.
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}")

    dates = np.sort(pd.Index(dates).unique().values)
    fold_size = len(dates) // (n_splits + 1)
    if fold_size == 0:
        return
    for i in range(1, n_splits + 1):
        yield i, dates[:i * fold_size], dates[i * fold_size:(i + 1) * fold_size]


def split_frames(clean: pd.DataFrame, n_splits: int = 5):
    # Separate from `validate_vol_forecast` because this is where walk-forward validity
    # lives, and a property only observable through a model score is one nobody checks.
    # Invariant: every training row's `label_end` falls strictly before the test block.
    for fold, train_dates, test_dates in _date_windows(clean.index, n_splits):
        test = clean[clean.index.isin(test_dates)]
        if test.empty:
            continue

        test_start = test.index.min()
        train = clean[clean.index.isin(train_dates) & (clean[LABEL_END_COL] < test_start)]
        if train.empty:
            continue

        yield fold, train, test


# Each forecaster takes (train, test, feature_cols) and returns one volatility forecast per
# test row, in the same annualized units as the label.

def _forecast_rw(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    # Zero parameters — the "do nothing" bar. Anything that cannot beat it is not a forecast.
    return test["rv_21d"].to_numpy()


def _forecast_ewma(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    # Computed as a feature so the baseline and the feature cannot drift apart.
    return test["ewma_vol"].to_numpy()


def _smearing_factor(log_actual: np.ndarray, log_predicted: np.ndarray) -> float:
    # Duan's smearing estimator. Fitting log(vol) under squared error and exponentiating
    # forecasts the conditional MEDIAN, not the mean, and vol is right-skewed: the factor
    # is ~1.10 here, so an uncorrected forecast reads ~9% low and QLIKE punishes it hard.

    # Estimated on TRAINING residuals only, so no test-block information leaks in, and
    # applied identically to HAR and XGBoost — both fit in logs — keeping them comparable.
    return float(np.mean(np.exp(log_actual - log_predicted)))


def _forecast_har(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    # Corsi's daily/weekly/monthly cascade, notoriously hard to beat. Logs keep forecasts
    # positive without clipping and make the residuals roughly homoskedastic, which OLS assumes.
    log_train_x = np.log(train[HAR_FEATURES])
    log_train_y = np.log(train[LABEL_COL].to_numpy())

    model = LinearRegression()
    model.fit(log_train_x, log_train_y)
    smearing = _smearing_factor(log_train_y, model.predict(log_train_x))
    return np.exp(model.predict(np.log(test[HAR_FEATURES]))) * smearing


def _forecast_xgb(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    # Same log target as HAR, so the two are directly comparable.
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
    # Zero for a perfect forecast, growing sharply when the forecast is too low.
    # Asymmetric on purpose: under-forecasting risk is the expensive error for a sizer.
    a = np.maximum(actual_vol.astype(float) ** 2, _EPS)
    f = np.maximum(forecast_vol.astype(float) ** 2, _EPS)
    ratio = a / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _rmse(actual_vol: np.ndarray, forecast_vol: np.ndarray) -> float:
    # Annualized volatility points. Monotone in MSE, so the ranking is identical.
    return float(np.sqrt(np.mean((actual_vol.astype(float) - forecast_vol.astype(float)) ** 2)))


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Three checks, because `dropna` alone passes all three values that cause trouble here.
    # Schema: narrowing the feature set to whatever the frame carried fits a different model
    # against the same gate, with nothing in the output saying the inputs changed.

    # Finiteness: an infinity survives `dropna` and reaches the estimator — `ewma_vol` can
    # produce one off a zero-variance stretch. Positivity: a zero realised vol is legitimate
    # for a halted week and becomes -inf under the log, poisoning a coefficient rather than raising.

    # Removed rows are counted and printed: a step that quietly discards most of the data
    # looks exactly like one that worked.
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
    # Read the fold-level numbers, not just the pooled ones: a model that wins overall while
    # losing half the folds has found a regime, not a forecast.

    # Consecutive rows share overlapping forward windows, so errors are autocorrelated and
    # the effective sample is smaller than the row count. That affects confidence in a narrow
    # margin; it does not bias the comparison, since every model scores the same rows.
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
    print("\n=== Phase 1 gate ===")
    print("  XGBoost must beat BOTH EWMA and HAR-RV on BOTH QLIKE and RMSE.")

    xgb = summary["xgb"]
    checks = []
    for baseline in GATE_BASELINES:
        for metric in ("qlike", "rmse"):
            baseline_score = summary[baseline][metric]
            better = xgb[metric] < baseline_score
            checks.append(better)
            # A zero baseline loss is unlikely but legitimate, and there is no percentage
            # improvement over zero. Dividing anyway crashes the gate on a run that should
            # simply have failed it, so the absolute gap is printed instead.
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
    # `kind` is an argument rather than a constant because if the gate fails, the baseline
    # is what gets saved — and the sizing layer neither knows nor cares which it loaded.
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

    # Saved with the model so production carries the correction validation measured it with.
    # Without it every forecast reads ~9% low, and an inverse-vol sizer reading a calmer
    # market than the one it is in takes ~10% more exposure per name than intended.
    payload["smearing"] = _smearing_factor(log_y, model.predict(x))
    payload["log_target"] = True
    payload["horizon"] = FORWARD_DAYS
    payload["trained_rows"] = len(clean)
    payload["trained_through"] = str(clean.index.max().date())

    # This is the only saved model: an interrupted dump over the live path would destroy
    # the one that was working.
    with atomic_path(VOL_MODEL_PATH) as tmp:
        joblib.dump(payload, tmp)
    print(f"Saved {kind} vol forecaster -> {VOL_MODEL_PATH}  "
          f"({len(clean):,} rows, {len(payload['features'])} features, "
          f"smearing {payload['smearing']:.4f})")
    return VOL_MODEL_PATH


def predict_vol(payload: dict, df: pd.DataFrame) -> np.ndarray:
    # The sizing layer must come through here rather than calling the estimator directly:
    # the log transform and the smearing correction are part of the forecast, and applying
    # one without the other silently biases every position size.

    # Checked against the feature list stored *in the payload*, not the module constant. A
    # model has to be scored on the columns it was fitted on — reading the constant would
    # reorder or extend the matrix the day a feature is added, and the estimator would accept it.
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
