"""
Training pipeline for the swing trade direction classifier.
Uses XGBoost with walk-forward validation.
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, CONFIDENCE_THRESHOLD, FORWARD_DAYS

# Feature columns to use (excludes OHLCV raw prices and label)
FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "sma10_vs_sma20", "sma20_vs_sma50",
    "sma_10_slope", "sma_20_slope", "sma_50_slope",
    "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50",
    "rsi", "rsi_overbought", "rsi_oversold",
    "macd_norm", "macd_signal_norm", "macd_hist_norm", "macd_hist_slope_norm",
    "atr_pct",
    "volume_ratio_20d", "volume_ratio_5d",
    "gap", "daily_range", "close_position",
    "rs_vs_sector_5d", "rs_vs_sector_20d",
    "news_sentiment", "news_bullish_pct", "news_bearish_pct", "news_has_news",
]

LABEL_COL = "up_next_week"


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return only feature cols that exist in this dataframe."""
    return [c for c in FEATURE_COLS if c in df.columns]


def _build_model() -> XGBClassifier:
    """The single XGBoost configuration shared by validation, backtest, and the saved model."""
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         use_label_encoder=False, eval_metric="logloss",
                         random_state=42, verbosity=0)


def _date_windows(dates, n_splits: int = 5, embargo: int = FORWARD_DAYS):
    """
    Yield (train_dates, test_dates) for each expanding-window split over the sorted
    UNIQUE dates — never shuffling, time order preserved. Splitting on dates (not row
    position) is what makes this valid for a pooled multi-ticker frame: 'train on the
    past, test on the next block' then holds across every ticker simultaneously, even
    though many rows share each date.

    Embargo: the label looks `embargo` (FORWARD_DAYS) trading days ahead, so the last
    `embargo` train dates would have labels computed from closes inside the test block.
    We drop those dates so train and test never overlap — without this gap, AUC is
    inflated by leakage. (Assumes a shared trading calendar, true for US equities here.)
    """
    dates = np.sort(pd.Index(dates).unique().values)
    fold_size = len(dates) // (n_splits + 1)
    if fold_size == 0:
        return
    for i in range(1, n_splits + 1):
        train_cutoff = i * fold_size - embargo
        if train_cutoff <= 0:
            continue
        train_dates = dates[:train_cutoff]
        test_dates = dates[i * fold_size : (i + 1) * fold_size]
        if len(test_dates) == 0:
            continue
        yield i, train_dates, test_dates


def _walk_forward_folds(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5,
                        embargo: int = FORWARD_DAYS):
    """
    Yield (fold_index, train, test, proba) for each date-based expanding-window split
    (see _date_windows). `proba` is the model's predicted probability of an up-move for
    each test row. Callers pass an already-cleaned df (NaN labels/features dropped) and
    decide what to do with each fold — classification metrics (validate) or a trade
    simulation (backtest). Works for a single ticker (one row per date) or a pooled
    frame (many tickers per date).
    """
    for i, train_dates, test_dates in _date_windows(df.index, n_splits, embargo):
        train = df[df.index.isin(train_dates)]
        test = df[df.index.isin(test_dates)]
        if len(train) == 0 or len(test) == 0:
            continue

        model = _build_model()
        model.fit(train[feature_cols], train[LABEL_COL])
        proba = model.predict_proba(test[feature_cols])[:, 1]
        yield i, train, test, proba


def walk_forward_validate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Walk-forward validation: train on past, test on future.
    Never shuffles — respects time order.
    Returns aggregated classification metrics across all folds.
    """
    df = df.dropna(subset=[LABEL_COL])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    results = []
    for i, train, test, proba in _walk_forward_folds(df, feature_cols, n_splits):
        y_test = test[LABEL_COL]
        if y_test.nunique() < 2:
            continue  # AUC undefined when a fold's test set is all one class
        pred = (proba >= 0.5).astype(int)
        results.append({
            "fold": i,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
            "n_train": len(train),
            "n_test": len(test),
        })

    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    print(f"\nMean AUC: {summary['auc'].mean():.3f}")
    print(f"Mean Accuracy: {summary['accuracy'].mean():.3f}")
    return summary.to_dict("records")


def walk_forward_backtest(
    df: pd.DataFrame,
    n_splits: int = 5,
    threshold: float = CONFIDENCE_THRESHOLD,
    forward_days: int = FORWARD_DAYS,
    cost_per_trade: float = 0.0,
) -> dict:
    """
    Economic backtest over the same walk-forward folds as walk_forward_validate.

    Where walk_forward_validate measures *classification* quality (AUC/accuracy),
    this measures *money*: it applies the production decision rule to each
    out-of-sample fold (long if proba >= threshold, short if proba <= 1 - threshold,
    else flat — the same no-trade zone used in predict.py) and books the realized
    forward_days return via the precomputed `future_return` label.

    Trades are non-overlapping: after entering, we skip forward_days rows so each
    ~forward_days holding period is counted once (consecutive daily signals would
    otherwise share most of their holding window and overstate activity).

    `cost_per_trade` is a round-trip cost as a return fraction (e.g. 0.001 = 10 bps)
    subtracted from each trade's return.
    """
    if "future_return" not in df.columns:
        raise ValueError(
            "df has no 'future_return' column; call add_label() before backtesting."
        )

    df = df.dropna(subset=[LABEL_COL, "future_return"])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    # Collect out-of-sample predictions across folds, in time order. Retrains each fold
    # exactly as walk_forward_validate does (shared _walk_forward_folds), so the economic
    # view and the classification view agree on the regime.
    oos_parts = [
        pd.DataFrame(
            {"proba": proba, "future_return": test["future_return"].values},
            index=test.index,
        )
        for _, _, test, proba in _walk_forward_folds(df, feature_cols, n_splits)
    ]

    if not oos_parts:
        print("No out-of-sample folds to backtest.")
        return {"n_trades": 0, "n_oos_rows": 0}

    oos = pd.concat(oos_parts)

    # No-trade zone in action across all out-of-sample rows.
    n_long = int((oos["proba"] >= threshold).sum())
    n_short = int((oos["proba"] <= (1 - threshold)).sum())
    n_flat = len(oos) - n_long - n_short

    # Non-overlapping trade simulation in time order.
    probas = oos["proba"].values
    future_returns = oos["future_return"].values
    equity = 1.0
    curve = []
    net_returns = []
    i = 0
    while i < len(oos):
        proba = probas[i]
        if proba >= threshold:
            side = 1
        elif proba <= (1 - threshold):
            side = -1
        else:
            i += 1
            continue
        net = side * future_returns[i] - cost_per_trade
        equity *= (1 + net)
        net_returns.append(net)
        curve.append(equity)
        i += forward_days  # non-overlapping hold

    n_trades = len(net_returns)
    print("\n=== Economic backtest (out-of-sample, walk-forward) ===")
    print(f"  decision rule:       long >= {threshold:.2f}, short <= {1 - threshold:.2f}, else flat")
    print(f"  cost per trade:      {cost_per_trade:.4%}")
    print(f"  OOS rows:            {len(oos)}  (long sig {n_long}, short sig {n_short}, no-trade {n_flat})")

    if n_trades == 0:
        print("  trades taken:        0  (every signal fell in the no-trade zone)")
        return {"n_trades": 0, "n_oos_rows": len(oos),
                "n_long_signals": n_long, "n_short_signals": n_short, "n_no_trade": n_flat}

    rets = np.array(net_returns)
    eq = np.array(curve)
    hit_rate = float((rets > 0).mean())
    cum_return = equity - 1.0
    avg = float(rets.mean())
    std = float(rets.std(ddof=1)) if n_trades > 1 else 0.0
    periods_per_year = 252 / forward_days  # approximate: assumes back-to-back holds
    sharpe = (avg / std) * np.sqrt(periods_per_year) if std > 0 else float("nan")
    max_dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())

    bh_return = float("nan")
    if "Close" in df.columns:
        span = df.loc[oos.index[0]:oos.index[-1], "Close"]
        if len(span) > 1 and span.iloc[0] != 0:
            bh_return = float(span.iloc[-1] / span.iloc[0] - 1.0)

    print(f"  trades taken:        {n_trades}  (non-overlapping {forward_days}-day holds)")
    print(f"  hit rate:            {hit_rate:.1%}")
    print(f"  avg return / trade:  {avg:.2%}")
    print(f"  cumulative return:   {cum_return:.1%}")
    print(f"  Sharpe (annualised): {sharpe:.2f}")
    print(f"  max drawdown:        {max_dd:.1%}")
    if bh_return == bh_return:  # not NaN
        print(f"  buy & hold (span):   {bh_return:.1%}")

    return {
        "n_trades": n_trades,
        "n_oos_rows": len(oos),
        "n_long_signals": n_long,
        "n_short_signals": n_short,
        "n_no_trade": n_flat,
        "hit_rate": hit_rate,
        "avg_return_per_trade": avg,
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "buy_and_hold_return": bh_return,
        "final_equity": equity,
        "cost_per_trade": cost_per_trade,
        "threshold": threshold,
    }


def walk_forward_holdout(train_df: pd.DataFrame, holdout_df: pd.DataFrame,
                         n_splits: int = 5, embargo: int = FORWARD_DAYS) -> dict:
    """
    The honesty check for a pooled model: does it generalize to tickers it never saw?

    For each date-based fold, fit on the TRAIN-universe rows up to the cutoff and score
    the HOLDOUT tickers' rows in the next block. So both axes stay clean: the model is
    tested on out-of-sample dates AND out-of-sample tickers, with the same embargo as
    walk_forward_validate. Pass holdout_df built from tickers excluded from train_df
    (see split_universe). Report only at the very end, after all tuning is done.
    """
    feature_cols = [c for c in get_available_features(train_df) if c in holdout_df.columns]
    train_df = train_df.dropna(subset=[LABEL_COL] + feature_cols)
    holdout_df = holdout_df.dropna(subset=[LABEL_COL] + feature_cols)

    all_dates = train_df.index.union(holdout_df.index)
    results = []
    for i, train_dates, test_dates in _date_windows(all_dates, n_splits, embargo):
        tr = train_df[train_df.index.isin(train_dates)]
        te = holdout_df[holdout_df.index.isin(test_dates)]
        if len(tr) == 0 or len(te) == 0 or te[LABEL_COL].nunique() < 2:
            continue
        model = _build_model()
        model.fit(tr[feature_cols], tr[LABEL_COL])
        proba = model.predict_proba(te[feature_cols])[:, 1]
        results.append({
            "fold": i,
            "auc": roc_auc_score(te[LABEL_COL], proba),
            "n_train": len(tr),
            "n_test": len(te),
        })

    if not results:
        print("No evaluable holdout folds.")
        return {"folds": []}

    summary = pd.DataFrame(results)
    print("\n=== Holdout check (never-trained tickers, out-of-sample dates) ===")
    print(summary.to_string(index=False))
    print(f"\nHoldout mean AUC: {summary['auc'].mean():.3f}")
    return summary.to_dict("records")


POOLED_MODEL_PATH = MODELS_DIR / "pooled_xgb.joblib"


def train_pooled_model(df: pd.DataFrame) -> Path:
    """
    Train one model on the full pooled dataset and save it to POOLED_MODEL_PATH.
    This single model backs pick-a-ticker prediction for every ticker (predict.py loads
    it regardless of which ticker is asked about) — there are no longer per-ticker models.
    """
    df = df.dropna(subset=[LABEL_COL])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    model = _build_model()
    model.fit(df[feature_cols], df[LABEL_COL])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, POOLED_MODEL_PATH)
    print(f"Pooled model saved -> {POOLED_MODEL_PATH}  ({len(df)} rows, {len(feature_cols)} features)")
    return POOLED_MODEL_PATH
