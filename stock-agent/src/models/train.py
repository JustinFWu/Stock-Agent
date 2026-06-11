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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, FEATURES_DIR, CONFIDENCE_THRESHOLD, FORWARD_DAYS

# Feature columns to use (excludes OHLCV raw prices and label)
FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "sma10_vs_sma20", "sma20_vs_sma50",
    "sma_10_slope", "sma_20_slope", "sma_50_slope",
    "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50",
    "rsi", "rsi_overbought", "rsi_oversold",
    "macd", "macd_signal", "macd_hist", "macd_hist_slope",
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


def _walk_forward_folds(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5):
    """
    Yield (fold_index, train, test, proba) for each expanding-window split: train on all
    rows up to the fold, test on the next block, never shuffling (time order preserved).
    `proba` is the model's predicted probability of an up-move for each test row. Callers
    pass an already-cleaned df (NaN labels/features dropped) and decide what to do with each
    fold — classification metrics (validate) or a trade simulation (backtest).
    """
    fold_size = len(df) // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train = df.iloc[: i * fold_size]
        test = df.iloc[i * fold_size : (i + 1) * fold_size]
        if len(test) == 0:
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


def train_final_model(df: pd.DataFrame, ticker: str) -> Path:
    """Train on all available data and save model."""
    df = df.dropna(subset=[LABEL_COL])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df[LABEL_COL]

    model = _build_model()
    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{ticker.upper()}_xgb.joblib"
    joblib.dump({"model": model, "features": feature_cols}, path)
    print(f"Model saved -> {path}")
    return path
