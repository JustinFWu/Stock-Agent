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
from config import MODELS_DIR, FEATURES_DIR

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
]

LABEL_COL = "up_next_week"


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return only feature cols that exist in this dataframe."""
    return [c for c in FEATURE_COLS if c in df.columns]


def walk_forward_validate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Walk-forward validation: train on past, test on future.
    Never shuffles — respects time order.
    Returns aggregated metrics across all folds.
    """
    df = df.dropna(subset=[LABEL_COL])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    fold_size = len(df) // (n_splits + 1)
    results = []

    for i in range(1, n_splits + 1):
        train = df.iloc[: i * fold_size]
        test = df.iloc[i * fold_size : (i + 1) * fold_size]

        if len(test) == 0:
            continue

        X_train = train[feature_cols]
        y_train = train[LABEL_COL]
        X_test = test[feature_cols]
        y_test = test[LABEL_COL]

        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              use_label_encoder=False, eval_metric="logloss",
                              random_state=42, verbosity=0)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
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


def train_final_model(df: pd.DataFrame, ticker: str) -> Path:
    """Train on all available data and save model."""
    df = df.dropna(subset=[LABEL_COL])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df[LABEL_COL]

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          use_label_encoder=False, eval_metric="logloss",
                          random_state=42, verbosity=0)
    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{ticker.upper()}_xgb.joblib"
    joblib.dump({"model": model, "features": feature_cols}, path)
    print(f"Model saved -> {path}")
    return path
