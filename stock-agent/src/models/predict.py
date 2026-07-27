"""
Loads a trained model and produces a prediction for a given ticker.
Applies the no-trade zone: only signals when confidence >= threshold.
"""

import joblib
import pandas as pd
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, CONFIDENCE_THRESHOLD


@dataclass
class Prediction:
    ticker: str
    signal: str          # "UP", "DOWN", or "NO_TRADE"
    confidence: float    # model probability (0-1)
    direction: int       # 1 = up, 0 = down


POOLED_MODEL_PATH = MODELS_DIR / "pooled_xgb.joblib"


def predict(ticker: str, features_row: pd.Series) -> Prediction:
    """
    Given the latest feature row for a ticker, return a Prediction. All tickers are scored
    by the single pooled model (pick-a-ticker inference); `ticker` only labels the result.
    """
    if not POOLED_MODEL_PATH.exists():
        raise FileNotFoundError("No pooled model. Train first: python pipeline.py --train")

    artifact = joblib.load(POOLED_MODEL_PATH)
    model = artifact["model"]

    if not hasattr(model, "predict_proba"):
        # The pooled model is now a cross-sectional RANKING regressor (predicts demeaned
        # 5-day return). A standalone UP/DOWN/NO_TRADE call for one ticker is meaningless
        # for a ranker — its output only has meaning ranked against the rest of the
        # universe on the same date. The universe-ranking prediction path is not built yet
        # (gated on the ranking model clearing its validation gate); use
        # `python pipeline.py --rank-validate` to evaluate the model instead.
        raise NotImplementedError(
            "Single-ticker predict() is retired: the pooled model is a cross-sectional "
            "ranker. Rank a full universe by predicted demeaned return per date instead "
            "(see cross_sectional_validate); the per-ticker prediction path is not wired yet."
        )

    feature_cols = artifact["features"]
    X = features_row[feature_cols].values.reshape(1, -1)
    proba = model.predict_proba(X)[0][1]  # probability of UP

    if proba >= CONFIDENCE_THRESHOLD:
        signal = "UP"
        direction = 1
    elif proba <= (1 - CONFIDENCE_THRESHOLD):
        signal = "DOWN"
        direction = 0
    else:
        signal = "NO_TRADE"
        direction = -1

    return Prediction(ticker=ticker, signal=signal, confidence=proba, direction=direction)
