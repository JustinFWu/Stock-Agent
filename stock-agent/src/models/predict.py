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


def predict(ticker: str, features_row: pd.Series) -> Prediction:
    """
    Given the latest feature row for a ticker, return a Prediction.
    """
    path = MODELS_DIR / f"{ticker.upper()}_xgb.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model for {ticker}. Train first.")

    artifact = joblib.load(path)
    model = artifact["model"]
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
