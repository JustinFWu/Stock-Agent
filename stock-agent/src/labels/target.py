"""
Generates the prediction target label.
Label: will the stock close higher 5 trading days from now?
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import FORWARD_DAYS


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'up_next_week' column: 1 if close[t + FORWARD_DAYS] > close[t], else 0.
    The last FORWARD_DAYS rows will have NaN labels (future unknown).
    """
    df = df.copy()
    future_close = df["Close"].shift(-FORWARD_DAYS)
    df["future_return"] = (future_close - df["Close"]) / df["Close"]
    df["up_next_week"] = (future_close > df["Close"]).astype(float)
    df.loc[df["up_next_week"].isna(), "up_next_week"] = float("nan")
    return df


def get_labeled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN labels or NaN features. Ready for training."""
    df = add_label(df)
    df = df.dropna(subset=["up_next_week"])
    return df
