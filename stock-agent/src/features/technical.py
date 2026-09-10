import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ATR_PERIOD, RETURN_HORIZONS

# MACD, MA crossovers and RSI lived here to predict return direction, measured at
# noise-level IC, and were removed. What remains feeds volatility and momentum only.


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    for n in RETURN_HORIZONS:
        df[f"return_{n}d"] = close.pct_change(n)
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / close
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    # Volume clusters with volatility, which is what makes the ratio a vol feature.
    volume = df["Volume"]
    df["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    df["volume_ratio_21d"] = volume / volume.rolling(21).mean()
    return df


def add_gap_range(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    day_range = df["High"] - df["Low"]

    df["gap"] = (df["Open"] - prev_close) / prev_close
    df["daily_range"] = day_range / df["Close"]
    # A zero-range bar (halted or untraded) has no meaningful close position, so leave
    # it NaN rather than letting the division produce an infinity.
    df["close_position"] = (df["Close"] - df["Low"]) / day_range.where(day_range > 0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_returns(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_gap_range(df)
    return df
