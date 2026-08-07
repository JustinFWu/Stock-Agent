"""
Price/volume features from OHLCV bars.

Scope note: this used to carry MACD, moving-average crossovers and RSI. Those were
built to predict the direction of returns, measured at noise-level information
coefficient, and are gone. What remains is what actually feeds the current system —
past returns (which carry the leverage effect into a volatility forecast, and the
formation window for momentum), range, and volume.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ATR_PERIOD, RETURN_HORIZONS


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple returns over several horizons."""
    close = df["Close"]
    for n in RETURN_HORIZONS:
        df[f"return_{n}d"] = close.pct_change(n)
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range, and its price-normalized form."""
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
    """Volume relative to its own recent average. Volume clusters with volatility."""
    volume = df["Volume"]
    df["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    df["volume_ratio_21d"] = volume / volume.rolling(21).mean()
    return df


def add_gap_range(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight gap and intraday range."""
    prev_close = df["Close"].shift(1)
    day_range = df["High"] - df["Low"]

    df["gap"] = (df["Open"] - prev_close) / prev_close
    df["daily_range"] = day_range / df["Close"]
    # Where the close sits in the day's range (0 = low, 1 = high). A zero-range bar
    # (halted or untraded) has no meaningful position, so leave it NaN rather than
    # letting the division produce an infinity.
    df["close_position"] = (df["Close"] - df["Low"]) / day_range.where(day_range > 0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all price/volume feature builders on a raw OHLCV frame."""
    df = df.copy()
    df = add_returns(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_gap_range(df)
    return df
