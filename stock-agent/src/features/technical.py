"""
Computes technical indicators and features from OHLCV data.
Uses the `ta` library for standard indicators.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RSI_PERIOD, ATR_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, MA_PERIODS, RETURN_HORIZONS


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add return features over multiple horizons."""
    close = df["Close"]
    for n in RETURN_HORIZONS:
        df[f"return_{n}d"] = close.pct_change(n)
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA values and slope/crossover features."""
    close = df["Close"]
    for period in MA_PERIODS:
        ma = close.rolling(period).mean()
        df[f"sma_{period}"] = ma
        df[f"sma_{period}_slope"] = ma.diff(5) / ma.shift(5)  # 5-day slope
        df[f"price_vs_sma_{period}"] = (close - ma) / ma      # % above/below MA

    # Crossover signals
    if 10 in MA_PERIODS and 20 in MA_PERIODS:
        df["sma10_vs_sma20"] = df["sma_10"] / df["sma_20"] - 1
    if 20 in MA_PERIODS and 50 in MA_PERIODS:
        df["sma20_vs_sma50"] = df["sma_20"] / df["sma_50"] - 1

    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI indicator."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD line, signal, and histogram."""
    close = df["Close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    df["macd_hist_slope"] = df["macd_hist"].diff(3)
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Add Average True Range (volatility)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / close  # normalized ATR
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume ratio and trends."""
    volume = df["Volume"]
    df["volume_ratio_20d"] = volume / volume.rolling(20).mean()
    df["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    return df


def add_gap_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add gap and daily range metrics."""
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])  # 0=low, 1=high
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature builders on a raw OHLCV dataframe."""
    df = df.copy()
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_gap_range(df)
    return df
