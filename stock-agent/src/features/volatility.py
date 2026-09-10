import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RV_WINDOWS, EWMA_LAMBDA, TRADING_DAYS

# Three estimator families because they read different parts of the bar and carry
# different noise: close-to-close is assumption-free and matches the forward label,
# Parkinson is ~5x more efficient but blind to gaps, Garman-Klass uses the full bar.

_VOL_FLOOR = 1e-6  # a dead or halted stretch would otherwise produce a zero we take logs of


def _annualize(variance_per_day: pd.Series) -> pd.Series:
    return np.sqrt((variance_per_day * TRADING_DAYS).clip(lower=_VOL_FLOOR))


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_close_to_close_vol(df: pd.DataFrame) -> pd.DataFrame:
    squared = df["log_return_1d"] ** 2
    for w in RV_WINDOWS:
        df[f"rv_{w}d"] = _annualize(squared.rolling(w).mean())
    return df


def add_parkinson_vol(df: pd.DataFrame) -> pd.DataFrame:
    log_hl = np.log(df["High"] / df["Low"])
    per_bar = (log_hl ** 2) / (4 * np.log(2))
    for w in RV_WINDOWS:
        if w == 1:
            continue  # a one-bar range estimate is the raw bar; averaging gains nothing
        df[f"park_{w}d"] = _annualize(per_bar.rolling(w).mean())
    return df


def add_garman_klass_vol(df: pd.DataFrame) -> pd.DataFrame:
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    # The estimator is unbiased, not non-negative, so single bars can come out negative.
    # The floor therefore applies to the window average, not per bar.
    per_bar = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    for w in RV_WINDOWS:
        if w == 1:
            continue
        df[f"gk_{w}d"] = _annualize(per_bar.rolling(w).mean())
    return df


def add_ewma_vol(df: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    # RiskMetrics EWMA. Also the baseline the model must beat, computed here so both
    # uses read one definition and cannot drift apart.
    squared = df["log_return_1d"] ** 2
    ewma_var = squared.ewm(alpha=1 - lam, adjust=False).mean()
    df["ewma_vol"] = _annualize(ewma_var)
    return df


def add_vol_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    # Second-order structure. A ratio above 1 means vol is elevated against its own
    # recent regime, and vol mean-reverts; high vol_of_vol means the current level is
    # less trustworthy.
    df["vol_ratio_5_21"] = df["rv_5d"] / df["rv_21d"]
    df["vol_ratio_21_63"] = df["rv_21d"] / df["rv_63d"]
    df["vol_of_vol"] = df["rv_21d"].rolling(63).std() / df["rv_21d"]
    return df


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_log_return(df)
    df = add_close_to_close_vol(df)
    df = add_parkinson_vol(df)
    df = add_garman_klass_vol(df)
    df = add_ewma_vol(df)
    df = add_vol_dynamics(df)
    return df
