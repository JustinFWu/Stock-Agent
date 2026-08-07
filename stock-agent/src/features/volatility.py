"""
Realized-volatility features.

The premise of this branch: forward returns are not predictable from price history,
but forward volatility is — realized vol is strongly autocorrelated, which is exactly
what makes a forecast possible. Everything here is a backward-looking estimate of how
volatile a name has been, at several horizons and by several estimators.

Three estimator families, because they use different parts of the bar and carry
different amounts of noise:

  close-to-close   only the close. Noisiest, but assumption-free and the thing the
                   forward label is actually measured with.
  Parkinson        the high-low range. Roughly 5x more efficient than close-to-close
                   for the same window, but blind to overnight moves.
  Garman-Klass     the full OHLC bar. More efficient still; can go slightly negative
                   on a single bar, so it is floored once averaged.

All outputs are annualized volatilities (not variances), so they are directly
comparable to each other and to the forward label.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RV_WINDOWS, EWMA_LAMBDA, TRADING_DAYS

_VOL_FLOOR = 1e-6  # keeps a dead/halted stretch from producing a zero we later take logs of


def _annualize(variance_per_day: pd.Series) -> pd.Series:
    return np.sqrt((variance_per_day * TRADING_DAYS).clip(lower=_VOL_FLOOR))


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Daily log return — the building block for every variance estimate below."""
    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_close_to_close_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Annualized close-to-close realized vol over each window in RV_WINDOWS."""
    squared = df["log_return_1d"] ** 2
    for w in RV_WINDOWS:
        df[f"rv_{w}d"] = _annualize(squared.rolling(w).mean())
    return df


def add_parkinson_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parkinson range volatility. Uses only the high-low range, which extracts far more
    information per bar than the close alone, at the cost of ignoring overnight gaps.
    """
    log_hl = np.log(df["High"] / df["Low"])
    per_bar = (log_hl ** 2) / (4 * np.log(2))
    for w in RV_WINDOWS:
        if w == 1:
            continue  # a one-bar range estimate is just the raw bar; no averaging to gain
        df[f"park_{w}d"] = _annualize(per_bar.rolling(w).mean())
    return df


def add_garman_klass_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garman-Klass volatility, using the whole OHLC bar. Individual bars can come out
    negative (the estimator is unbiased, not non-negative), which is why the floor is
    applied after the window average rather than per bar.
    """
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    per_bar = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    for w in RV_WINDOWS:
        if w == 1:
            continue
        df[f"gk_{w}d"] = _annualize(per_bar.rolling(w).mean())
    return df


def add_ewma_vol(df: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """
    RiskMetrics EWMA volatility: var_t = lam * var_{t-1} + (1 - lam) * r_t^2.

    This is also the EWMA *baseline* the model has to beat — it is computed here as a
    feature so both uses read from one definition and cannot drift apart.
    """
    squared = df["log_return_1d"] ** 2
    ewma_var = squared.ewm(alpha=1 - lam, adjust=False).mean()
    df["ewma_vol"] = _annualize(ewma_var)
    return df


def add_vol_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second-order structure: how volatility is currently moving, not just its level.

    vol_ratio_*  term structure — short-window vol against a longer one. Above 1 means
                 vol is elevated relative to its own recent regime, and vol mean-reverts.
    vol_of_vol   dispersion of the 21-day vol over the last quarter, scaled by its level.
                 High vol-of-vol means the current level is less trustworthy.
    """
    df["vol_ratio_5_21"] = df["rv_5d"] / df["rv_21d"]
    df["vol_ratio_21_63"] = df["rv_21d"] / df["rv_63d"]
    df["vol_of_vol"] = df["rv_21d"].rolling(63).std() / df["rv_21d"]
    return df


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every volatility estimator on a raw OHLCV frame."""
    df = df.copy()
    df = add_log_return(df)
    df = add_close_to_close_vol(df)
    df = add_parkinson_vol(df)
    df = add_garman_klass_vol(df)
    df = add_ewma_vol(df)
    df = add_vol_dynamics(df)
    return df
