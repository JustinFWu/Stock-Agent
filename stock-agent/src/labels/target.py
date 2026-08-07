"""
Forward-looking labels.

  add_forward_return  the realized return over the next FORWARD_DAYS. Not a training
                      target here — it is the substrate the Phase 3 momentum backtest
                      books its P&L against.
  add_forward_vol     the training target for this branch: annualized realized
                      volatility over the next FORWARD_DAYS.

Both leave the last FORWARD_DAYS rows NaN, since their future has not happened yet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import FORWARD_DAYS, TRADING_DAYS


def add_forward_return(df: pd.DataFrame, horizon: int = FORWARD_DAYS) -> pd.DataFrame:
    """Realized simple return from this close to the close `horizon` bars ahead."""
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df["future_return"] = (future_close - df["Close"]) / df["Close"]
    return df


def add_forward_vol(df: pd.DataFrame, horizon: int = FORWARD_DAYS) -> pd.DataFrame:
    """
    Annualized realized volatility over the NEXT `horizon` bars — the prediction target.

    Measured close-to-close from daily log returns over t+1 .. t+horizon, deliberately
    excluding the bar at t, which is already known when the forecast is made. Uncentred
    (mean not subtracted), the standard realized-variance convention: at a weekly
    horizon the drift term is negligible against the noise and estimating it costs more
    than it buys.

    A window of genuinely flat closes — a halt, or a data gap — produces zero
    volatility, which is not a real observation and would blow up the log target the
    models train on. Those rows are set NaN and drop out.
    """
    df = df.copy()
    log_return = np.log(df["Close"] / df["Close"].shift(1))
    squared = log_return ** 2

    # rolling(h).mean() at t+h is the mean over t+1..t+h; shifting back by h lands it on t.
    forward_var = squared.rolling(horizon).mean().shift(-horizon)
    forward_vol = np.sqrt(forward_var * TRADING_DAYS)

    df["forward_vol"] = forward_vol.where(forward_vol > 0)
    return df


def add_labels(df: pd.DataFrame, horizon: int = FORWARD_DAYS) -> pd.DataFrame:
    """Attach every forward label used downstream."""
    df = add_forward_return(df, horizon)
    df = add_forward_vol(df, horizon)
    return df
