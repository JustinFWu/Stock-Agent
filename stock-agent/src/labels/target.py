import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import FORWARD_DAYS, TRADING_DAYS

# Both labels leave the last FORWARD_DAYS rows NaN — their future has not happened yet.


def add_forward_return(df: pd.DataFrame, horizon: int = FORWARD_DAYS) -> pd.DataFrame:
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df["future_return"] = (future_close - df["Close"]) / df["Close"]
    return df


def add_forward_vol(df: pd.DataFrame, horizon: int = FORWARD_DAYS) -> pd.DataFrame:
    # Measured over t+1..t+horizon, excluding t, which is already known at forecast time.
    # Uncentred by convention: at a weekly horizon the drift term is negligible against
    # the noise and estimating it costs more than it buys.
    df = df.copy()
    log_return = np.log(df["Close"] / df["Close"].shift(1))
    squared = log_return ** 2

    # rolling(h).mean() at t+h is the mean over t+1..t+h; shifting back by h lands it on t.
    forward_var = squared.rolling(horizon).mean().shift(-horizon)
    forward_vol = np.sqrt(forward_var * TRADING_DAYS)

    # Genuinely flat closes (a halt, a data gap) give zero vol, which is not a real
    # observation and would blow up the log target the models train on.
    df["forward_vol"] = forward_vol.where(forward_vol > 0)

    # A walk-forward split must purge training rows whose outcome reaches into the test
    # block, and counting calendar dates gets that wrong: the label looks ahead `horizon`
    # *observed bars*, which for a name with gaps can land weeks past the horizon-th date.
    df["label_end"] = df.index.to_series().shift(-horizon)
    return df


def add_labels(df: pd.DataFrame, horizon: int = FORWARD_DAYS) -> pd.DataFrame:
    df = add_forward_return(df, horizon)
    df = add_forward_vol(df, horizon)
    return df
