"""
Generates the prediction target labels.

Two flavours live here:
  add_label                 - the ABSOLUTE, per-ticker label: will this stock close
                              higher 5 trading days from now? Computed on a single
                              ticker's frame.
  add_cross_sectional_labels - the RELATIVE, cross-sectional labels: how does this
                              stock's 5-day forward return rank against every *other*
                              ticker on the same date? Must be computed on the pooled
                              (stacked) frame, because it needs the whole cross-section.
"""

import numpy as np
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


def add_cross_sectional_labels(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative (cross-sectional) labels to a POOLED, multi-ticker frame.

    Unlike add_label, these compare each ticker against every other ticker on the SAME
    date, so they can only be computed once the per-ticker frames are stacked (see
    build_pooled_dataset). All normalization is strictly within a single date's
    cross-section — no future dates are ever pooled into the mean/rank — which cancels
    the common market move (beta) and keeps the memory-flagged lookahead leakage out.

    Requires the absolute label's 'future_return' column (call add_label first). Adds,
    for each row keyed by its date in the index:

      xs_mean_return     - mean future_return across all tickers on that date
      xs_demeaned_return - future_return minus that date's mean. Regression target: the
                           part of the move that is specific to this name, not the market.
      xs_rank_pct        - per-date percentile rank of future_return in [0, 1]. 1.0 = best
                           performer that day. A ready-made learning-to-rank target.
      xs_top_tercile     - 1 if in the top third that day, 0 if in the bottom third,
                           NaN for the middle third (dropped) and for undated/unlabelled
                           rows. Binary "separate winners from losers" target.

    Rows whose future_return is NaN (the last FORWARD_DAYS bars of each ticker) get NaN
    across all four columns and are naturally excluded downstream by dropna on the label.
    """
    df = pooled_df.copy()
    if "future_return" not in df.columns:
        raise ValueError("add_cross_sectional_labels needs 'future_return'; call add_label first.")

    by_date = df.groupby(level=0)["future_return"]
    df["xs_mean_return"] = by_date.transform("mean")
    df["xs_demeaned_return"] = df["future_return"] - df["xs_mean_return"]
    df["xs_rank_pct"] = by_date.transform(lambda s: s.rank(pct=True))

    # NaN rank_pct (missing future_return) fails both comparisons -> stays NaN.
    df["xs_top_tercile"] = np.where(
        df["xs_rank_pct"] >= 2 / 3, 1.0,
        np.where(df["xs_rank_pct"] <= 1 / 3, 0.0, np.nan),
    )
    return df
