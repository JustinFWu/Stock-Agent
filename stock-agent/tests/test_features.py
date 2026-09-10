"""
The feature builders — the bottom of the Phase 1 stack.

Everything the volatility model sees comes through here, and until now none of it
was tested. These are cheap assertions against hand-computed values rather than
snapshots: a snapshot test locks in whatever the code did on the day it was
written, including the bug, which is exactly the failure mode a model stack
cannot afford at its input layer.

The property that matters most is the last one: every feature here must be a
function of the current bar and earlier ones only. A single forward-looking
window would put future information into a model whose whole claim is that it
forecasts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import make_bars
from config import EWMA_LAMBDA, TRADING_DAYS
from src.features.technical import build_features
from src.features.volatility import build_volatility_features


def constant_move_bars(periods: int = 60, move: float = 0.01) -> pd.DataFrame:
    """
    Bars whose log return alternates exactly +/- `move`, so realised vol is exact.

    Alternating rather than constant keeps the price from running away over sixty
    sessions while leaving every squared return identical, which is what makes the
    expected realised volatility a closed form instead of an approximation.
    """
    dates = pd.bdate_range("2020-01-01", periods=periods)
    signs = np.where(np.arange(periods) % 2 == 0, 1.0, -1.0)
    signs[0] = 0.0  # the first bar has no prior close, so it contributes no return
    close = 100.0 * np.exp(np.cumsum(signs * move))

    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": np.full(periods, 1e6)},
        index=dates,
    )


def test_realized_vol_is_the_annualised_root_mean_square_return():
    """rv_5d over returns of a fixed magnitude must be exactly |r| * sqrt(252)."""
    move = 0.01
    features = build_volatility_features(constant_move_bars(move=move))

    expected = move * np.sqrt(TRADING_DAYS)
    assert features["rv_5d"].iloc[-1] == pytest.approx(expected)
    assert features["rv_21d"].iloc[-1] == pytest.approx(expected)


def test_ewma_vol_follows_the_riskmetrics_recursion():
    """
    The EWMA feature is also the EWMA *baseline* the model has to beat.

    It is worth pinning to the recursion by hand: if this drifts, the gate moves
    with it and the model looks better or worse for a reason that has nothing to
    do with the model.
    """
    bars = make_bars(periods=120, seed=3)
    features = build_volatility_features(bars)

    squared = (np.log(bars["Close"] / bars["Close"].shift(1)) ** 2).dropna().to_numpy()
    # `adjust=False` seeds the recursion with the first observation rather than with
    # zero, so the hand-rolled version has to start there too or it trails by the
    # weight the seed never lost.
    variance = squared[0]
    for value in squared[1:]:
        variance = EWMA_LAMBDA * variance + (1 - EWMA_LAMBDA) * value

    assert features["ewma_vol"].iloc[-1] == pytest.approx(np.sqrt(variance * TRADING_DAYS))


def test_range_estimators_stay_positive():
    """
    Garman-Klass is unbiased, not non-negative, and can come out below zero per bar.

    The floor is applied after the window average for that reason. A negative
    variance would become NaN under the square root and silently delete rows.
    """
    features = build_volatility_features(make_bars(periods=200, seed=5))

    for column in ("park_5d", "park_21d", "gk_5d", "gk_21d", "gk_63d"):
        values = features[column].dropna()
        assert not values.empty
        assert (values > 0).all(), f"{column} produced a non-positive volatility"


def test_vol_ratios_compare_the_windows_they_name():
    features = build_volatility_features(make_bars(periods=200, seed=6))
    row = features.iloc[-1]

    assert row["vol_ratio_5_21"] == pytest.approx(row["rv_5d"] / row["rv_21d"])
    assert row["vol_ratio_21_63"] == pytest.approx(row["rv_21d"] / row["rv_63d"])


def test_gap_measures_the_open_against_the_previous_close():
    bars = make_bars(periods=30, seed=7)
    features = build_features(bars)

    expected = (bars["Open"].iloc[5] - bars["Close"].iloc[4]) / bars["Close"].iloc[4]
    assert features["gap"].iloc[5] == pytest.approx(expected)


def test_atr_pct_normalises_atr_by_price():
    features = build_features(make_bars(periods=60, seed=8))
    row = features.iloc[-1]
    assert row["atr_pct"] == pytest.approx(row["atr"] / row["Close"])


def test_close_position_is_undefined_on_a_zero_range_bar():
    """
    A halted or untraded bar has no high-low range and therefore no position in it.

    Letting the division through produces an infinity, which survives `dropna` and
    reaches the model — the exact class of value `prepare` now rejects.
    """
    bars = make_bars(periods=20, seed=9)
    flat = bars.index[10]
    bars.loc[flat, ["Open", "High", "Low", "Close"]] = 100.0

    features = build_features(bars)
    assert pd.isna(features.loc[flat, "close_position"])
    assert np.isfinite(features["close_position"].dropna()).all()


def test_no_feature_reads_a_future_bar():
    """
    The firewall, checked at the feature layer rather than assumed.

    Features are built twice — once on the full history, once on history truncated
    at a chosen date — and every value on and before that date must match. A single
    forward-looking window (a centred rolling mean, a `shift(-1)`) breaks this, and
    would otherwise only show up as a suspiciously good validation score.
    """
    bars = make_bars(periods=300, seed=11)
    cutoff = bars.index[200]

    full = build_volatility_features(build_features(bars))
    truncated = build_volatility_features(build_features(bars.loc[:cutoff]))

    columns = [c for c in truncated.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
    pd.testing.assert_frame_equal(full.loc[:cutoff, columns], truncated[columns])
