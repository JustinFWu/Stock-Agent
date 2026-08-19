"""
Metrics, including the relative block the Phase 3 gate is meant to read.

The relative statistics matter more than the absolute ones here, because the
absolute ones are the numbers that a survivorship-biased universe inflates.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from config import TRADING_DAYS
from src.backtest.metrics import summarize_relative, summarize


def curve(daily_return: float, days: int = TRADING_DAYS * 4) -> pd.Series:
    dates = pd.bdate_range("2020-01-01", periods=days)
    return pd.Series(100.0 * (1 + daily_return) ** np.arange(days), index=dates)


def test_summarize_recovers_a_known_cagr():
    """A curve compounding at a fixed rate has a CAGR that can be computed by hand."""
    equity = curve(0.0004)
    zeros = pd.Series(0.0, index=equity.index)
    metrics = summarize(equity, zeros, zeros)
    assert metrics["cagr"] == pytest.approx((1.0004) ** TRADING_DAYS - 1, rel=1e-6)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    # A constant return has no risk, so no risk-adjusted ratio exists. The float
    # noise left in its standard deviation must not be mistaken for volatility.
    assert np.isnan(metrics["sharpe"])


def test_drawdown_is_measured_from_the_peak():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0],
                       index=pd.bdate_range("2020-01-01", periods=4))
    zeros = pd.Series(0.0, index=equity.index)
    assert summarize(equity, zeros, zeros)["max_drawdown"] == pytest.approx(-0.5)


def test_cost_drag_is_annualised_against_average_nav():
    # One extra mark, so the curve spans exactly TRADING_DAYS return periods.
    equity = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=TRADING_DAYS + 1))
    costs = pd.Series(0.0, index=equity.index)
    costs.iloc[0] = 1.0  # 1% of NAV, paid once, over one year
    traded = pd.Series(0.0, index=equity.index)
    assert summarize(equity, costs, traded)["cost_drag_ann"] == pytest.approx(0.01, rel=1e-3)


def test_turnover_is_measured_against_the_nav_of_the_day():
    """
    Turnover accumulates day by day, not against the average NAV over the run.

    On a curve that grows, dividing early trading by a mean NAV several times its
    actual size understates turnover — and understates it in the flattering
    direction. Here the same fraction of NAV is traded on each of two days at very
    different NAV levels, so a correct measure returns exactly 2 x that fraction.
    """
    equity = pd.Series([100.0, 100.0, 1000.0, 1000.0],
                       index=pd.bdate_range("2020-01-01", periods=4))
    traded = pd.Series([10.0, 0.0, 100.0, 0.0], index=equity.index)  # 10% of NAV, twice
    costs = pd.Series(0.0, index=equity.index)

    years = 3 / TRADING_DAYS
    metrics = summarize(equity, costs, traded)
    assert metrics["ann_turnover"] == pytest.approx(0.2 / years)


def test_identical_curves_have_no_information_ratio():
    """
    A strategy that is its own baseline adds nothing, and must report nothing.

    This is exactly the Phase 2 situation — the placeholder strategy is the
    baseline — so the degenerate case has to produce clean output rather than a
    division by float noise. A volatile curve is used so the individual Sharpes
    are real numbers and only the *active* statistics collapse.
    """
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=TRADING_DAYS * 2)
    equity = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates))),
                       index=dates)

    relative = summarize_relative(equity, equity)
    assert relative["excess_cagr"] == pytest.approx(0.0, abs=1e-12)
    assert relative["tracking_error"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(relative["information_ratio"])
    assert relative["sharpe_diff"] == pytest.approx(0.0, abs=1e-12)


def test_summarize_relative_cancels_a_shared_component():
    """
    The point of the whole exercise: a common inflation must not show up as skill.

    Strategy and baseline here share an identical drift and differ only by noise
    with zero mean. Excess CAGR should be near zero even though both curves have
    a large absolute return.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=TRADING_DAYS * 5)
    common = 0.0006 + rng.normal(0, 0.01, len(dates))
    noise = rng.normal(0, 0.002, len(dates))

    baseline = pd.Series(100 * np.cumprod(1 + common), index=dates)
    strategy = pd.Series(100 * np.cumprod(1 + common + noise), index=dates)

    relative = summarize_relative(strategy, baseline)
    assert abs(relative["excess_cagr"]) < 0.02
    assert relative["tracking_error"] > 0
    assert relative["beta_to_baseline"] == pytest.approx(1.0, abs=0.1)


def test_summarize_relative_needs_overlapping_dates():
    early = curve(0.0004)
    late = curve(0.0004)
    late.index = late.index + pd.DateOffset(years=20)
    with pytest.raises(ValueError, match="overlap"):
        summarize_relative(early, late)
