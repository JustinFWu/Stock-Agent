"""
Cost model arithmetic.

The numbers here are checked against hand-computed values rather than against
the implementation, because a cost model that is merely self-consistent is the
single easiest way to produce a backtest that is wrong in the flattering
direction.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.backtest.costs import ALPACA_COSTS, ZERO_COSTS, CostModel


def test_spread_is_charged_per_side():
    """2bp half-spread on $100k is $20 each way, regardless of size."""
    model = CostModel(half_spread_bps=2.0, impact_coef=0.0)
    rate = model.slippage_rate(notional=100_000, adv_notional=1e9, daily_vol=0.02)
    assert rate == pytest.approx(0.0002)


def test_impact_follows_the_square_root_law():
    """Quadrupling the trade doubles the impact — the defining property of the model."""
    model = CostModel(half_spread_bps=0.0, impact_coef=0.5)
    small = model.slippage_rate(1e6, adv_notional=1e9, daily_vol=0.02)
    large = model.slippage_rate(4e6, adv_notional=1e9, daily_vol=0.02)
    assert large == pytest.approx(2 * small)


def test_impact_matches_hand_computation():
    """coef * daily_vol * sqrt(participation) = 0.5 * 0.02 * sqrt(0.01) = 10bp."""
    model = CostModel(half_spread_bps=0.0, impact_coef=0.5)
    rate = model.slippage_rate(notional=1e7, adv_notional=1e9, daily_vol=0.02)
    assert rate == pytest.approx(0.001)


def test_unknown_volume_is_charged_pessimistically():
    """A missing ADV must not read as a free trade."""
    model = CostModel(half_spread_bps=0.0, impact_coef=0.5)
    unknown = model.slippage_rate(1e6, adv_notional=float("nan"), daily_vol=0.02)
    tiny_but_known = model.slippage_rate(1e6, adv_notional=1e12, daily_vol=0.02)
    assert unknown > tiny_but_known > 0


def test_measured_zero_volatility_means_no_impact():
    """Impact is a volatility-scaled quantity; with no volatility there is nothing to move."""
    assert ALPACA_COSTS.slippage_rate(1e6, 1e9, daily_vol=0.0) == pytest.approx(2e-4)


def test_unknown_volatility_is_charged_pessimistically():
    """
    A missing volatility must not read as a free trade, the way a missing ADV does not.

    This is not symmetry for its own sake. ADV and daily volatility come from the
    same bars over the same rolling window, so whenever one is missing the other
    is too — and a missing volatility that zeroed impact would make the pessimistic
    ADV fallback unreachable in the engine.
    """
    model = CostModel(half_spread_bps=0.0, impact_coef=0.5, fallback_daily_vol=0.03)
    unknown = model.slippage_rate(1e6, adv_notional=float("nan"), daily_vol=float("nan"))
    assert unknown > 0
    assert unknown == pytest.approx(0.5 * 0.03 * (0.10 ** 0.5))


def test_commission_respects_the_floor():
    model = CostModel(commission_per_share=0.005, min_commission=1.0)
    assert model.commission(shares=50, notional=5_000) == pytest.approx(1.0)
    assert model.commission(shares=1_000, notional=100_000) == pytest.approx(5.0)


def test_zero_model_is_actually_free():
    assert ZERO_COSTS.slippage_rate(1e9, 1e6, daily_vol=0.5) == 0.0
    assert ZERO_COSTS.commission(1e6, 1e9) == 0.0
