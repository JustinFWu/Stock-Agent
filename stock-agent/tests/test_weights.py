"""
The risk constraints applied to every strategy's proposal.

These run in one place so a new strategy inherits them and cannot forget them,
which also means a bug here silently affects every strategy at once.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.data.universe import UniverseSpec
from src.strategy.weights import EqualWeightStrategy, target_weights

UNIVERSE = UniverseSpec(id="synthetic", tickers=("AAA", "BBB", "CCC"),
                        point_in_time=True, caveats=())


class FixedProposal:
    """Returns whatever it was constructed with, so constraints can be tested alone."""

    name = "fixed"

    def __init__(self, weights: dict):
        self.weights = weights

    def propose(self, as_of, history, candidates):
        return pd.Series(self.weights)


def weights_for(panel, proposal: dict, **kwargs) -> pd.Series:
    strategy = FixedProposal(proposal)
    return target_weights(panel.dates[-1], panel, strategy, universe=UNIVERSE,
                          min_history=5, **kwargs)


def test_conviction_scores_keep_their_shape(flat_panel):
    """
    A proposal on a scale other than portfolio fractions must survive the limits.

    Clipping before scaling would push every one of these above-cap scores down to
    exactly `max_weight`, and the gross rescaling would then hand back a perfectly
    equal-weight portfolio — the strategy's entire signal deleted, silently, with
    output that looks completely reasonable.
    """
    # Cap lifted so only the scaling step is under test; the cap has its own test.
    result = weights_for(flat_panel, {"AAA": 8.0, "BBB": 4.0, "CCC": 2.0}, max_weight=1.0)

    assert result.sum() == pytest.approx(1.0)
    assert result["AAA"] == pytest.approx(result["BBB"] * 2)
    assert result["BBB"] == pytest.approx(result["CCC"] * 2)


def test_per_name_cap_binds(flat_panel):
    result = weights_for(flat_panel, {"AAA": 0.9, "BBB": 0.05, "CCC": 0.05}, max_weight=0.4)
    assert result["AAA"] == pytest.approx(0.4)


def test_deliberate_cash_is_not_scaled_up(flat_panel):
    """
    A proposal summing to less than 1 means "hold cash" and must be left alone.

    This is how Phase 3's volatility targeting expresses that the market is
    dangerous. Anything that renormalised to fully invested would delete the one
    decision the sizing layer exists to make.
    """
    result = weights_for(flat_panel, {"AAA": 0.2, "BBB": 0.1}, max_weight=0.5)
    assert result.sum() == pytest.approx(0.3)


def test_gross_ceiling_scales_down(flat_panel):
    result = weights_for(flat_panel, {"AAA": 0.6, "BBB": 0.6}, max_gross=0.8, max_weight=1.0)
    assert result.sum() == pytest.approx(0.8)
    assert result["AAA"] == pytest.approx(result["BBB"])


def test_non_finite_and_negative_proposals_are_dropped(flat_panel):
    result = weights_for(flat_panel, {"AAA": 0.5, "BBB": float("nan"), "CCC": -0.3})
    assert list(result.index) == ["AAA"]


def test_a_name_outside_the_universe_is_ignored(flat_panel):
    """Membership is enforced here, not left to the strategy's good behaviour."""
    result = weights_for(flat_panel, {"AAA": 0.5, "ZZZ": 0.5})
    assert "ZZZ" not in result.index


def test_weights_are_rounded_and_sorted(flat_panel):
    result = target_weights(flat_panel.dates[-1], flat_panel, EqualWeightStrategy(),
                            universe=UNIVERSE, min_history=5)
    assert list(result.index) == sorted(result.index)
    assert all(w == round(w, 10) for w in result)
