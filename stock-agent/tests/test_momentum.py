import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import make_panel
from config import MAX_GROSS, MAX_WEIGHT, MIN_SELECTED_NAMES
from src.data.universe import UniverseSpec
from src.strategy.momentum import (MomentumStrategy, PanelVolForecast, RealizedVolForecast,
                                   VolTargetedMomentum, formation_return)
from src.strategy.weights import target_weights

UNIVERSE = UniverseSpec(id="synthetic", tickers=("AAA", "BBB", "CCC"),
                        point_in_time=True, caveats=())


class FixedVols:
    name = "fixed"

    def __init__(self, vols: dict):
        self.values = vols

    def vols(self, as_of, tickers, history):
        return pd.Series(self.values, dtype=float).reindex(tickers)


def linear_closes(n: int = 100) -> pd.Series:
    return pd.Series(np.arange(1.0, n + 1.0), index=pd.bdate_range("2020-01-01", periods=n))


def test_formation_return_skips_the_recent_window():
    # 97 is the close 3 bars back, 87 is 10 bars before that. If the skip were dropped the
    # window would end at 100 and the factor would be measuring short-term reversal.
    closes = linear_closes()
    assert formation_return(closes, lookback=10, skip=3) == pytest.approx(97 / 87 - 1)


def test_formation_return_with_no_skip_ends_at_the_last_bar():
    closes = linear_closes()
    assert formation_return(closes, lookback=10, skip=0) == pytest.approx(100 / 90 - 1)


def test_formation_return_needs_the_whole_window():
    closes = linear_closes(n=13)
    assert np.isnan(formation_return(closes, lookback=10, skip=3))

    exact = linear_closes(n=14)
    assert np.isfinite(formation_return(exact, lookback=10, skip=3))


def test_formation_return_counts_observed_bars_not_calendar_rows():
    # A gap in the middle must not shorten the window. Counting index positions instead of
    # observed bars would measure a shorter, higher-vol window for exactly the names with gaps.
    closes = linear_closes()
    gapped = closes.copy()
    gapped.iloc[50:60] = np.nan

    observed = gapped.dropna()
    assert formation_return(gapped, lookback=10, skip=3) == pytest.approx(
        observed.iloc[-4] / observed.iloc[-14] - 1)


def test_selection_takes_the_top_fraction_by_formation_return(drifting_panel):
    # min_names lifted so only the fraction is under test; its own floor has its own tests.
    strategy = MomentumStrategy(lookback=100, skip=5, top_fraction=1 / 3, min_names=1)
    selected = strategy.select(drifting_panel, drifting_panel.dates[-1], ["AAA", "BBB", "CCC"])

    assert selected == ["AAA"]


def test_selection_ignores_prices_after_as_of(drifting_panel):
    # CCC is the worst name up to the cut, then spikes. A selection that saw the spike would
    # pick it; the truncation in `target_weights` is what makes that impossible.
    as_of = drifting_panel.dates[300]
    closes = drifting_panel.closes.copy()
    closes.loc[closes.index[301:], "CCC"] *= 100.0
    spiked = make_panel(closes)

    strategy = MomentumStrategy(lookback=100, skip=5, top_fraction=1 / 3, min_names=1)
    assert strategy.select(spiked.as_of(as_of), as_of, ["AAA", "BBB", "CCC"]) == ["AAA"]


def test_both_arms_of_the_gate_select_identically(drifting_panel):
    # If selection differed, the gate would be measuring the names as well as the sizing.
    selection = {"lookback": 100, "skip": 5, "top_fraction": 2 / 3}
    unscaled = MomentumStrategy(**selection)
    scaled = VolTargetedMomentum(FixedVols({"AAA": 0.2, "BBB": 0.2, "CCC": 0.2}), **selection)

    as_of = drifting_panel.dates[-1]
    names = ["AAA", "BBB", "CCC"]
    assert unscaled.select(drifting_panel, as_of, names) == scaled.select(drifting_panel, as_of, names)


def test_momentum_equal_weights_its_selection(drifting_panel):
    strategy = MomentumStrategy(lookback=100, skip=5, top_fraction=2 / 3, min_names=1)
    proposed = strategy.propose(drifting_panel.dates[-1], drifting_panel, ["AAA", "BBB", "CCC"])

    assert len(proposed) == 2
    assert proposed.sum() == pytest.approx(1.0)
    assert proposed.nunique() == 1


def test_vol_target_scales_a_single_name_to_the_target(drifting_panel):
    # One name, diagonal covariance: portfolio vol is the name's vol, so the weight is exactly
    # target / vol. cov_lookback is short enough that the correlation step declines to fit.
    strategy = VolTargetedMomentum(FixedVols({"AAA": 0.2, "BBB": 0.2, "CCC": 0.2}),
                                   vol_target=0.10, cov_lookback=5,
                                   lookback=100, skip=5, top_fraction=1 / 3, min_names=1)
    proposed = strategy.propose(drifting_panel.dates[-1], drifting_panel, ["AAA", "BBB", "CCC"])

    assert proposed["AAA"] == pytest.approx(0.5)


def test_vol_target_sizes_inversely_and_hits_the_target(drifting_panel):
    vols = {"AAA": 0.10, "BBB": 0.20, "CCC": 0.20}
    strategy = VolTargetedMomentum(FixedVols(vols), vol_target=0.10, cov_lookback=5,
                                   lookback=100, skip=5, top_fraction=2 / 3)
    proposed = strategy.propose(drifting_panel.dates[-1], drifting_panel, ["AAA", "BBB", "CCC"])

    sigma = np.array([vols[t] for t in proposed.index])
    realised = float(np.sqrt((proposed.to_numpy() ** 2 * sigma ** 2).sum()))

    assert (proposed / proposed.sum()).idxmax() == "AAA"
    assert realised == pytest.approx(0.10)


def test_a_dangerous_forecast_produces_a_deliberate_cash_position(drifting_panel):
    strategy = VolTargetedMomentum(FixedVols({"AAA": 0.80, "BBB": 0.80, "CCC": 0.80}),
                                   vol_target=0.10, cov_lookback=5,
                                   lookback=100, skip=5, top_fraction=1 / 3, min_names=1)
    proposed = strategy.propose(drifting_panel.dates[-1], drifting_panel, ["AAA", "BBB", "CCC"])

    assert proposed.sum() == pytest.approx(0.125)


def test_the_scale_cap_bounds_a_calm_forecast(drifting_panel):
    strategy = VolTargetedMomentum(FixedVols({"AAA": 0.01, "BBB": 0.01, "CCC": 0.01}),
                                   vol_target=0.10, cov_lookback=5, scale_cap=1.5,
                                   lookback=100, skip=5, top_fraction=1 / 3, min_names=1)
    proposed = strategy.propose(drifting_panel.dates[-1], drifting_panel, ["AAA", "BBB", "CCC"])

    assert proposed.sum() == pytest.approx(1.5)


def test_shrunk_correlation_raises_the_portfolio_vol_estimate(drifting_panel):
    # Correlated names are riskier than the diagonal assumption says, so the same forecast must
    # buy less of them. This is the whole reason shrinkage is in the sizing path.
    vols = FixedVols({"AAA": 0.2, "BBB": 0.2, "CCC": 0.2})
    selection = {"lookback": 100, "skip": 5, "top_fraction": 1.0}

    diagonal = VolTargetedMomentum(vols, vol_target=0.10, cov_lookback=5, **selection)
    correlated = VolTargetedMomentum(vols, vol_target=0.10, cov_lookback=250, **selection)

    as_of = drifting_panel.dates[-1]
    lean = diagonal.propose(as_of, drifting_panel, ["AAA", "BBB", "CCC"]).sum()
    full = correlated.propose(as_of, drifting_panel, ["AAA", "BBB", "CCC"]).sum()

    assert full < lean


def test_a_non_positive_vol_target_is_refused():
    with pytest.raises(ValueError, match="vol_target"):
        VolTargetedMomentum(FixedVols({}), vol_target=0.0)


def test_an_out_of_range_top_fraction_is_refused():
    with pytest.raises(ValueError, match="top_fraction"):
        MomentumStrategy(top_fraction=0.0)


def test_selection_never_falls_below_what_the_per_name_cap_can_hold(drifting_panel):
    # A decile of 82 names is eight, and eight names under a 10% cap top out at 80% gross.
    # The floor is what stops the risk limit from silently converting the rest into cash.
    strategy = MomentumStrategy(lookback=100, skip=5, top_fraction=0.01, min_names=3)
    selected = strategy.select(drifting_panel, drifting_panel.dates[-1], ["AAA", "BBB", "CCC"])

    assert len(selected) == 3


def test_the_selection_floor_cannot_exceed_the_candidates(drifting_panel):
    strategy = MomentumStrategy(lookback=100, skip=5, top_fraction=0.01, min_names=50)
    selected = strategy.select(drifting_panel, drifting_panel.dates[-1], ["AAA", "BBB", "CCC"])

    assert len(selected) == 3


def test_the_default_selection_can_reach_full_investment():
    # The invariant the floor exists to hold, asserted against the real constants rather than
    # a fixture: whatever the decile works out to, the book must be able to reach MAX_GROSS.
    assert MIN_SELECTED_NAMES * MAX_WEIGHT >= MAX_GROSS


def test_a_defaulted_strategy_carries_the_configured_floor():
    assert MomentumStrategy().min_names == MIN_SELECTED_NAMES


def test_a_meaningless_selection_floor_is_refused():
    with pytest.raises(ValueError, match="min_names"):
        MomentumStrategy(min_names=0)


def test_a_missing_forecast_drops_the_name(drifting_panel):
    strategy = VolTargetedMomentum(FixedVols({"AAA": np.nan, "BBB": 0.2, "CCC": 0.2}),
                                   vol_target=0.10, cov_lookback=5,
                                   lookback=100, skip=5, top_fraction=1.0)
    proposed = strategy.propose(drifting_panel.dates[-1], drifting_panel, ["AAA", "BBB", "CCC"])

    assert "AAA" not in proposed.index
    assert set(proposed.index) == {"BBB", "CCC"}


def test_panel_forecast_never_reads_past_as_of(drifting_panel):
    dates = drifting_panel.dates
    panel = pd.DataFrame(0.2, index=dates, columns=["AAA", "BBB", "CCC"])
    panel.loc[dates[301]:, :] = 99.0

    forecast = PanelVolForecast(panel)
    assert forecast.vols(dates[300], ["AAA"], drifting_panel)["AAA"] == pytest.approx(0.2)


def test_realized_forecast_returns_annualised_vol(drifting_panel):
    forecast = RealizedVolForecast()
    vols = forecast.vols(drifting_panel.dates[-1], ["AAA", "BBB", "CCC"], drifting_panel)

    assert len(vols) == 3
    assert (vols > 0).all()


def test_the_strategies_run_through_the_shared_weight_path(drifting_panel):
    # Nothing here may bypass the constraint layer; this is the seam the Phase 2 gate protects.
    strategy = VolTargetedMomentum(FixedVols({"AAA": 0.2, "BBB": 0.2, "CCC": 0.2}),
                                   vol_target=0.10, cov_lookback=5,
                                   lookback=100, skip=5, top_fraction=1.0)
    result = target_weights(drifting_panel.dates[-1], drifting_panel, strategy,
                            universe=UNIVERSE, min_history=110, max_sector_weight=1.0)

    assert not result.empty
    assert result.max() <= 0.10 + 1e-12
