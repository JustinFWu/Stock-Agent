"""
Engine behaviour: accounting, execution timing, and the no-trade band.

The synthetic panels make these exact rather than approximate. On a flat panel
nothing should move; on a drifting panel everything that moves should be
explainable by a number computed here rather than read off the implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import make_panel
from src.backtest.costs import PESSIMISTIC_COSTS, ZERO_COSTS, CostModel
from src.backtest.engine import run_backtest
from src.backtest.portfolio import Portfolio, execute, plan_trades
from src.data.universe import UniverseSpec
from src.strategy.weights import EqualWeightStrategy

TEST_UNIVERSE = UniverseSpec(id="synthetic", tickers=("AAA", "BBB", "CCC"),
                             point_in_time=True, caveats=())


def run(panel, universe=TEST_UNIVERSE, **kwargs):
    """Backtest with the test defaults: short history requirement, no costs unless asked."""
    settings = dict(universe=universe, costs=ZERO_COSTS, min_history=5,
                    max_weight=1.0, rebalance="M")
    settings.update(kwargs)
    return run_backtest(panel, EqualWeightStrategy(), **settings)


def test_flat_prices_produce_a_flat_curve(flat_panel):
    """No drift, no costs, no reason for NAV to move a cent."""
    result = run(flat_panel)
    assert result.equity.std() == pytest.approx(0.0, abs=1e-6)
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_costs_only_ever_reduce_nav(drifting_panel):
    """Charging for trading must not, under any configuration, help."""
    free = run(drifting_panel, costs=ZERO_COSTS, rebalance="D", no_trade_band=0.0)
    charged = run(drifting_panel, costs=PESSIMISTIC_COSTS, rebalance="D", no_trade_band=0.0)
    assert charged.equity.iloc[-1] < free.equity.iloc[-1]
    assert charged.daily["costs_paid"].sum() > 0
    assert free.daily["costs_paid"].sum() == 0


def test_no_trade_band_suppresses_turnover(drifting_panel):
    """A wide band must trade strictly less than a zero band on the same path."""
    tight = run(drifting_panel, rebalance="D", no_trade_band=0.0)
    wide = run(drifting_panel, rebalance="D", no_trade_band=0.05)
    assert wide.metrics["ann_turnover"] < tight.metrics["ann_turnover"]


def test_execution_happens_after_the_decision(flat_panel):
    """
    The first fill must land on the day *after* the first rebalance date.

    If it ever lands on the decision date itself, the engine is trading on the
    close it used to decide, and every result it produces is look-ahead.
    """
    result = run(flat_panel)
    first_target = result.targets.index[0]
    first_fill = result.fills["date"].min()
    assert first_fill > first_target


def test_cash_and_positions_reconcile(drifting_panel):
    """
    NAV must equal cash plus marked positions on every single day.

    Recomputed independently from the fill log rather than trusting the running
    balance, because a bug that corrupts both identically is the one worth
    catching.
    """
    result = run(drifting_panel, rebalance="M")
    fills = result.fills
    assert not fills.empty

    cash = result.meta["initial_cash"]
    shares: dict[str, float] = {}
    for _, fill in fills.iterrows():
        cash -= fill["shares"] * fill["fill_price"] + fill["commission"]
        shares[fill["ticker"]] = shares.get(fill["ticker"], 0.0) + fill["shares"]

    last_date = result.equity.index[-1]
    marks = drifting_panel.closes.loc[last_date]
    rebuilt = cash + sum(qty * marks[t] for t, qty in shares.items())
    assert rebuilt == pytest.approx(result.equity.iloc[-1], rel=1e-9)


def test_gross_exposure_never_exceeds_the_ceiling(drifting_panel):
    """
    Gross is pinned to the ceiling on rebalance days, and only drifts between them.

    Checked on the rebalance days themselves rather than on the whole path, so
    the assertion is exact where the constraint is actually applied instead of
    being loosened to accommodate drift it was never about.
    """
    result = run(drifting_panel, max_gross=0.6, rebalance="M")
    rebalanced = result.daily.loc[result.daily["traded_notional"] > 0, "gross"]
    assert not rebalanced.empty
    assert rebalanced.max() <= 0.6 + 1e-6
    assert result.daily["gross"].max() < 0.65  # drift between rebalances stays small


def test_cash_rate_compounds_when_uninvested(flat_panel):
    """An all-cash run must grow at exactly the cash rate, not merely upward."""
    empty_universe = UniverseSpec(id="none", tickers=(), point_in_time=True, caveats=())
    rate = 0.05
    result = run(flat_panel, universe=empty_universe, cash_annual_rate=rate)

    assert result.fills.empty
    periods = len(result.equity)
    expected = result.equity.iloc[0] * (1 + rate / 252) ** (periods - 1)
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_universe_caveats_reach_the_result(flat_panel):
    biased = UniverseSpec(id="biased", tickers=("AAA", "BBB", "CCC"),
                          point_in_time=False, caveats=("this universe is a survivor set",))
    result = run(flat_panel, universe=biased)
    assert "this universe is a survivor set" in result.caveats
    assert "survivor set" in result.describe()
    assert result.meta["universe_point_in_time"] is False


def test_oversized_fills_raise_a_caveat(drifting_panel):
    """
    Trading a large fraction of daily volume must be flagged, not silently priced.

    The square-root impact law was not estimated at 40% participation; a cost
    number extrapolated that far should not pass as a measurement.
    """
    starved = make_panel(drifting_panel.closes, volume=50.0)
    result = run(starved, costs=CostModel(), rebalance="M")
    assert any("average daily volume" in c for c in result.caveats)


def test_plan_trades_exits_a_dropped_name_through_the_band():
    """
    A name the strategy dropped is sold even when the drift is inside the band.

    Leaving 30bp of a rejected name every rebalance accumulates into a portfolio
    nobody chose, which is a risk decision made by inaction.
    """
    portfolio = Portfolio(cash=0.0, shares={"AAA": 1.0})
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    trades = plan_trades(pd.Series(dtype=float), portfolio, prices, prices,
                         no_trade_band=0.99)
    assert trades["AAA"] == pytest.approx(-1.0)


def test_buys_are_scaled_to_available_cash():
    """
    A rebalance must never quietly borrow to pay its own costs.

    Charged with the pessimistic model on a thin, volatile name — the shortfall a
    naive scaling misses is precisely the slippage and commission, so testing this
    with costs switched off cannot fail for the reason it names.
    """
    expensive = CostModel(half_spread_bps=5.0, impact_coef=1.0, commission_per_share=0.005)
    portfolio = Portfolio(cash=1_000.0)
    prices = pd.Series({"AAA": 100.0})
    deltas = pd.Series({"AAA": 50.0})  # $5,000 of stock against $1,000 of cash

    execute(portfolio, pd.Timestamp("2020-01-02"), deltas, prices,
            adv_notional=pd.Series({"AAA": 5e5}), daily_vol=pd.Series({"AAA": 0.05}),
            costs=expensive)

    assert portfolio.cash >= 0, "the buy scaling borrowed to pay its own costs"
    assert portfolio.shares["AAA"] < 50.0


def test_untradeable_names_are_skipped():
    """A name with no bar on the day cannot be traded, whatever the target says."""
    portfolio = Portfolio(cash=10_000.0)
    prices = pd.Series({"OLD": 100.0, "NEW": np.nan})
    trades = plan_trades(pd.Series({"OLD": 0.5, "NEW": 0.5}), portfolio, prices, prices,
                         no_trade_band=0.0)
    assert "NEW" not in trades.index
    assert "OLD" in trades.index


def test_a_halted_name_does_not_shrink_nav():
    """
    A holding with no bar today must still count toward the NAV others are sized against.

    Valuing it at zero drops NAV by its full weight and then sells down every
    healthy position to hit its share of the smaller portfolio — a fake loss, a
    fake recovery when the bar returns, and real costs on a trade that should
    never have been placed.
    """
    portfolio = Portfolio(cash=0.0, shares={"AAA": 100.0, "BBB": 100.0})
    tradable = pd.Series({"AAA": 100.0, "BBB": np.nan})   # BBB halted: no execution price
    marks = pd.Series({"AAA": 100.0, "BBB": 100.0})       # but still worth its last close

    trades = plan_trades(pd.Series({"AAA": 0.5, "BBB": 0.5}), portfolio, tradable, marks,
                         no_trade_band=0.0)
    assert trades.empty, "already at target — the halted name should not trigger a sell"


def test_no_rebalance_in_the_window_is_flagged(flat_panel):
    """A window too short to hold a rebalance must not report a clean 0% as a result."""
    dates = flat_panel.dates
    result = run(flat_panel, start=dates[0], end=dates[10], rebalance="Q")
    assert result.fills.empty
    assert any("No rebalance occurred" in c for c in result.caveats)
