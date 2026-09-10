"""
Execution-time correctness: what the engine may know at the open, what it does
when it cannot trade, and what it refuses to run at all.

These cover the two P1 defects in the backtest loop. Both produced complete,
plausible equity curves — which is why neither showed up in the existing suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import make_panel
from src.backtest.costs import ZERO_COSTS, CostModel
from src.backtest.engine import run_backtest
from src.backtest.portfolio import Portfolio, execute
from src.data.panel import PricePanel
from src.data.universe import UniverseSpec
from src.strategy.weights import EqualWeightStrategy

UNIVERSE = UniverseSpec(id="synthetic-2", tickers=("AAA", "BBB"),
                        point_in_time=True, caveats=())
THREE_NAME_UNIVERSE = UniverseSpec(id="synthetic", tickers=("AAA", "BBB", "CCC"),
                                   point_in_time=True, caveats=())


def run(panel, universe=UNIVERSE, **kwargs):
    settings = {"universe": universe, "costs": ZERO_COSTS, "min_history": 5,
                "max_weight": 1.0, "rebalance": "M"}
    settings.update(kwargs)
    return run_backtest(panel, EqualWeightStrategy(), **settings)


def second_execution_day(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """
    The morning after the second month-end — the first fill where a book already exists.

    The first execution day is uninformative here: the portfolio is all cash, so NAV
    does not depend on any holding's mark and the defect under test cannot show.
    """
    month_ends = pd.Series(dates, index=dates).resample("ME").last().dropna()
    return dates[dates.get_loc(month_ends.iloc[1]) + 1]


def halted_panel(gap_close=None):
    """
    Two drifting names, with BBB missing its open on the second execution day.

    `gap_close` overrides BBB's *closing* price that same day. The pair matters
    because nothing knowable at the open differs between the two variants.
    """
    dates = pd.bdate_range("2020-01-01", periods=60)
    prices = pd.DataFrame(
        {"AAA": 100.0 * 1.004 ** np.arange(len(dates)),
         "BBB": 100.0 * 0.997 ** np.arange(len(dates))},
        index=dates,
    )
    panel = make_panel(prices)
    execution_day = second_execution_day(dates)

    opens = panel.opens.copy()
    opens.loc[execution_day, "BBB"] = np.nan  # halted at the open: cannot be traded

    closes = panel.closes.copy()
    if gap_close is not None:
        closes.loc[execution_day, "BBB"] = gap_close

    return PricePanel(opens=opens, highs=panel.highs, lows=panel.lows,
                      closes=closes, volumes=panel.volumes), execution_day


def test_an_execution_days_close_cannot_change_that_mornings_orders():
    """
    The look-ahead regression — and it reached the whole book, not just one name.

    A holding with no bar at the open still has to be valued, and the fallback used
    to be that day's *close*, which is not knowable at the open. A halted name's
    evening price therefore set the NAV every other name was sized against:
    measured, moving one name's close from 100 to 50 turned another name's morning
    order from nothing into a 125-share sale. The fallback is now the previous
    session's close, which is identical across the two runs below.
    """
    orders = {}
    for gap_close in (100.0, 50.0):
        panel, execution_day = halted_panel(gap_close)
        fills = run(panel).fills
        orders[gap_close] = fills[(fills["date"] == execution_day)
                                  & (fills["ticker"] == "AAA")]["shares"].sum()

    assert orders[100.0] == pytest.approx(orders[50.0]), \
        "an unknowable closing price changed the size of that morning's trades"


def test_a_blocked_trade_is_carried_to_the_next_session():
    """
    A halt defers a trade; it does not cancel it.

    Dropped silently, a name the strategy wanted stayed unbought — or one it had
    dropped stayed held — until the next scheduled rebalance, however far off that
    was, with nothing in the output saying so.
    """
    panel, execution_day = halted_panel()
    result = run(panel)
    fills = result.fills

    on_the_day = fills[(fills["date"] == execution_day) & (fills["ticker"] == "BBB")]
    assert on_the_day.empty, "BBB had no open and could not have been traded"

    later = fills[(fills["date"] > execution_day) & (fills["ticker"] == "BBB")]
    assert not later.empty, "the deferred trade was never placed"

    next_session = panel.dates[panel.dates.get_loc(execution_day) + 1]
    assert later["date"].min() == next_session, "the retry did not happen at the next open"
    assert any("carried an unfilled trade" in c for c in result.caveats)


def test_deferring_does_not_rebalance_the_rest_of_the_book():
    """
    The retry is restricted to the names that were actually blocked.

    Re-planning the whole target would drag every position that had since drifted
    back through the band, turning one halted name into a portfolio-wide trade on a
    day the strategy never asked to rebalance.
    """
    panel, execution_day = halted_panel()
    next_session = panel.dates[panel.dates.get_loc(execution_day) + 1]

    retried = run(panel).fills
    retried = retried[retried["date"] == next_session]
    assert set(retried["ticker"]) == {"BBB"}


# --------------------------------------------------------------------------- #
# Affordability under fixed fees
# --------------------------------------------------------------------------- #

def flat_fee_model(minimum: float) -> CostModel:
    """Fees that do not shrink with the order — the case proportional scaling misses."""
    return CostModel(min_commission=minimum, half_spread_bps=0.0, impact_coef=0.0,
                     fallback_participation=0.0, fallback_daily_vol=0.0)


def test_a_minimum_commission_cannot_push_cash_negative():
    """
    Scaling buys by available/required assumes every cost shrinks with the order.

    A fixed minimum fee does not: the final fill recomputes the same flat charge
    against a smaller notional and overruns the cash reserved for it. Measured
    before the fix, this exact setup finished at -0.50.
    """
    portfolio = Portfolio(cash=100.0)

    execute(portfolio, pd.Timestamp("2020-01-02"), pd.Series({"AAA": 2.0}),
            pd.Series({"AAA": 100.0}), pd.Series({"AAA": 1e9}),
            pd.Series({"AAA": 0.01}), flat_fee_model(1.0))

    assert portfolio.cash >= 0, "the buy scaling borrowed to pay its own fixed fee"


def test_buys_are_skipped_when_the_fees_alone_exceed_the_balance():
    """Below the minimum fee there is no affordable trade, and zero is the honest size."""
    portfolio = Portfolio(cash=10.0)

    fills = execute(portfolio, pd.Timestamp("2020-01-02"), pd.Series({"AAA": 5.0}),
                    pd.Series({"AAA": 100.0}), pd.Series({"AAA": 1e9}),
                    pd.Series({"AAA": 0.01}), flat_fee_model(50.0))

    assert fills == []
    assert portfolio.cash == pytest.approx(10.0)


def test_several_simultaneous_buys_stay_within_the_balance():
    """The scale is solved against the whole basket, not one name at a time."""
    costs = CostModel(min_commission=1.0, half_spread_bps=2.0, impact_coef=0.5)
    names = ["AAA", "BBB", "CCC"]
    portfolio = Portfolio(cash=500.0)

    execute(portfolio, pd.Timestamp("2020-01-02"),
            pd.Series(dict.fromkeys(names, 4.0)), pd.Series(dict.fromkeys(names, 100.0)),
            pd.Series(dict.fromkeys(names, 1e7)), pd.Series(dict.fromkeys(names, 0.02)),
            costs)

    assert portfolio.cash >= 0


def test_an_affordable_order_is_not_scaled_down():
    """The bisection must return exactly 1.0 when the whole order fits."""
    portfolio = Portfolio(cash=100_000.0)

    execute(portfolio, pd.Timestamp("2020-01-02"), pd.Series({"AAA": 10.0}),
            pd.Series({"AAA": 100.0}), pd.Series({"AAA": 1e9}),
            pd.Series({"AAA": 0.01}), flat_fee_model(1.0))

    assert portfolio.shares["AAA"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Settings that would make a run meaningless
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    {"initial_cash": 0.0},
    {"initial_cash": -1000.0},
    {"max_weight": float("nan")},
    {"max_gross": float("nan")},
    {"no_trade_band": -0.01},
])
def test_nonsensical_settings_are_refused(flat_panel, bad):
    """
    NaN is the one worth stating: every comparison against it is False, so the limit
    is simply not enforced and the run returns a plausible curve computed without it.
    """
    with pytest.raises(ValueError):
        run(flat_panel, universe=THREE_NAME_UNIVERSE, **bad)


@pytest.mark.parametrize("bad", [{"max_weight": -0.1}, {"max_gross": -1.0}])
def test_negative_ceilings_are_refused(flat_panel, bad):
    """A negative ceiling flips positive proposals through zero into short positions."""
    with pytest.raises(ValueError, match="negative"):
        run(flat_panel, universe=THREE_NAME_UNIVERSE, **bad)


def test_a_negative_cost_rate_is_refused():
    """A cost model that pays you to trade rewards the churn it exists to penalise."""
    with pytest.raises(ValueError, match="non-negative"):
        CostModel(half_spread_bps=-2.0)


def test_excluded_names_are_disclosed_on_the_result(flat_panel):
    """
    A run on 74 of 82 names is a different experiment from one on all 82.

    The exclusions used to be printed by the loader and then forgotten, leaving the
    console the only record that the universe had changed under the result.
    """
    thinned = PricePanel(opens=flat_panel.opens, highs=flat_panel.highs,
                         lows=flat_panel.lows, closes=flat_panel.closes,
                         volumes=flat_panel.volumes, missing=("XYZ", "QRS"))
    result = run(thinned, universe=THREE_NAME_UNIVERSE)

    assert any("no cached bars" in c and "XYZ" in c for c in result.caveats)
