# The pre-trade veto. Every limit before this one was enforced at weight formation, on a
# vector, once. These tests are about the book that actually came back — which drifts from
# the target by design, through scaled buys, deferred names and a month between rebalances.


import numpy as np
import pandas as pd
import pytest

from stock_agent.data.universe import UniverseSpec
from stock_agent.execution.broker import Account, OrderIntent, Side
from stock_agent.execution.veto import Rejection, veto

SESSION = pd.Timestamp("2026-09-18")
PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}, dtype=float)

# AAA and BBB share a sector so the group cap can bind without the per-name cap binding first.
UNIVERSE = UniverseSpec(id="test", tickers=("AAA", "BBB", "CCC"), point_in_time=False,
                        caveats=(), sectors={"AAA": "XLK", "BBB": "XLK", "CCC": "XLF"})

EQUITY = 100_000.0


def intent(ticker="AAA", qty=50.0, side=Side.BUY) -> OrderIntent:
    return OrderIntent(session_date=SESSION, ticker=ticker, qty=qty, side=side)


def check(intents, *, positions=None, buying_power=EQUITY, marks=PRICES, **kwargs):
    return veto(intents,
                positions=positions or {},
                account=Account(cash=buying_power, equity=EQUITY, buying_power=buying_power),
                marks=marks, universe=UNIVERSE, **kwargs)


def test_a_compliant_rebalance_passes_untouched():
    # The expected answer on every ordinary session: the weight path already applied these
    # caps, so nothing here should fire. A rejection is an alert, not routine flow control.
    result = check([intent("AAA", 50.0), intent("BBB", 100.0)])

    assert result.is_clean
    assert len(result.allowed) == 2


def test_the_kill_switch_stops_everything():
    result = check([intent("AAA"), intent("BBB")], kill_switch_tripped=True)

    assert result.allowed == ()
    assert {r.reason for r in result.rejected} == {Rejection.KILL_SWITCH}


def test_a_buy_through_the_per_name_cap_is_rejected():
    # 150 shares at 100 is 15% of equity against a 10% cap.
    result = check([intent("AAA", 150.0)], max_weight=0.10)

    assert result.allowed == ()
    assert result.rejected[0].reason is Rejection.PER_NAME_CAP


def test_the_cap_binds_on_the_post_trade_book_not_the_order():
    # 60 shares is 6% of equity on its own and would pass any check of the order alone.
    # Landing on an existing 8% position it breaches, which is the whole point of checking
    # against the broker's book rather than against the intent.
    result = check([intent("AAA", 60.0)], positions={"AAA": 80.0}, max_weight=0.10)

    assert result.rejected[0].reason is Rejection.PER_NAME_CAP


def test_a_sell_that_reduces_an_overweight_name_is_allowed():
    # Rejecting this would pin the breach in place. The test is whether the intent makes
    # the position worse, not whether the position is compliant afterwards.
    result = check([intent("AAA", 50.0, Side.SELL)], positions={"AAA": 200.0},
                   max_weight=0.10)

    assert result.is_clean


def test_a_sell_that_would_open_a_short_is_rejected():
    # The real long-only check, and it is on the post-trade quantity rather than the side:
    # a plain SELL of more than the book holds opens a short exactly as SELL_SHORT does.
    result = check([intent("AAA", 20.0, Side.SELL)], positions={"AAA": 10.0})

    assert result.rejected[0].reason is Rejection.WOULD_OPEN_SHORT


def test_an_explicit_short_is_rejected_by_the_same_test():
    result = check([intent("AAA", 5.0, Side.SELL_SHORT)])
    assert result.rejected[0].reason is Rejection.WOULD_OPEN_SHORT


def test_covering_an_existing_short_is_allowed():
    # A position getting smaller, even though it is negative on both sides of the trade.
    result = check([intent("AAA", 10.0, Side.BUY_TO_COVER)], positions={"AAA": -20.0})
    assert result.is_clean


def test_the_sector_cap_binds_on_the_group():
    # AAA at 10% plus BBB at 20% is the 30% ceiling exactly; one more share breaches it.
    result = check([intent("BBB", 20.0)],
                   positions={"AAA": 100.0, "BBB": 400.0},
                   max_weight=1.0, max_sector_weight=0.30)

    assert result.rejected[0].reason is Rejection.SECTOR_CAP


def test_a_name_outside_the_breaching_sector_is_left_alone():
    result = check([intent("BBB", 20.0), intent("CCC", 4.0)],
                   positions={"AAA": 100.0, "BBB": 400.0},
                   max_weight=1.0, max_sector_weight=0.30)

    assert [i.ticker for i in result.allowed] == ["CCC"]
    assert [r.intent.ticker for r in result.rejected] == ["BBB"]


def test_the_gross_cap_binds_on_the_whole_book():
    result = check([intent("AAA", 200.0)], positions={"BBB": 1800.0},
                   max_weight=1.0, max_sector_weight=1.0, max_gross=1.0)

    assert result.rejected[0].reason is Rejection.GROSS_CAP


def test_buys_that_cannot_be_paid_for_are_rejected_not_scaled():
    # The roadmap's design test. The backtester scales buys down to fit the cash because
    # there that is an accounting convenience; here a shortfall means our arithmetic and
    # the venue's disagree, and shrinking the order would hide it.
    result = check([intent("AAA", 100.0)], buying_power=1_000.0)

    assert result.allowed == ()
    assert result.rejected[0].reason is Rejection.INSUFFICIENT_CASH
    assert result.rejected[0].intent.qty == 100.0   # unchanged, not resized


def test_sale_proceeds_fund_the_same_sessions_buys():
    # An ordinary rebalance sells to buy. Counting only the pre-trade buying power would
    # reject almost every session, and a veto that fires routinely stops being read.
    result = check([intent("BBB", 200.0, Side.SELL), intent("AAA", 100.0)],
                   positions={"BBB": 200.0}, buying_power=0.0)

    assert result.is_clean


def test_an_order_too_large_for_the_name_is_rejected():
    # A caveat in the backtester, a refusal here: an order this size against a name's daily
    # volume is more likely a sizing bug than a decision.
    adv = pd.Series({"AAA": 50_000.0}, dtype=float)
    result = check([intent("AAA", 100.0)], adv_notional=adv, max_participation=0.10)

    assert result.rejected[0].reason is Rejection.ORDER_TOO_LARGE


def test_a_normal_sized_order_passes_the_participation_check():
    adv = pd.Series({"AAA": 50_000_000.0}, dtype=float)
    assert check([intent("AAA", 100.0)], adv_notional=adv).is_clean


def test_an_unknown_adv_is_not_a_rejection():
    # The cost model treats missing volume pessimistically because it is pricing. This is
    # refusing, and refusing on an absence of data would stop the book on a data gap.
    adv = pd.Series({"BBB": 50_000_000.0}, dtype=float)
    assert check([intent("AAA", 100.0)], adv_notional=adv).is_clean


def test_an_intent_we_cannot_price_is_rejected():
    # No mark means no limit can be applied, and passing it through unchecked is exactly
    # what this layer exists to stop.
    result = check([intent("AAA", 10.0)], marks=pd.Series({"AAA": np.nan}, dtype=float))

    assert result.rejected[0].reason is Rejection.UNPRICED


def test_it_rejects_and_never_resizes():
    # Every allowed intent is the object that came in, and every rejected one keeps its
    # size. A veto that quietly shrinks an order is negotiating.
    submitted = [intent("AAA", 150.0), intent("BBB", 100.0)]
    result = check(submitted, max_weight=0.10)

    assert set(result.allowed) <= set(submitted)
    assert [r.intent.qty for r in result.rejected] == [150.0]
    assert all(i in submitted for i in result.allowed)


def test_a_nan_limit_is_refused():
    # The quiet failure this repo guards everywhere: NaN fails every comparison, so the
    # limit is not applied and the output looks like a book that respected it.
    with pytest.raises(ValueError, match="finite"):
        check([intent("AAA")], max_weight=float("nan"))


def test_nothing_in_means_nothing_out():
    result = check([])
    assert result.is_clean and result.allowed == ()


def test_the_description_names_what_was_rejected_and_why():
    # Read by whoever is woken up by it.
    result = check([intent("AAA", 150.0)], max_weight=0.10)
    assert "per_name_cap" in result.describe()
    assert "AAA" in result.describe()
