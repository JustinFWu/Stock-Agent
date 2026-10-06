# Step 3 of the Phase 4 sketch: target weights against the broker's own book. Two things
# are being pinned here beyond the arithmetic — that NAV comes from the venue rather than
# from our marks, and that intents cannot be generated until recovery has accounted for
# every order the log still has open.


import numpy as np
import pandas as pd
import pytest

from stock_agent.execution.broker import Account, BrokerError, OrderIntent, Side
from stock_agent.execution.fake import FailPoint, FakeBroker, Fault
from stock_agent.execution.reconcile import reconcile
from stock_agent.execution.recovery import RecoveryReport, recover, submit_intent
from stock_agent.execution.store import OrderStore

SESSION = pd.Timestamp("2026-09-18")
PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}


def marks(**overrides) -> pd.Series:
    return pd.Series({**PRICES, **overrides}, dtype=float)


def account(cash=100_000.0, equity=100_000.0, buying_power=None) -> Account:
    return Account(cash=cash, equity=equity,
                   buying_power=cash if buying_power is None else buying_power)


def store(tmp_path) -> OrderStore:
    return OrderStore(tmp_path / "orders.jsonl")


def run(tmp_path, target, *, positions=None, acct=None, prices=None, mark_prices=None,
        recovery=None, band=0.0, **kwargs):
    prices = marks() if prices is None else prices
    return reconcile(
        SESSION, pd.Series(target, dtype=float),
        positions=positions or {},
        account=acct or account(),
        prices=prices,
        marks=marks() if mark_prices is None else mark_prices,
        store=kwargs.pop("order_store", None) or store(tmp_path),
        recovery=recovery if recovery is not None else RecoveryReport(),
        no_trade_band=band,
        **kwargs,
    )


def test_an_empty_book_buys_the_whole_target(tmp_path):
    result = run(tmp_path, {"AAA": 0.5, "BBB": 0.5})

    assert [(i.ticker, i.qty, i.side) for i in result.intents] == [
        ("AAA", 500.0, Side.BUY),   # 50% of 100,000 at 100
        ("BBB", 1000.0, Side.BUY),  # 50% of 100,000 at 50
    ]


def test_weights_become_shares_against_the_brokers_equity_not_our_marks(tmp_path):
    # The decision this module exists to make. Our marks value the book at 90,000; the
    # broker says the account is worth 120,000. The target is a fraction of the venue's
    # number, because the venue is the one that will settle the trade.
    result = run(tmp_path, {"AAA": 0.5},
                 positions={"BBB": 1800.0},          # 1800 * 50 = 90,000 marked
                 acct=account(cash=0.0, equity=120_000.0, buying_power=120_000.0))

    buy = next(i for i in result.intents if i.ticker == "AAA")
    assert buy.qty == 600.0                          # 50% of 120,000 at 100
    assert result.nav == 120_000.0
    assert result.marked_nav == 90_000.0


def test_the_nav_disagreement_is_reported(tmp_path):
    result = run(tmp_path, {}, positions={"AAA": 1000.0},
                 acct=account(cash=0.0, equity=200_000.0, buying_power=0.0))

    # 100,000 marked against 200,000 of reported equity.
    assert result.nav_disagreement == pytest.approx(-0.5)


def test_an_existing_position_is_only_topped_up(tmp_path):
    # The diff, not the target. Reconciling against a book we already hold half of must
    # not re-buy the half that is already there.
    result = run(tmp_path, {"AAA": 0.5}, positions={"AAA": 200.0})

    assert [(i.ticker, i.qty, i.side) for i in result.intents] == [("AAA", 300.0, Side.BUY)]


def test_a_name_the_target_dropped_is_sold_out(tmp_path):
    result = run(tmp_path, {"BBB": 1.0}, positions={"AAA": 100.0},
                 acct=account(cash=90_000.0, equity=100_000.0))

    sell = next(i for i in result.intents if i.ticker == "AAA")
    assert (sell.qty, sell.side) == (100.0, Side.SELL)


def test_the_no_trade_band_is_the_backtesters(tmp_path):
    # Not a second implementation. A 1% drift under a 5% band is nothing to do.
    result = run(tmp_path, {"AAA": 0.51}, positions={"AAA": 500.0}, band=0.05)
    assert result.intents == ()

    moved = run(tmp_path, {"AAA": 0.60}, positions={"AAA": 500.0}, band=0.05)
    assert [i.qty for i in moved.intents] == [100.0]


def test_a_dropped_name_is_exited_through_the_band(tmp_path):
    # Inherited from `plan_trades`: an exit is a risk decision, not a rebalancing nicety.
    result = run(tmp_path, {}, positions={"AAA": 1.0}, band=0.99)
    assert [(i.ticker, i.side) for i in result.intents] == [("AAA", Side.SELL)]


def test_a_name_with_no_price_is_blocked_not_silently_dropped(tmp_path):
    result = run(tmp_path, {"AAA": 0.5, "BBB": 0.5}, prices=marks(AAA=np.nan))

    assert result.blocked == ("AAA",)
    assert [i.ticker for i in result.intents] == ["BBB"]


def test_a_working_order_defers_its_name(tmp_path):
    # An order still live on a name means the position is moving, so any size computed
    # against it is computed against a number that is about to change.
    held = OrderIntent(session_date=SESSION, ticker="AAA", qty=5.0, side=Side.BUY)
    report = RecoveryReport(working=(_record(held),))

    result = run(tmp_path, {"AAA": 0.5, "BBB": 0.5}, recovery=report)

    assert result.deferred == ("AAA",)
    assert [i.ticker for i in result.intents] == ["BBB"]


def test_an_unresolved_order_outside_the_report_refuses_to_generate(tmp_path):
    # The structural form of "recovery strictly before generation". Until now that rule
    # lived in a comment; here it is a refusal.
    log = store(tmp_path)
    venue = FakeBroker(prices=dict(PRICES)).arm(Fault("submit", FailPoint.AFTER_ACCEPT))
    with pytest.raises(BrokerError):
        submit_intent(log, venue, OrderIntent(session_date=SESSION, ticker="AAA",
                                              qty=5.0, side=Side.BUY))

    with pytest.raises(ValueError, match="still open in the log"):
        run(tmp_path, {"AAA": 0.5}, order_store=log)


def test_generation_resumes_once_recovery_has_settled_it(tmp_path):
    log = store(tmp_path)
    venue = FakeBroker(prices=dict(PRICES)).arm(Fault("submit", FailPoint.AFTER_ACCEPT))
    with pytest.raises(BrokerError):
        submit_intent(log, venue, OrderIntent(session_date=SESSION, ticker="AAA",
                                              qty=5.0, side=Side.BUY))

    report = recover(log, venue)
    assert report.is_clean

    result = run(tmp_path, {"BBB": 0.5}, order_store=log, recovery=report)
    assert [i.ticker for i in result.intents] == ["BBB"]


def test_a_holding_we_cannot_price_is_reported(tmp_path):
    # Valued at zero by the portfolio, so it neither appears in the weights nor objects.
    result = run(tmp_path, {}, positions={"ZZZ": 10.0})
    assert result.unpriced == ("ZZZ",)


def test_a_buy_against_a_short_is_a_cover(tmp_path):
    # Nothing here generates a short, but the broker can hold one. Keying the side off the
    # existing position is what keeps the vocabulary honest when it does.
    result = run(tmp_path, {}, positions={"AAA": -10.0},
                 acct=account(cash=101_000.0, equity=100_000.0))

    assert [(i.ticker, i.qty, i.side) for i in result.intents] == [
        ("AAA", 10.0, Side.BUY_TO_COVER)]


def test_dust_is_not_an_order(tmp_path):
    # A delta below the quantisation floor rounds to zero shares, and an intent refuses a
    # non-positive quantity — so it has to be dropped here rather than raising there.
    result = run(tmp_path, {"AAA": 1e-12}, positions={})
    assert result.intents == ()


def test_worthless_equity_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no NAV to size against"):
        run(tmp_path, {"AAA": 0.5}, acct=account(cash=0.0, equity=0.0))


def test_the_session_date_reaches_every_intent(tmp_path):
    result = run(tmp_path, {"AAA": 0.5, "BBB": 0.5})
    assert {i.session_date for i in result.intents} == {SESSION}
    # Deterministic ids are what make a resubmit a duplicate rather than a second position.
    assert len({i.client_order_id for i in result.intents}) == 2


def _record(intent: OrderIntent):
    from stock_agent.execution.store import OrderRecord
    return OrderRecord(intent=intent)
