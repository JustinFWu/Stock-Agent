# The FakeBroker exists for one scenario a paper account will not reproduce on demand: the
# process dies between the broker accepting an order and our side recording that it did.
# Everything here is either that scenario or the arithmetic it rests on.

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.broker import (Broker, BrokerError, DuplicateOrderError, OrderIntent,
                                  OrderState, Side)
from src.execution.fake import FailPoint, FakeBroker, Fault

SESSION = pd.Timestamp("2026-09-15")
PRICES = {"AAA": 100.0, "BBB": 50.0}


def broker(**kwargs):
    kwargs.setdefault("prices", dict(PRICES))
    return FakeBroker(**kwargs)


def buy(ticker="AAA", qty=10.0):
    return OrderIntent(session_date=SESSION, ticker=ticker, qty=qty, side=Side.BUY)


def sell(ticker="AAA", qty=10.0):
    return OrderIntent(session_date=SESSION, ticker=ticker, qty=qty, side=Side.SELL)


def test_the_fake_satisfies_the_broker_protocol():
    # Cheap, and it is what keeps the fake honest as the Protocol grows: a method added to
    # Broker and forgotten here would otherwise surface first in the Alpaca adapter.
    assert isinstance(broker(), Broker)


def test_a_filled_buy_moves_shares_and_cash():
    b = broker()
    b.submit(buy(qty=10.0))

    assert b.positions() == {"AAA": 10.0}
    assert b.account().cash == 99_000.0
    assert b.account().equity == 100_000.0


def test_a_sell_that_closes_a_position_drops_it():
    b = broker(holdings={"AAA": 10.0})
    b.submit(sell(qty=10.0))
    assert b.positions() == {}


def test_a_resubmitted_intent_is_refused():
    # The core of idempotency. Same session, same intent, same id — the broker rejects it
    # rather than opening a second position.
    b = broker()
    b.submit(buy())

    with pytest.raises(DuplicateOrderError):
        b.submit(buy())
    assert b.positions() == {"AAA": 10.0}


def test_a_crash_after_accept_leaves_the_book_moved_and_the_caller_blind():
    # The window the whole design is built around: submit raises and the position is there
    # regardless. Anything that reads the exception as "it did not happen" is already wrong.
    b = broker().arm(Fault("submit", FailPoint.AFTER_ACCEPT))
    with pytest.raises(BrokerError):
        b.submit(buy())

    assert b.positions() == {"AAA": 10.0}
    assert b.order_status(buy().client_order_id).state is OrderState.FILLED


def test_a_crash_before_accept_leaves_nothing_behind():
    b = broker().arm(Fault("submit", FailPoint.BEFORE_ACCEPT))
    with pytest.raises(BrokerError):
        b.submit(buy())

    assert b.positions() == {}
    assert b.order_status(buy().client_order_id) is None
    assert b.accepted_ids == ()


def test_restarting_after_a_crash_does_not_double_the_position():
    # The headline. The process dies between submit and the write that records it, comes
    # back with no memory of having sent anything, and resubmits the same intent.
    b = broker().arm(Fault("submit", FailPoint.AFTER_ACCEPT))
    with pytest.raises(BrokerError):
        b.submit(buy())

    with pytest.raises(DuplicateOrderError):
        b.submit(buy())

    assert b.positions() == {"AAA": 10.0}
    assert len(b.accepted_ids) == 1
    assert b.account().cash == 99_000.0


def test_order_status_is_how_a_recovered_session_learns_what_landed():
    b = broker().arm(Fault("submit", FailPoint.AFTER_ACCEPT))
    with pytest.raises(BrokerError):
        b.submit(buy())

    status = b.order_status(buy().client_order_id)
    assert status.is_terminal
    assert status.filled_qty == 10.0
    assert status.avg_fill_price == 100.0


def test_an_unknown_order_id_is_none_rather_than_an_error():
    # "Never sent" and "sent, outcome unknown" are different answers and recovery branches
    # on exactly this, so the absent case must not arrive as an exception.
    assert broker().order_status("2026-09-15:AAA:000000000000") is None


def test_a_pending_order_is_not_terminal_until_it_fills():
    b = broker(auto_fill=False)
    b.submit(buy())

    status = b.order_status(buy().client_order_id)
    assert status.state is OrderState.PENDING
    assert not status.is_terminal


def test_a_partial_fill_reports_what_landed():
    b = broker(auto_fill=False)
    b.submit(buy(qty=10.0))
    b.fill_open(qty=4.0)

    status = b.order_status(buy(qty=10.0).client_order_id)
    assert status.state is OrderState.PARTIALLY_FILLED
    assert status.filled_qty == 4.0
    assert b.positions() == {"AAA": 4.0}


def test_two_partials_average_the_fill_price():
    b = broker(auto_fill=False)
    b.submit(buy(qty=10.0))
    b.fill_open(qty=5.0, price=100.0)
    b.fill_open(qty=5.0, price=110.0)

    status = b.order_status(buy(qty=10.0).client_order_id)
    assert status.state is OrderState.FILLED
    assert status.avg_fill_price == 105.0


def test_cancel_stops_a_working_order():
    # How the kill switch pulls orders instead of merely declining to add new ones.
    b = broker(auto_fill=False)
    b.submit(buy())
    b.cancel(buy().client_order_id)

    assert b.order_status(buy().client_order_id).state is OrderState.CANCELED
    assert b.positions() == {}


def test_cancelling_a_filled_order_does_not_rewrite_it():
    b = broker()
    b.submit(buy())
    b.cancel(buy().client_order_id)
    assert b.order_status(buy().client_order_id).state is OrderState.FILLED


def test_fills_can_be_read_from_a_point_in_time():
    b = broker(now="2026-09-15")
    b.submit(buy("AAA"))
    b.now = pd.Timestamp("2026-09-16")
    b.submit(buy("BBB"))

    assert [f.ticker for f in b.fills()] == ["AAA", "BBB"]
    assert [f.ticker for f in b.fills(since="2026-09-16")] == ["BBB"]


def test_the_fake_shorts_rather_than_protecting_us():
    # Faithful to a margin account on purpose. If the fake refused, a passing veto test
    # would prove nothing about the veto.
    b = broker(holdings={"AAA": 5.0})
    b.submit(sell(qty=10.0))
    assert b.positions() == {"AAA": -5.0}


def test_a_missing_price_is_refused_before_the_order_is_accepted():
    b = broker()
    with pytest.raises(BrokerError, match="no price"):
        b.submit(buy("ZZZ"))
    assert b.accepted_ids == ()


def test_an_armed_fault_fires_once():
    # Recovery has to run against a broker that is working again, or the test measures the
    # fault rather than the recovery.
    b = broker().arm(Fault("positions"))
    with pytest.raises(BrokerError):
        b.positions()
    assert b.positions() == {}


def test_a_fault_can_be_aimed_at_one_name():
    b = broker().arm(Fault("submit", FailPoint.AFTER_ACCEPT, ticker="BBB"))
    b.submit(buy("AAA"))

    with pytest.raises(BrokerError):
        b.submit(buy("BBB"))
    assert b.positions() == {"AAA": 10.0, "BBB": 10.0}
