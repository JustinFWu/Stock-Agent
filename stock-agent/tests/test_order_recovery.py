# The scenario the order log exists for, end to end: the process dies with an order in
# flight, comes back with no memory of it, and has to reach the truth without placing it
# again. The FakeBroker is what makes the crash happen on demand.

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.broker import BrokerError, OrderIntent, OrderState, Side
from src.execution.fake import FailPoint, FakeBroker, Fault
from src.execution.recovery import recover, submit_intent
from src.execution.store import LocalState, OrderStore

SESSION = pd.Timestamp("2026-09-15")
PRICES = {"AAA": 100.0, "BBB": 50.0}


def buy(ticker="AAA", qty=10.0):
    return OrderIntent(session_date=SESSION, ticker=ticker, qty=qty, side=Side.BUY)


def broker(**kwargs):
    kwargs.setdefault("prices", dict(PRICES))
    return FakeBroker(**kwargs)


def test_a_clean_submit_records_the_ack(tmp_path):
    store, venue = OrderStore(tmp_path / "orders.jsonl"), broker()
    record = submit_intent(store, venue, buy())

    assert record.state is LocalState.SUBMITTED
    assert record.known_to_broker
    assert venue.positions() == {"AAA": 10.0}


def test_the_intent_is_written_before_the_order_is_sent(tmp_path):
    # The ordering claim. The broker is made to die before accepting, so the only way a
    # record can exist afterwards is if it was written first.
    store = OrderStore(tmp_path / "orders.jsonl")
    venue = broker().arm(Fault("submit", FailPoint.BEFORE_ACCEPT))

    with pytest.raises(BrokerError):
        submit_intent(store, venue, buy())

    assert venue.accepted_ids == ()
    assert store.get(buy().client_order_id).state is LocalState.INTENDED


def test_an_unknown_outcome_records_no_outcome(tmp_path):
    # The broker took it and then the call failed. Writing "failed" here is how a filled
    # order becomes invisible, so the record stays INTENDED for recovery to settle.
    store = OrderStore(tmp_path / "orders.jsonl")
    venue = broker().arm(Fault("submit", FailPoint.AFTER_ACCEPT))

    with pytest.raises(BrokerError):
        submit_intent(store, venue, buy())

    record = store.get(buy().client_order_id)
    assert record.state is LocalState.INTENDED
    assert not record.known_to_broker
    assert venue.positions() == {"AAA": 10.0}   # it did land, and our side cannot tell


def test_a_crash_after_accept_is_settled_on_restart(tmp_path):
    # The headline. A new store over the same log is our side coming back with no memory;
    # the FakeBroker is the venue, which never forgot.
    path = tmp_path / "orders.jsonl"
    venue = broker().arm(Fault("submit", FailPoint.AFTER_ACCEPT))

    with pytest.raises(BrokerError):
        submit_intent(OrderStore(path), venue, buy())

    restarted = OrderStore(path)
    report = recover(restarted, venue)

    assert [r.client_order_id for r in report.resolved] == [buy().client_order_id]
    assert restarted.get(buy().client_order_id).state is LocalState.RESOLVED
    assert restarted.get(buy().client_order_id).filled_qty == 10.0
    assert venue.positions() == {"AAA": 10.0}
    assert report.is_clean


def test_a_settled_order_cannot_be_intended_again(tmp_path):
    # Second line of defence behind the deterministic id. Once recovery has resolved an
    # order, re-generating it means the reconciler ignored the position it created.
    path = tmp_path / "orders.jsonl"
    venue = broker()
    store = OrderStore(path)
    submit_intent(store, venue, buy())
    recover(store, venue)

    with pytest.raises(ValueError, match="already in the order log"):
        submit_intent(store, venue, buy())
    assert venue.positions() == {"AAA": 10.0}


def test_an_intent_the_broker_never_saw_is_abandoned(tmp_path):
    # Died before the submit landed. Nothing happened, and recovery is allowed to say so
    # only because there is no evidence the broker ever held it.
    path = tmp_path / "orders.jsonl"
    venue = broker().arm(Fault("submit", FailPoint.BEFORE_ACCEPT))

    with pytest.raises(BrokerError):
        submit_intent(OrderStore(path), venue, buy())

    report = recover(OrderStore(path), venue)
    assert [r.client_order_id for r in report.abandoned] == [buy().client_order_id]
    assert report.is_clean
    assert venue.positions() == {}


def test_a_still_working_order_blocks_its_name(tmp_path):
    # Not resolvable and not abandonable: the order can still move the book. The report
    # names it so the reconciler can leave that name alone rather than size against a
    # position that is still changing.
    store = OrderStore(tmp_path / "orders.jsonl")
    venue = broker(auto_fill=False)
    submit_intent(store, venue, buy("AAA"))

    report = recover(store, venue)
    assert report.blocked_tickers == frozenset({"AAA"})
    assert not report.is_clean
    assert store.get(buy("AAA").client_order_id).state is LocalState.SUBMITTED


def test_a_partial_fill_is_reported_as_working_not_resolved(tmp_path):
    store = OrderStore(tmp_path / "orders.jsonl")
    venue = broker(auto_fill=False)
    submit_intent(store, venue, buy("AAA", qty=10.0))
    venue.fill_open(qty=4.0)

    report = recover(store, venue)
    assert report.blocked_tickers == frozenset({"AAA"})
    assert store.get(buy("AAA").client_order_id).filled_qty == 4.0
    assert store.get(buy("AAA").client_order_id).broker_state is OrderState.PARTIALLY_FILLED


def test_recovery_is_also_the_fill_poll(tmp_path):
    # Same call at the end of a session as at the start. One implementation, so the two
    # cannot come to disagree about what a partial means.
    store = OrderStore(tmp_path / "orders.jsonl")
    venue = broker(auto_fill=False)
    submit_intent(store, venue, buy("AAA"))
    assert not recover(store, venue).is_clean

    venue.fill_open()
    report = recover(store, venue)
    assert report.is_clean
    assert [r.client_order_id for r in report.resolved] == [buy("AAA").client_order_id]


def test_an_acknowledged_order_the_broker_has_lost_is_refused(tmp_path):
    # A venue standing in for one that dropped an order it had acked. There is no honest
    # inference available here, so recovery stops rather than guessing it never filled.
    store = OrderStore(tmp_path / "orders.jsonl")
    submit_intent(store, broker(), buy())

    with pytest.raises(BrokerError, match="Refusing to guess"):
        recover(store, broker())


def test_a_duplicate_submit_asks_what_actually_happened(tmp_path):
    # The store is fresh, so nothing local stops the resubmit — the broker's rejection is
    # the only defence left, and the right response is to find out, not to assume.
    venue = broker()
    submit_intent(OrderStore(tmp_path / "first.jsonl"), venue, buy())

    second = OrderStore(tmp_path / "second.jsonl")
    record = submit_intent(second, venue, buy())

    assert record.state is LocalState.RESOLVED
    assert record.filled_qty == 10.0
    assert venue.positions() == {"AAA": 10.0}
    assert len(venue.accepted_ids) == 1


def test_recovery_leaves_nothing_open_across_several_names(tmp_path):
    path = tmp_path / "orders.jsonl"
    venue = broker().arm(Fault("submit", FailPoint.AFTER_ACCEPT, ticker="BBB"))
    store = OrderStore(path)

    submit_intent(store, venue, buy("AAA"))
    with pytest.raises(BrokerError):
        submit_intent(store, venue, buy("BBB"))

    report = recover(OrderStore(path), venue)
    assert {r.intent.ticker for r in report.resolved} == {"AAA", "BBB"}
    assert report.is_clean
    assert venue.positions() == {"AAA": 10.0, "BBB": 10.0}
