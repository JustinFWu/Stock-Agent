# The log is the only thing that remembers an order was sent, so the properties worth
# pinning are the ones a crash attacks: that a write lands before the thing it describes,
# that a reopened log says exactly what the live one did, and that a torn tail costs one
# line rather than the file.

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.broker import OrderAck, OrderIntent, OrderState, OrderStatus, Side
from src.execution.store import LocalState, OrderStore

SESSION = pd.Timestamp("2026-09-15")


def buy(ticker="AAA", qty=10.0):
    return OrderIntent(session_date=SESSION, ticker=ticker, qty=qty, side=Side.BUY)


def ack(intent, broker_order_id="fake-000001"):
    return OrderAck(client_order_id=intent.client_order_id,
                    broker_order_id=broker_order_id, accepted_at=SESSION)


def status(intent, state=OrderState.FILLED, filled_qty=10.0, avg_fill_price=100.0):
    return OrderStatus(client_order_id=intent.client_order_id, state=state,
                       filled_qty=filled_qty, avg_fill_price=avg_fill_price)


def test_an_intent_is_on_disk_the_moment_it_is_recorded(tmp_path):
    # The write-ahead claim, checked against the file rather than the object: an intent
    # that only exists in memory is an intent a crash erases.
    path = tmp_path / "orders.jsonl"
    OrderStore(path).record_intent(buy())

    written = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert written["event"] == "intended"
    assert written["ticker"] == "AAA"
    assert written["qty"] == 10.0
    assert written["side"] == "buy"


def test_a_fresh_intent_starts_unresolved(tmp_path):
    store = OrderStore(tmp_path / "orders.jsonl")
    record = store.record_intent(buy())

    assert record.state is LocalState.INTENDED
    assert not record.known_to_broker
    assert store.unresolved() == [record]


def test_an_ack_marks_the_order_known_to_the_broker(tmp_path):
    store = OrderStore(tmp_path / "orders.jsonl")
    store.record_intent(buy())
    record = store.record_ack(ack(buy()))

    assert record.state is LocalState.SUBMITTED
    assert record.known_to_broker
    assert record.broker_order_id == "fake-000001"


def test_a_terminal_observation_resolves_the_record(tmp_path):
    store = OrderStore(tmp_path / "orders.jsonl")
    store.record_intent(buy())
    record = store.record_observation(status(buy()))

    assert record.state is LocalState.RESOLVED
    assert record.filled_qty == 10.0
    assert record.avg_fill_price == 100.0
    assert store.unresolved() == []


def test_a_partial_observation_leaves_the_record_open(tmp_path):
    # A partial is evidence the broker has it and evidence that it is not done. Both
    # halves matter: the first stops recovery writing it off, the second stops a session
    # sizing against a position that is still moving.
    store = OrderStore(tmp_path / "orders.jsonl")
    store.record_intent(buy())
    record = store.record_observation(status(buy(), OrderState.PARTIALLY_FILLED, 4.0))

    assert record.state is LocalState.SUBMITTED
    assert record.known_to_broker
    assert store.unresolved() == [record]


def test_an_abandoned_record_is_closed_with_its_reason(tmp_path):
    store = OrderStore(tmp_path / "orders.jsonl")
    store.record_intent(buy())
    record = store.record_abandoned(buy().client_order_id, "the broker never had it")

    assert record.state is LocalState.ABANDONED
    assert record.note == "the broker never had it"
    assert store.unresolved() == []


def test_the_replayed_view_matches_the_live_one(tmp_path):
    # One fold serves both paths. Two implementations is how a log that replays correctly
    # in a test replays differently after the crash it was written for.
    path = tmp_path / "orders.jsonl"
    store = OrderStore(path)
    store.record_intent(buy("AAA"))
    store.record_ack(ack(buy("AAA")))
    store.record_observation(status(buy("AAA")))
    store.record_intent(buy("BBB"))

    assert OrderStore(path).records() == store.records()


def test_the_log_is_only_ever_appended_to(tmp_path):
    path = tmp_path / "orders.jsonl"
    store = OrderStore(path)
    store.record_intent(buy())
    after_intent = path.read_text(encoding="utf-8")

    store.record_ack(ack(buy()))
    assert path.read_text(encoding="utf-8").startswith(after_intent)


def test_intending_the_same_order_twice_is_refused(tmp_path):
    # A repeat under an id the log already knows means recovery did not resolve the first
    # one. Overwriting the record is how the evidence of that disappears.
    store = OrderStore(tmp_path / "orders.jsonl")
    store.record_intent(buy())

    with pytest.raises(ValueError, match="already in the order log"):
        store.record_intent(buy())


def test_an_event_with_no_intent_before_it_is_refused(tmp_path):
    store = OrderStore(tmp_path / "orders.jsonl")
    with pytest.raises(ValueError, match="out of order"):
        store.record_ack(ack(buy()))


def test_a_torn_final_line_costs_one_line_not_the_file(tmp_path):
    # A crash between the write and the fsync. Appending after the fragment would splice
    # the next event onto it and destroy a line that parsed fine before.
    path = tmp_path / "orders.jsonl"
    store = OrderStore(path)
    store.record_intent(buy())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"event": "submi')

    reopened = OrderStore(path)
    assert list(reopened.records()) == [buy().client_order_id]

    reopened.record_ack(ack(buy()))
    assert OrderStore(path).get(buy().client_order_id).state is LocalState.SUBMITTED


def test_corruption_in_the_middle_of_the_log_is_refused(tmp_path):
    # Distinct from a torn tail on purpose. A bad line with good lines after it is not a
    # crash artefact, and quietly skipping it would drop an order from the record.
    path = tmp_path / "orders.jsonl"
    store = OrderStore(path)
    store.record_intent(buy("AAA"))
    store.record_intent(buy("BBB"))

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = "{ not json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corruption"):
        OrderStore(path)


def test_a_missing_log_is_an_empty_one(tmp_path):
    # First run of a new deployment, and it must not look like a failure.
    store = OrderStore(tmp_path / "nested" / "orders.jsonl")
    assert store.records() == {}
    assert store.unresolved() == []


def test_a_log_whose_ids_no_longer_replay_is_refused(tmp_path):
    # The id is derived, so a change to the digest recipe re-keys every log written before
    # it and the next session resubmits the lot. Simulated here by editing a stored id.
    path = tmp_path / "orders.jsonl"
    OrderStore(path).record_intent(buy())
    path.write_text(path.read_text(encoding="utf-8").replace(
        buy().client_order_id, "2026-09-15:AAA:000000000000"), encoding="utf-8")

    with pytest.raises(ValueError, match="recipe has changed"):
        OrderStore(path)
