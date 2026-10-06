# The venue adapter. What is tested here is the *mapping* — Alpaca's words to the
# vocabulary in broker.py — because that is where a thin adapter can be wrong. The HTTP is
# not tested and should not be: these tests replace the one request method, so nothing here
# touches a network, and a test that mocked urllib would be testing urllib.
#
# The cases that matter are the ones where getting it wrong is expensive: a duplicate
# client_order_id read as a generic failure (a crash-recovered session would submit twice),
# a timeout read as a rejection (an order left live at the venue with no record), and an
# unrecognised status read as terminal (a session sizing against a position that may not
# exist).

import json
import urllib.error

import pandas as pd
import pytest

from stock_agent.execution.alpaca import AlpacaBroker, _is_duplicate, _venue_side
from stock_agent.execution.broker import (
    Broker,
    BrokerError,
    DuplicateOrderError,
    OrderIntent,
    OrderState,
    Side,
)

SESSION = pd.Timestamp("2026-09-18")


class StubbedBroker(AlpacaBroker):
    # Replaces the transport and records what was asked for. Subclassing rather than
    # monkeypatching so the type is still an AlpacaBroker and still satisfies the Protocol.
    def __init__(self, responses=None, **kwargs):
        kwargs.setdefault("api_key", "key")
        kwargs.setdefault("secret_key", "secret")
        super().__init__(**kwargs)
        self.responses = responses or {}
        self.calls = []

    def _request(self, method, path, body=None, query=None, none_on_404=False):
        self.calls.append((method, path, body, query))
        value = self.responses.get(path, {})
        if isinstance(value, Exception):
            raise value
        if value is None and none_on_404:
            return None
        return value


def http_error(code, payload):
    # urllib's HTTPError reads its body once, so it needs a real file-like object.
    import io
    return urllib.error.HTTPError(
        url="https://paper-api.alpaca.markets/v2/orders", code=code, msg="err",
        hdrs=None, fp=io.BytesIO(json.dumps(payload).encode()))


def intent(ticker="AAA", qty=10.0, side=Side.BUY):
    return OrderIntent(session_date=SESSION, ticker=ticker, qty=qty, side=side)


# --- construction and the safety rails ------------------------------------------------

def test_it_satisfies_the_broker_protocol():
    # `runtime_checkable` only checks the methods exist, which is exactly the thing that
    # breaks when the Protocol grows one and an adapter is not updated.
    assert isinstance(StubbedBroker(), Broker)


def test_paper_is_the_default_endpoint():
    assert "paper-api" in StubbedBroker().base_url


def test_the_live_endpoint_needs_more_than_flipping_the_flag():
    with pytest.raises(ValueError, match="i_understand_this_is_real_money"):
        AlpacaBroker(paper=False, api_key="k", secret_key="s")

    armed = AlpacaBroker(paper=False, api_key="k", secret_key="s",
                         i_understand_this_is_real_money=True)
    assert armed.base_url == "https://api.alpaca.markets"


def test_missing_keys_are_refused_at_construction(monkeypatch):
    # Rather than at the first request, which would be mid-session.
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        AlpacaBroker()


def test_keys_come_from_the_environment_when_not_passed(monkeypatch):
    # Read at construction, not at import, so loading a .env afterwards still works.
    monkeypatch.setenv("ALPACA_API_KEY", "from-env")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "also-from-env")
    assert AlpacaBroker().api_key == "from-env"


# --- positions and the account --------------------------------------------------------

def test_positions_are_signed_and_zeroes_are_dropped():
    venue = StubbedBroker({"/v2/positions": [
        {"symbol": "AAA", "qty": "100"},
        {"symbol": "BBB", "qty": "-50"},     # the venue's own sign for a short
        {"symbol": "CCC", "qty": "0"},
    ]})
    assert venue.positions() == {"AAA": 100.0, "BBB": -50.0}


def test_the_account_carries_the_venues_own_equity_and_buying_power():
    venue = StubbedBroker({"/v2/account": {
        "cash": "1000.50", "equity": "25000.75", "buying_power": "50001.50",
        "status": "ACTIVE"}})
    account = venue.account()

    assert account.cash == pytest.approx(1000.50)
    assert account.equity == pytest.approx(25000.75)
    assert account.buying_power == pytest.approx(50001.50)


def test_a_blocked_account_refuses_before_anything_is_submitted():
    # The venue's own halt. Proceeding would turn one clear refusal into a sequence of
    # per-order rejections.
    venue = StubbedBroker({"/v2/account": {
        "cash": "0", "equity": "0", "buying_power": "0",
        "status": "ACCOUNT_CLOSED", "trading_blocked": True}})
    with pytest.raises(BrokerError, match="blocked"):
        venue.account()


# --- submitting -----------------------------------------------------------------------

def test_submit_sends_the_computed_client_order_id_as_the_idempotency_key():
    venue = StubbedBroker({"/v2/orders": {"id": "venue-1", "submitted_at":
                                          "2026-09-18T20:00:00Z"}})
    order = intent()
    ack = venue.submit(order)

    _, path, body, _ = venue.calls[-1]
    assert path == "/v2/orders"
    assert body["client_order_id"] == order.client_order_id
    assert body["symbol"] == "AAA" and body["side"] == "buy"
    assert body["type"] == "market" and body["time_in_force"] == "day"
    assert ack.broker_order_id == "venue-1"
    assert ack.client_order_id == order.client_order_id


def test_the_four_sides_collapse_to_the_two_the_venue_takes():
    # Alpaca infers shorting and covering from the position, so our vocabulary narrows
    # here and nowhere else. This is the branch the Phase 4 sketch predicted, and the
    # reason keeping `Side` wide cost nothing.
    assert _venue_side(Side.BUY) == "buy"
    assert _venue_side(Side.BUY_TO_COVER) == "buy"
    assert _venue_side(Side.SELL) == "sell"
    assert _venue_side(Side.SELL_SHORT) == "sell"


def raising_transport(monkeypatch, exc):
    # These three cases live inside the real `_request`, which is what translates a
    # transport failure into our vocabulary — so they have to be stubbed a level below the
    # others, at urlopen, or the code under test is the part being replaced.
    def boom(*args, **kwargs):
        raise exc
    monkeypatch.setattr("stock_agent.execution.alpaca.urllib.request.urlopen", boom)
    return AlpacaBroker(api_key="key", secret_key="secret")


def test_a_duplicate_client_order_id_is_its_own_error(monkeypatch):
    # The success path for idempotency, and it must not be read as a generic failure: a
    # recovered session that treats this as "unknown" asks the venue, which is right, but
    # one that treated it as "failed" would submit again.
    venue = raising_transport(monkeypatch, http_error(
        422, {"code": 40010001, "message": "client_order_id must be unique"}))

    with pytest.raises(DuplicateOrderError, match="already holds"):
        venue.submit(intent())


def test_duplicate_detection_reads_the_message_as_well_as_the_code():
    # The numeric code has moved between API revisions; the message has not.
    assert _is_duplicate(422, "client_order_id must be unique")
    assert _is_duplicate(403, "a duplicate client_order_id was supplied")
    assert not _is_duplicate(422, "insufficient buying power")
    assert not _is_duplicate(500, "client_order_id must be unique")


def test_an_ordinary_refusal_is_a_broker_error_with_the_venues_words(monkeypatch):
    venue = raising_transport(monkeypatch,
                              http_error(403, {"message": "insufficient buying power"}))

    with pytest.raises(BrokerError, match="insufficient buying power") as caught:
        venue.submit(intent())
    assert not isinstance(caught.value, DuplicateOrderError)


@pytest.mark.parametrize("failure", [
    urllib.error.URLError("timed out"),
    TimeoutError("timed out"),
    OSError("connection reset"),
])
def test_silence_from_the_venue_is_an_unknown_outcome_and_says_so(monkeypatch, failure):
    # The case the entire write-ahead design exists for. The order may be live at the
    # venue, so this must not read as a failure to submit — `submit_intent` records
    # nothing and re-raises, and the next session's `order_status` settles it.
    venue = raising_transport(monkeypatch, failure)

    with pytest.raises(BrokerError, match="outcome is unknown"):
        venue.submit(intent())


# --- order status ---------------------------------------------------------------------

@pytest.mark.parametrize("venue_status,expected", [
    ("filled", OrderState.FILLED),
    ("partially_filled", OrderState.PARTIALLY_FILLED),
    ("new", OrderState.PENDING),
    ("accepted", OrderState.PENDING),
    ("pending_new", OrderState.PENDING),
    ("canceled", OrderState.CANCELED),
    ("expired", OrderState.CANCELED),
    ("done_for_day", OrderState.CANCELED),
    ("rejected", OrderState.REJECTED),
    ("suspended", OrderState.REJECTED),
])
def test_the_status_table_maps_every_lifecycle_word_we_know(venue_status, expected):
    venue = StubbedBroker({"/v2/orders:by_client_order_id": {
        "status": venue_status, "filled_qty": "0", "filled_avg_price": None}})
    assert venue.order_status("some-id").state is expected


def test_a_status_the_table_does_not_know_is_pending_and_alert_worthy():
    # Not terminal, so recovery keeps asking and the name stays deferred — the
    # conservative reading. Mapping it to FILLED would size against a position that may
    # not exist; raising would halt trading because a venue added a word.
    venue = StubbedBroker({"/v2/orders:by_client_order_id": {
        "status": "quantum_superposition", "filled_qty": "0"}})

    status = venue.order_status("some-id")

    assert status.state is OrderState.PENDING
    assert not status.is_terminal
    drained = venue.drain_unknown_statuses()
    assert len(drained) == 1 and "quantum_superposition" in drained[0]
    assert venue.drain_unknown_statuses() == ()      # drained, not repeated


def test_a_partial_fill_carries_its_quantity_and_average_price():
    venue = StubbedBroker({"/v2/orders:by_client_order_id": {
        "status": "partially_filled", "filled_qty": "37.5",
        "filled_avg_price": "101.25"}})
    status = venue.order_status("some-id")

    assert status.filled_qty == pytest.approx(37.5)
    assert status.avg_fill_price == pytest.approx(101.25)
    assert not status.is_terminal      # a partial can still complete


def test_an_order_the_venue_never_heard_of_is_none_not_an_error():
    # The one honest "no". Recovery reads None as "the broker never had it", which is only
    # safe because there is no evidence it ever did.
    venue = StubbedBroker({"/v2/orders:by_client_order_id": None})
    assert venue.order_status("never-sent") is None


def test_an_empty_avg_price_does_not_become_zero():
    # A zero fill price would value a position at nothing. Absent is not zero.
    venue = StubbedBroker({"/v2/orders:by_client_order_id": {
        "status": "new", "filled_qty": "0", "filled_avg_price": ""}})
    assert venue.order_status("some-id").avg_fill_price is None


# --- cancel and fills -----------------------------------------------------------------

def test_cancel_looks_the_order_up_then_deletes_it_by_the_venues_id():
    venue = StubbedBroker({"/v2/orders:by_client_order_id": {"id": "venue-9"}})
    venue.cancel("ours-1")

    paths = [path for _, path, _, _ in venue.calls]
    assert paths == ["/v2/orders:by_client_order_id", "/v2/orders/venue-9"]
    assert venue.calls[-1][0] == "DELETE"


def test_cancelling_an_order_the_venue_has_forgotten_is_not_an_error():
    # Cancel is what the kill switch calls to pull working orders, and "already gone" is
    # the outcome it wanted.
    venue = StubbedBroker({"/v2/orders:by_client_order_id": None})
    venue.cancel("ours-1")
    assert len(venue.calls) == 1          # nothing to delete


def test_fills_map_to_observations_rather_than_to_our_cost_estimates():
    venue = StubbedBroker({"/v2/account/activities": [
        {"order_id": "venue-1", "client_order_id": "ours-1", "symbol": "AAA",
         "qty": "10", "side": "buy", "price": "100.25",
         "transaction_time": "2026-09-18T14:30:00Z"},
        {"order_id": "venue-2", "client_order_id": "ours-2", "symbol": "BBB",
         "qty": "-5", "side": "sell", "price": "50.10",
         "transaction_time": "2026-09-18T14:31:00Z"},
    ]})
    fills = venue.fills()

    assert [f.ticker for f in fills] == ["AAA", "BBB"]
    assert [f.side for f in fills] == [Side.BUY, Side.SELL]
    # Quantity is positive and direction lives in the side, as everywhere else.
    assert all(f.qty > 0 for f in fills)
    assert fills[1].signed_qty == pytest.approx(-5.0)
    assert fills[0].fill_price == pytest.approx(100.25)


def test_fills_since_passes_the_cursor_through():
    venue = StubbedBroker({"/v2/account/activities": []})
    venue.fills(since=pd.Timestamp("2026-09-18T00:00:00Z"))

    _, _, _, query = venue.calls[-1]
    assert query["activity_types"] == "FILL"
    assert query["after"].startswith("2026-09-18")


def test_an_activity_row_with_no_order_is_skipped():
    # The activities feed carries dividends and journals too; only fills belong here.
    venue = StubbedBroker({"/v2/account/activities": [
        {"symbol": "AAA", "qty": "1", "price": "1", "side": "buy"},   # no order_id
    ]})
    assert venue.fills() == []
