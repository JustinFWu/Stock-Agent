# The client_order_id is the only thing standing between a crashed session and a doubled
# position, so what does and does not move it is the contract worth pinning down. Float
# noise must not change it; a different size must.

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.broker import QTY_PRECISION, OrderIntent, OrderState, OrderStatus, Side

SESSION = pd.Timestamp("2026-09-15")


def intent(ticker="AAA", qty=100.0, side=Side.BUY, session=SESSION):
    return OrderIntent(session_date=session, ticker=ticker, qty=qty, side=side)


def test_the_same_intent_produces_the_same_id():
    assert intent().client_order_id == intent().client_order_id


def test_float_noise_does_not_change_the_id():
    # Two summation orders reaching the same size have to deduplicate, or the id protects
    # nothing — the resubmit after a crash is precisely the recomputed number.
    assert intent(qty=100.0).client_order_id == intent(qty=100.0 + 1e-12).client_order_id


def test_a_different_size_changes_the_id():
    assert intent(qty=100.0).client_order_id != intent(qty=101.0).client_order_id


def test_a_different_side_changes_the_id():
    assert intent(side=Side.BUY).client_order_id != intent(side=Side.SELL).client_order_id


def test_a_different_session_changes_the_id():
    # The same trade on two days is two trades. The date stays in clear text so a human
    # reading the write-ahead log can tell which session an order belonged to.
    other = intent(session="2026-09-16")
    assert intent().client_order_id != other.client_order_id
    assert other.client_order_id.startswith("2026-09-16:AAA:")


def test_the_session_date_is_normalised():
    # A session stamped with a wall-clock time would produce a fresh id on every run and
    # the deduplication would never once fire.
    assert intent(session="2026-09-15 15:47:03").client_order_id == intent().client_order_id


@pytest.mark.parametrize("qty", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_qty_is_refused(qty):
    # NaN is the dangerous one: it passes `> 0` by failing the comparison rather than by
    # satisfying it, so the obvious check lets it through and every downstream sign with it.
    with pytest.raises(ValueError, match="positive"):
        intent(qty=qty)


def test_an_intent_needs_a_ticker():
    with pytest.raises(ValueError, match="ticker"):
        intent(ticker="")


def test_qty_is_quantised_on_construction():
    # Quantised in place, not just inside the digest, so the number that is hashed and the
    # number that is sent cannot drift apart.
    assert intent(qty=1 / 3).qty == round(1 / 3, QTY_PRECISION)


def test_direction_lives_in_side_not_in_the_sign_of_qty():
    assert intent(side=Side.BUY).signed_qty == 100.0
    assert intent(side=Side.SELL).signed_qty == -100.0
    assert intent(side=Side.SELL_SHORT).signed_qty == -100.0
    assert intent(side=Side.BUY_TO_COVER).signed_qty == 100.0


def test_only_sell_short_opens_a_short():
    assert Side.SELL_SHORT.opens_short
    assert not Side.BUY_TO_COVER.opens_short    # closes exposure, does not open it
    assert not Side.SELL.opens_short


def test_a_partial_fill_is_not_terminal():
    # Treating a partial as settled is how the next session sizes against a position that
    # is still growing underneath it.
    assert not OrderStatus("id", OrderState.PARTIALLY_FILLED, filled_qty=40.0).is_terminal
    assert not OrderStatus("id", OrderState.PENDING).is_terminal

    assert OrderStatus("id", OrderState.FILLED, 100.0).is_terminal
    assert OrderStatus("id", OrderState.CANCELED).is_terminal
    assert OrderStatus("id", OrderState.REJECTED).is_terminal
