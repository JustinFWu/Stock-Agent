# recover -> reconcile -> tripwire -> veto -> submit, threaded end to end.
#
# There is no session runner yet, so nothing in `src/` performs this sequence. That is
# exactly why it is pinned here: the four pieces were built separately and the claim that
# they compose — in this order, with the broker as the source of truth throughout — is the
# claim worth a test rather than a comment.

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.data.universe import UniverseSpec
from src.execution.broker import BrokerError
from src.execution.fake import FailPoint, FakeBroker, Fault
from src.execution.killswitch import KillSwitch, TripReason
from src.execution.recovery import recover, submit_intent
from src.execution.reconcile import reconcile
from src.execution.store import OrderStore
from src.execution.veto import veto

SESSION = pd.Timestamp("2026-09-18")
PRICES = {"AAA": 100.0, "BBB": 50.0}
MARKS = pd.Series(PRICES, dtype=float)
TARGET = pd.Series({"AAA": 0.05, "BBB": 0.05}, dtype=float)

UNIVERSE = UniverseSpec(id="test", tickers=("AAA", "BBB"), point_in_time=False,
                        caveats=(), sectors={"AAA": "XLK", "BBB": "XLF"})


def session(store: OrderStore, switch: KillSwitch, venue: FakeBroker, target=TARGET):
    # The sketch's ordering, and every step of it takes the broker's answer rather than
    # ours. Recovery first, because a session that generates before settling the last one
    # sizes against a position that is still moving.
    report = recover(store, venue)
    positions, account = venue.positions(), venue.account()

    plan = reconcile(SESSION, target, positions=positions, account=account,
                     prices=MARKS, marks=MARKS, store=store, recovery=report,
                     no_trade_band=0.0)

    breaches = switch.check(equity=account.equity, positions=positions,
                            nav_disagreement=plan.nav_disagreement,
                            unpriced_positions=plan.unpriced)

    decision = veto(plan.intents, positions=positions, account=account, marks=MARKS,
                    universe=UNIVERSE, kill_switch_tripped=switch.is_tripped)

    for intent in decision.allowed:
        submit_intent(store, venue, intent)
    return plan, breaches, decision


def setup(tmp_path, **broker_kwargs):
    broker_kwargs.setdefault("prices", dict(PRICES))
    return (OrderStore(tmp_path / "orders.jsonl"),
            KillSwitch(tmp_path / "kill_switch.json"),
            FakeBroker(**broker_kwargs))


def test_a_clean_session_reaches_the_target(tmp_path):
    store, switch, venue = setup(tmp_path)
    plan, breaches, decision = session(store, switch, venue)

    assert breaches == () and decision.is_clean
    assert venue.positions() == {"AAA": 50.0, "BBB": 100.0}
    assert plan.nav_disagreement == 0.0


def test_a_second_session_on_an_unchanged_target_does_nothing(tmp_path):
    # Idempotence at the level that matters: reconciling a book that already matches has
    # nothing to send, so an unattended loop does not churn.
    store, switch, venue = setup(tmp_path)
    session(store, switch, venue)
    plan, _, decision = session(store, switch, venue)

    assert plan.intents == ()
    assert decision.allowed == ()
    assert venue.positions() == {"AAA": 50.0, "BBB": 100.0}


def test_a_crash_mid_rebalance_does_not_double_the_position(tmp_path):
    # The roadmap names this as the real risk, ahead of signal decay. The broker takes the
    # BBB order, moves the book, and then the call fails — so our side knows nothing about
    # a position that exists.
    store, switch, venue = setup(tmp_path)
    venue.arm(Fault("submit", FailPoint.AFTER_ACCEPT, ticker="BBB"))

    with pytest.raises(BrokerError):
        session(store, switch, venue)

    assert venue.positions() == {"AAA": 50.0, "BBB": 100.0}

    # The restart: a new process against the same log and the same venue.
    restarted = OrderStore(tmp_path / "orders.jsonl")
    plan, _, decision = session(restarted, KillSwitch(tmp_path / "kill_switch.json"), venue)

    assert plan.intents == ()          # recovery settled BBB before anything was generated
    assert decision.allowed == ()
    assert venue.positions() == {"AAA": 50.0, "BBB": 100.0}
    assert len(venue.accepted_ids) == 2


def test_a_working_order_holds_its_name_back_for_a_session(tmp_path):
    # With auto-fill off the BBB order is accepted and stays open. The name is still moving,
    # so the next session must not size against it.
    store, switch, venue = setup(tmp_path, auto_fill=False)
    session(store, switch, venue)

    plan, _, _ = session(store, switch, venue)
    assert set(plan.deferred) == {"AAA", "BBB"}
    assert plan.intents == ()


def test_a_tripped_switch_stops_the_session_even_with_a_clean_plan(tmp_path):
    # The three parts doing their separate jobs: the tripwire notices, the state remembers,
    # and the veto is the only one that actually refuses.
    store, switch, venue = setup(tmp_path)
    switch.trip(TripReason.DRAWDOWN, "set by hand for this test")

    plan, _, decision = session(store, switch, venue)

    assert len(plan.intents) == 2      # the reconciler still does its arithmetic
    assert decision.allowed == ()      # and the veto is what stops it
    assert venue.positions() == {}


def test_the_switch_stays_tripped_across_the_restart_that_follows_it(tmp_path):
    store, switch, venue = setup(tmp_path)
    switch.trip(TripReason.NAV_MISMATCH, "set by hand for this test")
    session(store, switch, venue)

    reopened = KillSwitch(tmp_path / "kill_switch.json")
    _, _, decision = session(OrderStore(tmp_path / "orders.jsonl"), reopened, venue)

    assert reopened.is_tripped
    assert decision.allowed == ()
    assert venue.positions() == {}


def test_an_unexpected_short_trips_the_switch_and_the_veto_refuses_to_deepen_it(tmp_path):
    # Nothing here generates a short, so one in the book is a fact about the system. The
    # session that finds it stops, and the cover order it would have sent is not a
    # workaround — it is simply never reached, because the switch is already down.
    store, switch, venue = setup(tmp_path, holdings={"BBB": -40.0})
    plan, breaches, decision = session(store, switch, venue)

    assert [b.reason for b in breaches] == [TripReason.UNEXPECTED_POSITION]
    assert switch.is_tripped
    assert decision.allowed == ()
    assert any(i.ticker == "BBB" for i in plan.intents)
