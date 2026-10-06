# The session runner, end to end. `test_execution_chain.py` already proves the pieces
# compose when threaded by hand; what is tested here is the thing in `stock_agent/` that
# does the threading — the twelve steps in order, what it records, and what it refuses.
#
# The crash test is the one that matters. The roadmap names a crash halfway through a
# rebalance as the real risk ahead of signal decay, and it is the one thing a paper account
# will not reproduce on demand.

import json

import numpy as np
import pandas as pd
import pytest

from stock_agent.data.universe import UniverseSpec
from stock_agent.execution.fake import FailPoint, FakeBroker, Fault
from stock_agent.execution.killswitch import KillSwitch, TripReason
from stock_agent.execution.session import SessionOutcome, Severity, run_session
from stock_agent.execution.store import OrderStore
from stock_agent.strategy.weights import EqualWeightStrategy
from tests.conftest import make_panel

SESSION = pd.Timestamp("2026-09-18")
TODAY = pd.Timestamp("2026-09-19")

# Ten names in five sectors, which is not arbitrary. `MAX_WEIGHT` is 10%, so a two-name
# universe would have the per-name cap decide the whole portfolio — equal weight asks for
# 50% each, gets 10%, and the book sits 80% in cash. At ten names equal weight lands exactly
# on the cap, the book is fully invested, and a halving of prices is a real 50% drawdown
# rather than a 10% one. Five sectors of two keeps each group at 20% against a 30% cap.
TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ")
SECTORS = {t: "X%d" % (i // 2) for i, t in enumerate(TICKERS)}

UNIVERSE = UniverseSpec(id="test", tickers=TICKERS, point_in_time=False,
                        caveats=(), sectors=SECTORS)


def panel_for(tickers=TICKERS, price=100.0, days=400):
    # Flat bars, long enough to clear MIN_HISTORY_DAYS so the names are tradable. Flat
    # means every share count is computable by hand.
    dates = pd.bdate_range(end=SESSION, periods=days)
    prices = pd.DataFrame(price, index=dates, columns=list(tickers))
    return make_panel(prices)


def setup(tmp_path, **broker_kwargs):
    broker_kwargs.setdefault("prices", dict.fromkeys(TICKERS, 100.0))
    return (OrderStore(tmp_path / "orders.jsonl"),
            KillSwitch(tmp_path / "kill_switch.json"),
            FakeBroker(**broker_kwargs))


def run(tmp_path, store, switch, venue, *, submit=True, panel=None, **kwargs):
    return run_session(
        broker=venue, store=store, kill_switch=switch,
        strategy=EqualWeightStrategy(), universe=UNIVERSE,
        panel=panel if panel is not None else panel_for(),
        as_of=SESSION, submit=submit, no_trade_band=0.0,
        today=TODAY, session_log=tmp_path / "sessions.jsonl", **kwargs)


# --- the ordinary path ---------------------------------------------------------------

def test_a_clean_session_reaches_the_target_and_says_so(tmp_path):
    store, switch, venue = setup(tmp_path)
    report = run(tmp_path, store, switch, venue)

    assert report.outcome is SessionOutcome.COMPLETED
    assert report.is_clean, report.describe()
    # 100k equity, equal weight over ten names at 100.0 => 10% each => 100 shares each.
    assert venue.positions() == dict.fromkeys(TICKERS, 100.0)
    assert len(report.sent) == len(TICKERS)
    assert report.withheld == ()
    assert report.drift == ()


def test_a_second_session_on_an_unchanged_book_sends_nothing(tmp_path):
    # What an unattended loop does on most days. The gate is thirty sessions, and most of
    # them have to be no-ops or the thing churns commission for nothing.
    store, switch, venue = setup(tmp_path)
    run(tmp_path, store, switch, venue)
    report = run(tmp_path, store, switch, venue)

    assert report.is_clean
    assert report.plan is not None and report.plan.intents == ()
    assert report.sent == ()


def test_a_dry_run_plans_everything_and_sends_nothing(tmp_path):
    # The default, and the reason it is the default: every part of a session that can be
    # wrong is wrong before `submit` is reached.
    store, switch, venue = setup(tmp_path)
    report = run(tmp_path, store, switch, venue, submit=False)

    assert report.outcome is SessionOutcome.COMPLETED
    assert report.plan is not None and len(report.plan.intents) == len(TICKERS)
    assert report.sent == ()
    assert len(report.withheld) == len(TICKERS)
    assert venue.positions() == {}          # nothing reached the venue
    assert store.unresolved() == []          # and nothing was written ahead


# --- the crash ------------------------------------------------------------------------

def test_a_crash_between_the_send_and_the_record_does_not_double_the_position(tmp_path):
    # The venue takes the order and moves the book, then the call fails — so our side has
    # no memory of a position that exists. A restart must settle it rather than re-place it.
    store, switch, venue = setup(tmp_path)
    venue.arm(Fault("submit", FailPoint.AFTER_ACCEPT, ticker="BBB"))

    first = run(tmp_path, store, switch, venue)

    # The session itself completed: the send is what failed, and a session that reports
    # that is more useful than one that raises and leaves no record.
    assert first.outcome is SessionOutcome.COMPLETED
    assert any(a.kind == "submit_failed" for a in first.alerts)
    assert not first.is_clean

    # AAA went first and landed. BBB was accepted by the venue and then the call died, so
    # the venue holds it and our side does not know. Everything after BBB was not attempted.
    assert venue.positions() == {"AAA": 100.0, "BBB": 100.0}
    assert len(first.sent) == 1 and first.sent[0].startswith("2026-09-18:AAA:")

    # The restart: a new process against the same log and the same venue.
    restarted_store = OrderStore(tmp_path / "orders.jsonl")
    restarted_switch = KillSwitch(tmp_path / "kill_switch.json")
    second = run(tmp_path, restarted_store, restarted_switch, venue)

    # The point of the test. Recovery settled BBB before anything was generated, so the
    # second session does *not* re-place it — it places only the eight it never attempted.
    assert second.plan is not None
    assert set(second.plan.tickers) == set(TICKERS) - {"AAA", "BBB"}
    assert venue.positions()["AAA"] == 100.0
    assert venue.positions()["BBB"] == 100.0              # not 200
    assert venue.positions() == dict.fromkeys(TICKERS, 100.0)

    # Each name accepted exactly once across both sessions.
    assert len(venue.accepted_ids) == len(TICKERS)
    assert len(set(venue.accepted_ids)) == len(TICKERS)


def test_a_failed_send_stops_the_rest_of_the_queue(tmp_path):
    # The outcome of the failed one is unknown, so the book is no longer a number this
    # session can size against. Carrying on down the list would size against it anyway.
    store, switch, venue = setup(tmp_path)
    venue.arm(Fault("submit", FailPoint.BEFORE_ACCEPT, ticker="AAA"))

    report = run(tmp_path, store, switch, venue)

    assert report.sent == ()                           # AAA was first and it failed
    assert len(report.withheld) == len(TICKERS)        # AAA, plus every untried name
    assert not report.is_clean


# --- the kill switch ------------------------------------------------------------------

def test_a_switch_already_down_halts_before_anything_is_generated(tmp_path):
    store, switch, venue = setup(tmp_path)
    switch.trip(TripReason.DRAWDOWN, "set by hand for this test")

    report = run(tmp_path, store, switch, venue)

    assert report.outcome is SessionOutcome.HALTED
    assert report.plan is None               # nothing was even reconciled
    assert report.sent == ()
    assert venue.positions() == {}
    assert [a.kind for a in report.alerts] == ["kill_switch"]
    assert report.alerts[0].severity is Severity.CRITICAL


def test_a_drawdown_trips_the_switch_and_the_veto_is_what_refuses(tmp_path):
    # The three parts doing their separate jobs within one session: the tripwire notices,
    # the state remembers, and the veto is the only thing that actually stops an order.
    store, switch, venue = setup(tmp_path)
    run(tmp_path, store, switch, venue)                   # establishes the peak at 100k

    venue.prices = dict.fromkeys(TICKERS, 50.0)           # equity halves
    report = run(tmp_path, store, switch, venue,
                 panel=panel_for(price=50.0))

    assert any(b.reason is TripReason.DRAWDOWN for b in report.breaches)
    assert switch.is_tripped
    assert report.decision is not None and report.decision.allowed == ()
    assert not report.is_clean

    # And it survives the restart that so often follows whatever tripped it.
    assert KillSwitch(tmp_path / "kill_switch.json").is_tripped


def test_a_tripwire_breach_pulls_working_orders_only_when_armed(tmp_path):
    # Driven off the NAV tripwire rather than the drawdown one, because with `auto_fill`
    # off nothing ever fills, so equity never moves and there is no drawdown to find. The
    # broker prices a holding at 100 while our panel marks it at 50, which is exactly the
    # "these two are describing different portfolios" condition.
    store, switch, venue = setup(tmp_path, auto_fill=False, holdings={"AAA": 400.0})
    run(tmp_path, store, switch, venue)                   # leaves orders working

    report = run(tmp_path, store, switch, venue, panel=panel_for(price=50.0))

    assert any(b.reason is TripReason.NAV_MISMATCH for b in report.breaches), report.describe()
    assert any(a.kind == "cancelled" for a in report.alerts), report.describe()


def test_a_dry_run_never_cancels_at_the_venue(tmp_path):
    store, switch, venue = setup(tmp_path, auto_fill=False)
    run(tmp_path, store, switch, venue)
    switch.trip(TripReason.DRAWDOWN, "set by hand")

    report = run(tmp_path, store, switch, venue, submit=False)

    assert report.outcome is SessionOutcome.HALTED
    assert not any(a.kind == "cancelled" for a in report.alerts)


# --- what it refuses ------------------------------------------------------------------

def test_a_working_order_holds_its_name_back(tmp_path):
    store, switch, venue = setup(tmp_path, auto_fill=False)
    run(tmp_path, store, switch, venue)

    report = run(tmp_path, store, switch, venue)

    assert report.plan is not None
    assert set(report.plan.deferred) == set(TICKERS)
    assert report.plan.intents == ()
    assert any(a.kind == "orders_still_working" for a in report.alerts)


def test_an_unpriced_holding_is_critical_rather_than_valued_at_zero(tmp_path):
    # A position our data cannot price is valued at zero, so it neither appears in the
    # weights nor objects. That is the quiet failure the tripwire exists to make loud.
    store, switch, venue = setup(tmp_path, holdings={"ZZZ": 10.0},
                                 prices={**dict.fromkeys(TICKERS, 100.0), "ZZZ": 100.0})
    report = run(tmp_path, store, switch, venue)

    assert report.plan is not None and "ZZZ" in report.plan.unpriced
    assert any(a.kind == "unpriced_holding" and a.severity is Severity.CRITICAL
               for a in report.alerts)
    assert not report.is_clean


def test_a_stale_panel_fails_the_session_after_recovery_has_run(tmp_path):
    # The deliberate deviation from the Phase 4 sketch: the hard refusal still happens, but
    # recovery gets to settle the previous session's orders first.
    store, switch, venue = setup(tmp_path)
    report = run_session(
        broker=venue, store=store, kill_switch=switch,
        strategy=EqualWeightStrategy(), universe=UNIVERSE, panel=panel_for(),
        as_of=None,                       # not a replay, so the guards apply
        submit=True, today=SESSION + pd.Timedelta(days=90),
        session_log=tmp_path / "sessions.jsonl")

    assert report.outcome is SessionOutcome.FAILED
    assert "stale" in report.error.lower()
    assert report.staleness_days == 90
    assert report.sent == ()
    assert venue.positions() == {}


def test_a_position_the_log_cannot_explain_is_reported_as_drift(tmp_path):
    store, switch, venue = setup(tmp_path)
    run(tmp_path, store, switch, venue)

    # Something arrives at the venue that this system never ordered — a manual trade, a
    # corporate action, another process on the same account.
    venue.holdings["AAA"] = venue.holdings["AAA"] + 25.0
    report = run(tmp_path, store, switch, venue)

    opening = [d for d in report.drift if d.when == "opening"]
    assert [d.ticker for d in opening] == ["AAA"], report.describe()
    assert opening[0].gap == pytest.approx(25.0)
    assert any(a.kind == "position_drift:opening" for a in report.alerts)
    assert not report.is_clean


# --- the record -----------------------------------------------------------------------

def test_every_session_appends_exactly_one_line_to_the_log(tmp_path):
    # The gate is thirty clean unattended sessions, so the count has to come from a record
    # written as each one happens rather than reconstructed from the order log — which only
    # knows about sessions that placed an order.
    store, switch, venue = setup(tmp_path)
    log = tmp_path / "sessions.jsonl"

    run(tmp_path, store, switch, venue)
    run(tmp_path, store, switch, venue)

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["session_date"] == "2026-09-18" for row in rows)
    assert all(row["outcome"] == "completed" and row["clean"] for row in rows)
    assert rows[0]["sent"] and not rows[1]["sent"]


def test_a_halted_session_is_recorded_too(tmp_path):
    store, switch, venue = setup(tmp_path)
    switch.trip(TripReason.NAV_MISMATCH, "set by hand")
    run(tmp_path, store, switch, venue)

    rows = [json.loads(line)
            for line in (tmp_path / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "halted" and rows[0]["clean"] is False


def test_the_record_is_json_serialisable_even_with_numpy_in_it(tmp_path):
    # NAV and the disagreement come off pandas, so they can arrive as numpy scalars. A
    # record that cannot be written is a session that happened and left no trace.
    store, switch, venue = setup(tmp_path)
    report = run(tmp_path, store, switch, venue)

    encoded = json.dumps(report.as_record(), default=str)
    assert "2026-09-18" in encoded
    assert isinstance(json.loads(encoded)["nav"], (int, float, type(None)))


def test_is_clean_is_false_when_a_completed_session_raised_an_alert(tmp_path):
    # The gate counts clean sessions, and "completed" is not the same claim. An alert is by
    # definition the thing that needed a human, which is what unattended operation means.
    store, switch, venue = setup(tmp_path, holdings={"ZZZ": 1.0},
                                 prices={**dict.fromkeys(TICKERS, 100.0), "ZZZ": 1.0})
    report = run(tmp_path, store, switch, venue)

    assert report.outcome is SessionOutcome.COMPLETED
    assert report.alerts
    assert not report.is_clean


def test_the_runner_reports_a_broker_that_cannot_be_reached(tmp_path):
    # An unattended session that cannot reach the venue has to be a recorded failure, not a
    # quiet day — a quiet day is what the gate would otherwise count it as.
    store, switch, venue = setup(tmp_path)
    venue.arm(Fault("positions"))

    report = run(tmp_path, store, switch, venue)

    assert report.outcome is SessionOutcome.FAILED
    assert report.error
    assert not report.is_clean


def test_describe_mentions_the_dry_run_so_a_log_cannot_be_misread(tmp_path):
    store, switch, venue = setup(tmp_path)
    assert "dry run" in run(tmp_path, store, switch, venue, submit=False).describe()
    assert "dry run" not in run(tmp_path, store, switch, venue, submit=True).describe()


def test_an_empty_panel_is_refused_rather_than_traded_on(tmp_path):
    store, switch, venue = setup(tmp_path)
    empty = make_panel(pd.DataFrame(index=pd.DatetimeIndex([]), columns=list(TICKERS),
                                    dtype=float))
    with pytest.raises(IndexError):
        # No session date can be derived from a panel with no dates, and inventing one is
        # how a session trades against a book it has no prices for.
        run_session(broker=venue, store=store, kill_switch=switch,
                    strategy=EqualWeightStrategy(), universe=UNIVERSE, panel=empty,
                    as_of=None, submit=False, session_log=None)


def test_nav_comes_from_the_broker_and_the_disagreement_is_measured(tmp_path):
    # Where weights become shares: the venue's equity, with our own marks computed beside
    # it so a divergence is a number rather than a surprise.
    store, switch, venue = setup(tmp_path, cash=100_000.0)
    report = run(tmp_path, store, switch, venue, submit=False)

    assert report.plan is not None
    assert report.plan.nav == pytest.approx(100_000.0)
    assert report.plan.nav_disagreement == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(report.plan.marked_nav)
