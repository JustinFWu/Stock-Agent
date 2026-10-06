import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.config import ADV_WINDOW, NO_TRADE_BAND, SESSION_LOG_PATH
from stock_agent.data.panel import PricePanel, load_price_panel
from stock_agent.data.universe import UNIVERSE, UniverseSpec
from stock_agent.execution.broker import Broker, BrokerError, OrderIntent
from stock_agent.execution.killswitch import Breach, KillSwitch
from stock_agent.execution.reconcile import Reconciliation, reconcile
from stock_agent.execution.recovery import RecoveryReport, recover, submit_intent
from stock_agent.execution.store import OrderStore
from stock_agent.execution.veto import VetoResult, veto
from stock_agent.strategy.live import live_target_weights
from stock_agent.strategy.weights import Strategy

# The sequence, as a program. Every piece it calls already existed and
# `tests/test_execution_chain.py` already threaded them by hand — which was the honest
# state of it, and also the reason the Phase 4 gate stood at zero of thirty sessions. A
# sequence that lives in a test is a claim that the parts compose; a sequence that lives
# here is something that can run unattended and leave a record behind.

# The ordering is the design, and the one rule it exists to enforce is that RECOVERY RUNS
# BEFORE GENERATION. `reconcile` already refuses to produce intents unless the log is in
# the state a completed recovery leaves behind, so getting this wrong here is a ValueError
# rather than a double order. This module is what makes the rest of the order deliberate.

# Deviation from the Phase 4 sketch, recorded because it is deliberate. The sketch puts the
# staleness and coverage guards at step 2, before recovery at step 3. Those guards raise,
# so a stale cache would exit the session before recovery had settled the previous one's
# orders — and an unresolved order left unresolved for an extra day is the condition the
# whole write-ahead design exists to shorten. So: staleness and coverage are MEASURED early
# and fed to the tripwire, recovery runs next, and the hard refusal stays where it already
# lived, inside `live_target_weights`, which is reached after recovery. Nothing can trade on
# stale data either way; the difference is only whether the log gets settled first.

# NO LLM IN THE MONEY PATH. Nothing in this module calls a model. Deterministic checks own
# the alerts and the kill switch; a model may read `SessionReport.describe()` afterwards and
# draft a note about it, which is a different program.


class SessionOutcome(str, Enum):
    # What the session did, as one word, because the gate is counted from these.
    COMPLETED = "completed"       # ran the whole sequence; submitted whatever was allowed
    HALTED = "halted"             # the kill switch was already down; nothing was generated
    FAILED = "failed"             # the sequence could not be completed


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    severity: Severity
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.kind}: {self.detail}"


@dataclass(frozen=True)
class PositionDrift:
    # What the broker holds against what our record can explain. Reported per name rather
    # than as one boolean, because "which name" is the first thing anyone asks.
    #
    # `when` separates two different failures that would otherwise be one number:
    #
    #   "opening" — the book moved between sessions by something this system did not do.
    #               A manual trade, a corporate action, a second process on the account.
    #   "closing" — the book did not move the way this session's own fills say it did.
    #               A fill we recorded that the venue did not make, or the reverse.
    #
    # The closing check alone cannot see the first one, because it measures against the
    # book as this session found it — and that book already contains the surprise.
    ticker: str
    expected: float
    actual: float
    when: str = "closing"

    @property
    def gap(self) -> float:
        return self.actual - self.expected


@dataclass(frozen=True)
class SessionReport:
    session_date: pd.Timestamp
    outcome: SessionOutcome
    submitted: bool = False          # whether this run was armed to send orders

    recovery: RecoveryReport | None = None
    plan: Reconciliation | None = None
    breaches: tuple[Breach, ...] = ()
    decision: VetoResult | None = None

    sent: tuple[str, ...] = ()       # client_order_ids this session actually submitted
    withheld: tuple[str, ...] = ()   # allowed by the veto, not sent (dry run, or an error)
    fill_poll: RecoveryReport | None = None
    drift: tuple[PositionDrift, ...] = ()

    # The broker's book as this session left it, and the baseline the next session's
    # opening drift check measures against. Pairs rather than a dict so the report stays a
    # frozen dataclass that can be hashed and compared.
    closing_book: tuple[tuple[str, float], ...] = ()

    staleness_days: int | None = None
    coverage: float | None = None
    alerts: tuple[Alert, ...] = ()
    error: str = ""

    @property
    def is_clean(self) -> bool:
        # What the Phase 4 gate counts. Deliberately strict: a session that completed but
        # had something to alert about is not a clean session, because the gate is a claim
        # about unattended operation and an alert is the thing that needed attention.
        return (self.outcome is SessionOutcome.COMPLETED
                and not self.alerts
                and not self.drift
                and not self.breaches
                and (self.decision is None or self.decision.is_clean))

    def describe(self) -> str:
        lines = [f"session {self.session_date.date()} — {self.outcome.value}"
                 f"{'' if self.submitted else ' (dry run, nothing sent)'}"]

        if self.plan is not None:
            lines.append(f"  nav {self.plan.nav:,.2f} from the broker, "
                         f"marks disagree {self.plan.nav_disagreement:+.3%}")
            lines.append(f"  intents {len(self.plan.intents)}"
                         f"   blocked {len(self.plan.blocked)}"
                         f"   deferred {len(self.plan.deferred)}"
                         f"   unpriced {len(self.plan.unpriced)}")
        if self.decision is not None:
            lines.append("  " + self.decision.describe().replace("\n", "\n  "))
        if self.sent:
            lines.append(f"  sent {len(self.sent)}: {', '.join(self.sent)}")
        if self.withheld:
            lines.append(f"  withheld {len(self.withheld)}: {', '.join(self.withheld)}")
        if self.fill_poll is not None:
            lines.append(f"  fill poll: {len(self.fill_poll.resolved)} resolved, "
                         f"{len(self.fill_poll.working)} still working, "
                         f"{len(self.fill_poll.abandoned)} abandoned")
        for d in self.drift:
            lines.append(f"  DRIFT ({d.when}) {d.ticker}: broker {d.actual:,.4f} against "
                         f"{d.expected:,.4f} explained ({d.gap:+,.4f})")
        for alert in self.alerts:
            lines.append(f"  {alert}")
        if self.error:
            lines.append(f"  error: {self.error}")
        lines.append(f"  clean: {self.is_clean}")
        return "\n".join(lines)

    def as_record(self) -> dict:
        return {
            "session_date": f"{self.session_date:%Y-%m-%d}",
            "logged_at": pd.Timestamp.now("UTC").isoformat(),
            "outcome": self.outcome.value,
            "armed": self.submitted,
            "clean": self.is_clean,
            "intents": len(self.plan.intents) if self.plan else 0,
            "sent": list(self.sent),
            "withheld": list(self.withheld),
            "rejected": [
                {"ticker": r.intent.ticker, "reason": r.reason.value, "detail": r.detail}
                for r in (self.decision.rejected if self.decision else ())
            ],
            "breaches": [{"reason": b.reason.value, "detail": b.detail}
                         for b in self.breaches],
            "drift": [{"ticker": d.ticker, "expected": d.expected, "actual": d.actual,
                       "when": d.when} for d in self.drift],
            "closing_book": dict(self.closing_book),
            "nav": self.plan.nav if self.plan else None,
            "nav_disagreement": self.plan.nav_disagreement if self.plan else None,
            "staleness_days": self.staleness_days,
            "coverage": self.coverage,
            "alerts": [{"severity": a.severity.value, "kind": a.kind, "detail": a.detail}
                       for a in self.alerts],
            "error": self.error,
        }


@dataclass
class _Session:
    # A builder, so the twelve steps read as twelve steps rather than as one function
    # threading nine locals through forty lines. Nothing here is reusable; it exists to
    # keep `run_session` readable.
    broker: Broker
    store: OrderStore
    kill_switch: KillSwitch
    strategy: Strategy
    universe: UniverseSpec
    session_date: pd.Timestamp
    submit: bool
    no_trade_band: float
    alerts: list[Alert] = field(default_factory=list)

    def alert(self, severity: Severity, kind: str, detail: str) -> None:
        self.alerts.append(Alert(severity, kind, detail))


def run_session(
    *,
    broker: Broker,
    store: OrderStore,
    kill_switch: KillSwitch,
    strategy: Strategy,
    universe: UniverseSpec = UNIVERSE,
    panel: PricePanel | None = None,
    as_of=None,
    submit: bool = False,
    no_trade_band: float = NO_TRADE_BAND,
    today=None,
    session_log: os.PathLike | str | None = SESSION_LOG_PATH,
    cancel_working_on_trip: bool = True,
) -> SessionReport:
    # `submit=False` is the default and it means the whole sequence runs against the live
    # broker — positions, account, recovery, reconciliation, tripwires, veto — and stops
    # before `submit()`. That is the useful shape: everything that can be wrong about a
    # session is wrong before the order is sent, so a dry run is worth something.

    if panel is None:
        panel = load_price_panel(list(universe.tickers))

    # The session's date is the session being traded for, not the wall clock. An explicit
    # `as_of` is a deliberate replay of a session that was missed; the default is the last
    # one the data knows about.
    session_date = pd.Timestamp(as_of).normalize() if as_of is not None else panel.dates[-1]

    state = _Session(broker=broker, store=store, kill_switch=kill_switch,
                     strategy=strategy, universe=universe, session_date=session_date,
                     submit=submit, no_trade_band=no_trade_band)

    # The previous session's closing book, read back from the operating record. This is the
    # only baseline against which "the book moved and we did not move it" is answerable:
    # the order log knows what we did, and the broker knows where the book is now, but
    # neither knows where it was when we last looked. A fresh log has no baseline, so the
    # first session establishes one rather than reporting the whole account as drift.
    previous_book = _last_recorded_book(session_log)

    report = _run(state, panel, as_of, today, cancel_working_on_trip, previous_book)
    if session_log is not None:
        _append_session_log(session_log, report)
    return report


def _run(state: _Session, panel: PricePanel, as_of, today,
         cancel_working_on_trip: bool,
         previous_book: dict[str, float] | None = None) -> SessionReport:
    # Step 1 — the kill switch, before anything else and before anything is asked of the
    # venue. A switch that is already down means a human has not yet looked at why, and
    # the one thing that must not happen is a new order on top of the reason.
    if state.kill_switch.is_tripped:
        state.alert(Severity.CRITICAL, "kill_switch",
                    f"halted and not trading: {state.kill_switch.state.describe()}")
        return SessionReport(session_date=state.session_date, outcome=SessionOutcome.HALTED,
                             submitted=False, alerts=tuple(state.alerts))

    # Step 2 — the data, measured rather than enforced here. See the note at the top of the
    # module for why these two are numbers at this point and a refusal later.
    staleness_days, coverage = _measure_data(panel, state, today)

    try:
        # Step 3 — recovery, and the fill poll for anything the last session left open.
        # Strictly before generation: `reconcile` refuses otherwise.
        recovery = recover(state.store, state.broker)
        _alert_on_recovery(state, recovery)

        # Step 4 — ground truth. The broker's book and the broker's equity, never ours.
        positions = state.broker.positions()
        account = state.broker.account()

        # Half of step 12, and it belongs here rather than at the end: did the book move
        # between sessions by something other than our own fills? Checked before the
        # session acts on the book, so the answer is about the book it found.
        opening_drift = _opening_drift(state, previous_book, positions, recovery)

        # Step 5 — the target. This is where staleness and coverage become a refusal.
        # `today` is passed through rather than left to the wall clock, because that is
        # what makes the staleness guard testable — the same reason `live.py` accepts it.
        target = live_target_weights(state.strategy, as_of=as_of, panel=panel,
                                     universe=state.universe, today=today)

        # Step 6 — target minus actual, in shares, against the broker's equity.
        prices, marks, adv = _price_vectors(panel, state.session_date)
        plan = reconcile(state.session_date, target, positions=positions, account=account,
                         prices=prices, marks=marks, store=state.store, recovery=recovery,
                         no_trade_band=state.no_trade_band)
        _alert_on_plan(state, plan)

        # Step 7 — the tripwires. `check` both evaluates and persists a trip, so a breach
        # here is what makes the veto below refuse everything.
        breaches = state.kill_switch.check(
            equity=account.equity, positions=positions,
            staleness_days=staleness_days, coverage=coverage,
            nav_disagreement=plan.nav_disagreement, unpriced_positions=plan.unpriced)
        for breach in breaches:
            state.alert(Severity.CRITICAL, f"tripwire:{breach.reason.value}", breach.detail)

        # Pulling working orders is what makes the switch a stop rather than a pause. Only
        # when armed: a dry run must not change anything at the venue.
        if breaches and cancel_working_on_trip and state.submit:
            _cancel_working(state, recovery)

        # Step 8 — the veto, against the post-trade book. It should never fire.
        decision = veto(plan.intents, positions=positions, account=account, marks=marks,
                        universe=state.universe,
                        kill_switch_tripped=state.kill_switch.is_tripped,
                        adv_notional=adv)
        for rejection in decision.rejected:
            # Only a kill-switch rejection is expected. Every other reason means
            # `target_weights` returned something its own caps should have prevented.
            severity = (Severity.WARNING if state.kill_switch.is_tripped
                        else Severity.CRITICAL)
            state.alert(severity, f"veto:{rejection.reason.value}",
                        f"{rejection.intent.ticker}: {rejection.detail}")

        # Steps 9 and 10 — write-ahead, then send. One call, because the order of those two
        # is the entire crash contract and `submit_intent` is where it lives.
        sent, withheld = _submit_all(state, decision.allowed)

        # Step 11 — poll fills. The same function as step 3, deliberately: two
        # implementations is how a startup path and an end-of-session path come to disagree
        # about what a partial fill means.
        fill_poll = recover(state.store, state.broker)

        # Step 12 — post-session reconciliation. Does the broker's book match what our
        # record can account for?
        # Read once: the closing book is both the drift check's input and the baseline the
        # next session measures against, and reading it twice would let the two disagree.
        closing = _closing_positions(state)
        drift = opening_drift + _closing_drift(state, positions, fill_poll, closing)
        for d in drift:
            state.alert(Severity.CRITICAL, f"position_drift:{d.when}",
                        f"{d.ticker}: broker holds {d.actual:,.4f} against "
                        f"{d.expected:,.4f} our record can explain ({d.gap:+,.4f})")

        _alert_on_unknown_statuses(state)

        return SessionReport(
            session_date=state.session_date, outcome=SessionOutcome.COMPLETED,
            submitted=state.submit, recovery=recovery, plan=plan, breaches=breaches,
            decision=decision, sent=sent, withheld=withheld, fill_poll=fill_poll,
            drift=drift, staleness_days=staleness_days, coverage=coverage,
            closing_book=tuple(sorted((closing or {}).items())),
            alerts=tuple(state.alerts))

    except Exception as exc:
        # A failed session is a reported session. Swallowing the exception would make an
        # unattended run look like a quiet day, which is the one thing the gate must not be
        # able to count. The traceback is the operator's; the record is the gate's.
        state.alert(Severity.CRITICAL, type(exc).__name__, str(exc))
        return SessionReport(
            session_date=state.session_date, outcome=SessionOutcome.FAILED,
            submitted=state.submit, staleness_days=staleness_days, coverage=coverage,
            alerts=tuple(state.alerts), error=f"{type(exc).__name__}: {exc}")


def _measure_data(panel: PricePanel, state: _Session, today) -> tuple[int | None, float | None]:
    # Measured, not enforced. `live_target_weights` is what refuses, and it refuses with a
    # message written for this exact situation; duplicating the thresholds here would give
    # the repo two answers to "is this data fresh enough".
    if len(panel.dates) == 0:
        state.alert(Severity.CRITICAL, "no_data", "the price panel is empty")
        return None, None

    reference = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    staleness = int((reference - state.session_date).days)

    expected = len(state.universe.members_asof(state.session_date))
    if state.session_date in panel.dates:
        covered = len(panel.as_of(state.session_date).tradable_as_of(state.session_date))
        coverage = covered / expected if expected else None
    else:
        coverage = None
    return staleness, coverage


def _price_vectors(panel: PricePanel, session_date) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Three vectors, and the distinction between the first two is the same one
    # `plan_trades` is careful about in the backtest.
    #
    # `prices` are raw: a name with no bar on this session has no executable price and
    # lands in `blocked` rather than being traded at a stale one. `marks` are carried
    # forward, because the book has to be valued even where it cannot be traded.
    #
    # Both are the session's close, which is what a session running after that close
    # actually knows. The orders will fill at the next open, so these are an estimate used
    # for sizing and never a fill price — which is why the no-trade band has slack in it
    # and why the veto's cash check reads the venue's own buying power rather than ours.
    visible = panel.as_of(session_date)
    closes = visible.closes

    prices = closes.loc[session_date]
    marks = closes.ffill().loc[session_date]

    # The backtest shifts this by one session because today's close is not knowable at
    # today's open. Running after the close, it is knowable, so there is nothing to shift.
    adv = (closes * visible.volumes).rolling(ADV_WINDOW).mean().loc[session_date]
    return prices, marks, adv


def _submit_all(state: _Session, allowed) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not state.submit:
        return (), tuple(i.client_order_id for i in allowed)

    sent: list[str] = []
    for index, intent in enumerate(allowed):
        try:
            submit_intent(state.store, state.broker, intent)
            sent.append(intent.client_order_id)
        except BrokerError as exc:
            # The outcome of this one is unknown, so the session stops sending. Carrying on
            # down the list would place orders sized against a book whose state we have
            # just lost track of, and the next session's recovery is what settles it.
            state.alert(Severity.CRITICAL, "submit_failed",
                        f"{intent.ticker}: {exc}. Stopped sending; "
                        f"{len(allowed) - index - 1} intent(s) not attempted.")
            return tuple(sent), tuple(i.client_order_id for i in allowed[index:])
    return tuple(sent), ()


def _cancel_working(state: _Session, recovery: RecoveryReport) -> None:
    for record in recovery.working:
        try:
            state.broker.cancel(record.client_order_id)
            state.alert(Severity.WARNING, "cancelled",
                        f"{record.intent.ticker}: pulled on a tripwire breach")
        except BrokerError as exc:
            state.alert(Severity.CRITICAL, "cancel_failed",
                        f"{record.client_order_id}: {exc}")


def _opening_drift(state: _Session, previous_book: dict[str, float] | None,
                   positions: dict[str, float],
                   recovery: RecoveryReport) -> tuple[PositionDrift, ...]:
    # Did anything move the book since we last looked, other than our own orders?
    #
    # The baseline is the previous session's closing book. The allowance on top of it is
    # every order that resolved in *this* session's recovery — those are ours, they filled
    # overnight, and they are exactly the legitimate reason the book is different.
    if previous_book is None:
        return ()

    expected = dict(previous_book)
    for record in recovery.resolved:
        if record.filled_qty:
            expected[record.intent.ticker] = (expected.get(record.intent.ticker, 0.0)
                                              + record.filled_qty * record.intent.side.sign)

    # A name with an order still working is excluded: its quantity is mid-flight, and the
    # previous session recorded the book before that order had finished moving it.
    working = recovery.blocked_tickers
    return _compare(expected, positions, skip=working, when="opening")


def _closing_drift(state: _Session, positions_before: dict[str, float],
                   fill_poll: RecoveryReport,
                   actual: dict[str, float] | None) -> tuple[PositionDrift, ...]:
    # The question is not "does the book match the target" — it will not, because a market
    # order placed after the close fills at the next open, and a partial fill is legitimate.
    # The question is whether the book moved the way this session's own fills say it did.
    #
    # Expected = the book this session found, plus every fill the step-11 poll resolved.
    # Anything step 3's recovery settled was already settled before `positions_before` was
    # read, so it is in there and must not be added twice.
    expected = dict(positions_before)
    for record in fill_poll.resolved:
        if record.filled_qty:
            expected[record.intent.ticker] = (expected.get(record.intent.ticker, 0.0)
                                              + record.filled_qty * record.intent.side.sign)

    if actual is None:
        return ()
    return _compare(expected, actual, skip=fill_poll.blocked_tickers, when="closing")


def _closing_positions(state: _Session) -> dict[str, float] | None:
    try:
        return state.broker.positions()
    except BrokerError as exc:
        state.alert(Severity.CRITICAL, "drift_check_failed",
                    f"could not read positions back: {exc}")
        return None


def _compare(expected: dict[str, float], actual: dict[str, float],
             *, skip, when: str) -> tuple[PositionDrift, ...]:
    drifts = []
    for ticker in sorted(set(expected) | set(actual)):
        if ticker in skip:
            continue
        want = float(expected.get(ticker, 0.0))
        have = float(actual.get(ticker, 0.0))
        # A share is the unit; the tolerance is float noise, not a real allowance.
        if abs(have - want) > 1e-6:
            drifts.append(PositionDrift(ticker=ticker, expected=want, actual=have, when=when))
    return tuple(drifts)


def _last_recorded_book(session_log) -> dict[str, float] | None:
    # The most recent session that got far enough to record a closing book. A session that
    # failed before reading positions records none, so the baseline is the last one that
    # did and the fills in between are accounted for by recovery.
    if session_log is None:
        return None
    path = Path(session_log)
    if not path.exists():
        return None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A torn line is possible and is not a reason to refuse to run. It costs this
            # one baseline, and the session still reports its closing drift.
            continue
        book = row.get("closing_book")
        if isinstance(book, dict):
            return {str(t): float(q) for t, q in book.items()}
    return None


def _alert_on_recovery(state: _Session, recovery: RecoveryReport) -> None:
    for record in recovery.abandoned:
        state.alert(Severity.WARNING, "order_abandoned",
                    f"{record.intent.ticker} ({record.client_order_id}): the broker never "
                    "had it, so it was written off rather than assumed filled")
    if recovery.working:
        names = ", ".join(sorted(recovery.blocked_tickers))
        state.alert(Severity.WARNING, "orders_still_working",
                    f"{len(recovery.working)} order(s) open from a previous session on "
                    f"{names}; those names are held back this session")


def _alert_on_plan(state: _Session, plan: Reconciliation) -> None:
    if plan.blocked:
        state.alert(Severity.WARNING, "no_executable_price",
                    f"{', '.join(plan.blocked)} had no usable price this session")
    if plan.unpriced:
        state.alert(Severity.CRITICAL, "unpriced_holding",
                    f"the broker holds {', '.join(plan.unpriced)}, which our data cannot "
                    "price — the book being sized is not the book being held")


def _alert_on_unknown_statuses(state: _Session) -> None:
    # The adapter collects venue words it does not recognise rather than guessing at them.
    drain = getattr(state.broker, "drain_unknown_statuses", None)
    if drain is None:
        return
    for detail in drain():
        state.alert(Severity.WARNING, "unknown_order_status", detail)


def _append_session_log(path, report: SessionReport) -> None:
    # Append-only and fsynced, same reasoning as the order log: a session record sitting in
    # the OS buffer has not survived anything, and the gate counts these.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(report.as_record(), sort_keys=True, default=_json_safe)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def intents_summary(intents: tuple[OrderIntent, ...]) -> str:
    # Used by the CLI for a dry run, where the whole point is to read what *would* be sent.
    if not intents:
        return "    nothing to send"
    return "\n".join(
        f"    {i.side.value:<13} {i.qty:>12,.4f} {i.ticker:<6} {i.client_order_id}"
        for i in intents)
