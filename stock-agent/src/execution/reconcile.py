import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import NO_TRADE_BAND
from src.backtest.portfolio import Portfolio, plan_trades
from src.execution.broker import QTY_PRECISION, Account, OrderIntent, Side
from src.execution.recovery import RecoveryReport
from src.execution.store import OrderStore

# Target weights against the BROKER's positions, never against our own record. If the
# process died mid-rebalance, the local record is precisely the thing that is wrong.

# The trade arithmetic itself is `plan_trades`, the same function the backtester uses —
# the drift test, the no-trade band and the forced exit of a dropped name are one
# implementation or they are two that will disagree. Phase 2's review named the band and
# the rebalance schedule as the two remaining surfaces on which backtest and production
# can diverge; reusing the function is what closes the first of them.

# Where weights become shares, recorded because it is the decision this module exists to
# make. NAV is `broker.account().equity` — the venue's own number, not our marks. The two
# are computed side by side and their disagreement is returned, because a silent divergence
# between them is the exact failure `test_weight_parity.py` exists to catch one layer up.


@dataclass(frozen=True)
class Reconciliation:
    # `intents` is what would be sent; nothing here sends it. The veto runs in between, and
    # a reconciler that could submit would make that ordering a convention rather than a
    # structure.
    intents: tuple[OrderIntent, ...] = ()

    # Three different reasons a name the target wanted is absent, kept apart because they
    # call for different responses: no executable price today, an order still working on
    # it from a previous session, or a holding we cannot value at all.
    blocked: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()
    unpriced: tuple[str, ...] = ()

    nav: float = 0.0          # the broker's equity: what the target weights are fractions of
    marked_nav: float = 0.0   # the same book valued at our marks

    @property
    def nav_disagreement(self) -> float:
        # Signed, as a fraction of the broker's equity: positive means our marks value the
        # book above what the venue says it is worth.
        if self.nav <= 0:
            return 0.0
        return self.marked_nav / self.nav - 1.0

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(intent.ticker for intent in self.intents)


def reconcile(
    session_date,
    target: pd.Series,
    *,
    positions: dict[str, float],
    account: Account,
    prices: pd.Series,
    marks: pd.Series,
    store: OrderStore,
    recovery: RecoveryReport,
    no_trade_band: float = NO_TRADE_BAND,
) -> Reconciliation:
    # `recovery` and `store` are required arguments and that is the point of them. The
    # sketch's rule — recovery strictly before generation — was a comment until here;
    # intents cannot now be produced without a report that accounts for every order the
    # log still has open. Get it wrong and the session sizes against a position an
    # unresolved order is still about to move, then places a second order on top.
    _require_recovery(store, recovery)

    if not np.isfinite(account.equity) or account.equity <= 0:
        raise ValueError(
            f"the broker reports equity of {account.equity!r}, so there is no NAV to size "
            "against. Refusing to reconcile.")

    # Two price vectors, the same distinction `plan_trades` is careful about. `prices` are
    # raw: a name with no bar today has no executable price and lands in `blocked`. `marks`
    # are carried forward, and are what the book is valued at.
    book = Portfolio(cash=account.cash, shares={t: float(q) for t, q in positions.items() if q != 0})
    marked_nav = book.nav(marks)

    plan = plan_trades(target, book, prices, marks,
                       no_trade_band=no_trade_band, nav=account.equity)

    # A working order means the position is still moving, so any size computed against it
    # is computed against a number that is about to change. The name waits a session.
    deferred = sorted(recovery.blocked_tickers & set(plan.deltas.index))
    deltas = plan.deltas.drop(labels=deferred, errors="ignore")

    intents = tuple(
        OrderIntent(session_date=session_date, ticker=ticker, qty=qty, side=side)
        for ticker, qty, side in _sized(deltas, book.shares)
    )

    return Reconciliation(
        intents=intents,
        blocked=plan.blocked,
        deferred=tuple(deferred),
        unpriced=_unpriced(book.shares, marks),
        nav=float(account.equity),
        marked_nav=float(marked_nav),
    )


def _require_recovery(store: OrderStore, recovery: RecoveryReport) -> None:
    # Not "did someone call recover", which cannot be asked. The checkable version is that
    # nothing is open in the log except what this report already named as working — which
    # is exactly the state a completed recovery leaves behind.
    accounted = {record.client_order_id for record in recovery.working}
    stranded = sorted(record.client_order_id for record in store.unresolved()
                      if record.client_order_id not in accounted)
    if stranded:
        raise ValueError(
            f"{len(stranded)} order(s) are still open in the log and are not in the "
            f"recovery report: {', '.join(stranded)}. Run `recover()` against the broker "
            "before generating anything — a session that sizes past an unresolved order "
            "places a second one on top of it.")


def _sized(deltas: pd.Series, held: dict[str, float]):
    # Quantised here rather than left to `OrderIntent`, so a delta that rounds away is
    # dropped instead of raising on a non-positive quantity. Fractional shares stay:
    # Alpaca supports them and rounding to whole shares would reintroduce the drift the
    # no-trade band was sized to ignore.
    for ticker, delta in deltas.items():
        qty = round(abs(float(delta)), QTY_PRECISION)
        if qty <= 0:
            continue
        yield ticker, qty, _side(held.get(ticker, 0.0), float(delta))


def _side(current: float, delta: float) -> Side:
    # The vocabulary is complete; the behaviour is not. Nothing in v1 produces a short, and
    # a sell that would cross through zero is emitted as a plain SELL and rejected by the
    # veto on the post-trade quantity — which catches it whatever side it claims, and is
    # the check that keeps working the day something legitimately shorts.
    if delta > 0:
        return Side.BUY_TO_COVER if current < 0 else Side.BUY
    return Side.SELL_SHORT if current <= 0 else Side.SELL


def _unpriced(held: dict[str, float], marks: pd.Series) -> tuple[str, ...]:
    # A holding with no mark is valued at zero by `Portfolio.position_value`, so it neither
    # appears in the current weights nor objects. It is reported rather than inferred from
    # the NAV gap, because a small unpriced position hides inside the tolerance.
    return tuple(sorted(
        ticker for ticker, qty in held.items()
        if qty != 0 and not np.isfinite(marks.get(ticker, np.nan))))
