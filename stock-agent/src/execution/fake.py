import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.execution.broker import (Account, BrokerError, BrokerFill, DuplicateOrderError,
                                  OrderAck, OrderIntent, OrderState, OrderStatus)

# A broker that can be told to die at a chosen point. The roadmap names a crash halfway
# through a rebalance as the real risk rather than signal decay, and it is the one thing a
# paper account will not reproduce on demand.

# This object is the *remote* broker, so it outlives our process: a test simulates a
# restart by building a new session against the same FakeBroker, exactly as reconnecting
# to Alpaca would find the orders it accepted before the crash.

# It does not enforce long-only, and will short on a sell it cannot cover. That is faithful
# to a margin account, and it is what makes "the veto is what stopped this" provable rather
# than an artefact of the fake being polite.

FILL_TOLERANCE = 1e-9


class FailPoint(str, Enum):
    # The distinction the whole design turns on. BEFORE_ACCEPT is the harmless crash — the
    # broker never saw it. AFTER_ACCEPT is the dangerous one: the broker holds the order
    # and has already moved the book, and the caller sees only an exception.
    BEFORE_ACCEPT = "before_accept"
    AFTER_ACCEPT = "after_accept"


@dataclass
class Fault:
    # Fires once, then disarms, so a test says "fail this call" rather than "fail until I
    # say stop" and the recovery path afterwards runs against a broker that works again.
    method: str
    point: FailPoint = FailPoint.BEFORE_ACCEPT
    ticker: str | None = None            # `submit` only: fire on this name
    message: str = "injected broker failure"


@dataclass
class _Order:
    intent: OrderIntent
    broker_order_id: str
    accepted_at: pd.Timestamp
    state: OrderState = OrderState.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float | None = None

    @property
    def remaining(self) -> float:
        return max(self.intent.qty - self.filled_qty, 0.0)

    def status(self) -> OrderStatus:
        return OrderStatus(client_order_id=self.intent.client_order_id, state=self.state,
                           filled_qty=self.filled_qty, avg_fill_price=self.avg_fill_price)


@dataclass
class FakeBroker:
    prices: dict[str, float]
    cash: float = 100_000.0
    holdings: dict[str, float] = field(default_factory=dict)
    auto_fill: bool = True
    now: pd.Timestamp = pd.Timestamp("2026-01-02")

    _orders: dict[str, _Order] = field(default_factory=dict, init=False)
    _fills: list[BrokerFill] = field(default_factory=list, init=False)
    _faults: list[Fault] = field(default_factory=list, init=False)
    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.prices = {t: float(p) for t, p in self.prices.items()}
        self.holdings = {t: float(q) for t, q in self.holdings.items() if q != 0}
        self.now = pd.Timestamp(self.now)

    # --- Broker protocol -------------------------------------------------------------

    def positions(self) -> dict[str, float]:
        self._maybe_fail("positions")
        return {t: q for t, q in sorted(self.holdings.items()) if q != 0}

    def account(self) -> Account:
        self._maybe_fail("account")
        equity = self.cash + sum(q * self._price(t) for t, q in self.holdings.items())
        return Account(cash=self.cash, equity=equity, buying_power=max(self.cash, 0.0))

    def submit(self, intent: OrderIntent) -> OrderAck:
        self._maybe_fail("submit", FailPoint.BEFORE_ACCEPT, intent.ticker)
        price = self._price(intent.ticker)   # an unknown symbol is refused before acceptance

        coid = intent.client_order_id
        if coid in self._orders:
            raise DuplicateOrderError(
                f"{coid} was already accepted — refusing to place it twice. A resubmit "
                "means recovery did not resolve the previous session's order.")

        self._seq += 1
        order = _Order(intent=intent, broker_order_id=f"fake-{self._seq:06d}",
                       accepted_at=self.now)
        self._orders[coid] = order
        if self.auto_fill:
            self._book(order, intent.qty, price)

        # The crash window, and the reason this class exists: the order is accepted and the
        # book has already moved, and the caller is about to learn nothing about it.
        self._maybe_fail("submit", FailPoint.AFTER_ACCEPT, intent.ticker)
        return OrderAck(client_order_id=coid, broker_order_id=order.broker_order_id,
                        accepted_at=order.accepted_at)

    def order_status(self, client_order_id: str) -> OrderStatus | None:
        self._maybe_fail("order_status")
        order = self._orders.get(client_order_id)
        return None if order is None else order.status()

    def cancel(self, client_order_id: str) -> None:
        self._maybe_fail("cancel")
        order = self._orders.get(client_order_id)
        if order is None:
            raise BrokerError(f"{client_order_id} is not an order this broker has seen")
        if not order.status().is_terminal:
            order.state = OrderState.CANCELED

    def fills(self, since=None) -> list[BrokerFill]:
        self._maybe_fail("fills")
        if since is None:
            return list(self._fills)
        since = pd.Timestamp(since)
        return [f for f in self._fills if f.filled_at >= since]

    # --- Test controls ---------------------------------------------------------------

    def arm(self, fault: Fault) -> "FakeBroker":
        self._faults.append(fault)
        return self

    def set_price(self, ticker: str, price: float) -> None:
        self.prices[ticker] = float(price)

    def fill_open(self, client_order_id: str | None = None, *,
                  qty: float | None = None, price: float | None = None) -> None:
        # With `auto_fill` off, this is how a test lands a partial. An order that is
        # neither done nor abandoned is the state a reconciler is most likely to
        # mishandle, and submitting alone cannot reach it.
        targets = ([self._orders[client_order_id]] if client_order_id is not None
                   else [o for o in self._orders.values() if not o.status().is_terminal])
        for order in targets:
            amount = order.remaining if qty is None else min(float(qty), order.remaining)
            if amount > FILL_TOLERANCE:
                self._book(order, amount,
                           self._price(order.intent.ticker) if price is None else float(price))

    @property
    def accepted_ids(self) -> tuple[str, ...]:
        # Every id the broker took, including from calls that raised after accepting. "No
        # duplicate position after restart" is only half the claim — the other half is that
        # the second submit was refused, and this is what shows it.
        return tuple(self._orders)

    # --- Internals -------------------------------------------------------------------

    def _book(self, order: _Order, qty: float, price: float) -> None:
        ticker = order.intent.ticker
        signed = qty * order.intent.side.sign

        self.holdings[ticker] = self.holdings.get(ticker, 0.0) + signed
        if abs(self.holdings[ticker]) < FILL_TOLERANCE:
            self.holdings.pop(ticker)
        self.cash -= signed * price

        filled = order.filled_qty + qty
        prior = (order.avg_fill_price or 0.0) * order.filled_qty
        order.avg_fill_price = (prior + price * qty) / filled
        order.filled_qty = filled
        order.state = (OrderState.FILLED if order.remaining <= FILL_TOLERANCE
                       else OrderState.PARTIALLY_FILLED)

        self._fills.append(BrokerFill(
            client_order_id=order.intent.client_order_id, ticker=ticker, qty=qty,
            side=order.intent.side, fill_price=price, filled_at=self.now))

    def _price(self, ticker: str) -> float:
        # Refuses to guess rather than defaulting to zero: a missing price would value a
        # holding at nothing and make `account()` quietly wrong, which is the class of bug
        # every guard in this repo exists to prevent.
        if ticker not in self.prices:
            raise BrokerError(f"no price for {ticker} — seed it in FakeBroker(prices=...)")
        return self.prices[ticker]

    def _maybe_fail(self, method: str, point: FailPoint = FailPoint.BEFORE_ACCEPT,
                    ticker: str | None = None) -> None:
        for fault in self._faults:
            if fault.method != method or fault.point != point:
                continue
            if fault.ticker is not None and fault.ticker != ticker:
                continue
            self._faults.remove(fault)
            raise BrokerError(fault.message)
