import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd

# The only layer that knows a vendor exists. The reconciler, the order store and the veto
# are all written in these types, so widening the vocabulary here later is cheap while
# changing what the arguments *mean* is not.

# Quantities are always positive and direction lives in `Side`. A bare signed share count
# cannot tell "sell 100 AAPL I own" from "short 100 AAPL I do not", and those are different
# actions — one needs a locate, margin and borrow. Nothing generates a short in v1.

# Shares are quantised before anything hashes or compares them. Two arithmetically
# identical sizings can differ in the last bit through a different summation order, and an
# id that moves with float noise is an id that cannot deduplicate. Same reason as
# WEIGHT_PRECISION in weights.py.
QTY_PRECISION = 6

# 48 bits over one session's order book. Long enough that a collision is not a real event,
# short enough that a human reading the write-ahead log can compare two ids at a glance.
ID_DIGEST_CHARS = 12


class Side(str, Enum):
    # `str` mixin so an intent serialises into the order store as plain JSON. The
    # write-ahead record is read by a human reconstructing a crashed session, and a
    # custom encoder is one more thing that can be wrong at exactly the wrong moment.
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"

    @property
    def sign(self) -> int:
        return 1 if self in (Side.BUY, Side.BUY_TO_COVER) else -1

    @property
    def opens_short(self) -> bool:
        # Only SELL_SHORT opens exposure — BUY_TO_COVER closes it. This is the cheap
        # assertion; the veto's real check is the post-trade quantity, which also catches
        # a plain SELL of more than the book holds.
        return self is Side.SELL_SHORT


class OrderState(str, Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


# Named once so the order store and the Protocol cannot disagree about what "done"
# means. A partial is absent on purpose: it can still complete, and treating it as
# settled is how a session sizes its next order against a position that is still moving.
TERMINAL_ORDER_STATES = frozenset(
    {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED})


# Raised when the outcome of a call is unknown — a timeout, a dropped connection. The
# caller must not assume the order failed to land, which is the entire reason
# `order_status` is on the Protocol rather than a convenience.
class BrokerError(Exception):
    pass


# The broker already holds this client_order_id. This is the *success* path for
# idempotency: a resubmit after a crash has to be refused rather than filled twice.
class DuplicateOrderError(BrokerError):
    pass


@dataclass(frozen=True)
class OrderIntent:
    session_date: pd.Timestamp
    ticker: str
    qty: float           # always positive; direction lives in `side`
    side: Side

    def __post_init__(self) -> None:
        # Positive qty is the invariant the whole `side` design rests on. A negative one
        # would invert every downstream sign silently, and a NaN passes `> 0` by failing
        # the comparison rather than satisfying it.
        qty = float(self.qty)
        if not math.isfinite(qty) or qty <= 0:
            raise ValueError(f"{self.ticker or '<no ticker>'}: intent qty must be finite "
                             f"and positive, got {self.qty!r} — direction belongs in `side`.")
        if not self.ticker:
            raise ValueError("an intent needs a ticker")

        object.__setattr__(self, "ticker", str(self.ticker))
        object.__setattr__(self, "qty", round(qty, QTY_PRECISION))
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "session_date", pd.Timestamp(self.session_date).normalize())

    @property
    def client_order_id(self) -> str:
        # Computed, never stored: an id that travels as a field can drift from the intent
        # it names, and "same session, same intent, same id" has to be structural rather
        # than a thing each caller remembers to do.

        # Note what this does *not* protect against. The digest covers qty, so a recovered
        # session that re-sizes a name by one share produces a new id and submits again.
        # That is why crash recovery runs before any order is generated, not after.
        return f"{self.session_date:%Y-%m-%d}:{self.ticker}:{self._digest()}"

    @property
    def signed_qty(self) -> float:
        return self.qty * self.side.sign

    def _digest(self) -> str:
        canonical = f"{self.ticker}|{self.qty:.{QTY_PRECISION}f}|{self.side.value}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:ID_DIGEST_CHARS]


@dataclass(frozen=True)
class Account:
    # `equity` is the NAV the veto sizes its caps against, and it comes from the broker
    # rather than from our own mark-to-market — same rule as positions. `buying_power` is
    # the broker's own answer, which is not `cash` the moment margin exists.
    cash: float
    equity: float
    buying_power: float


@dataclass(frozen=True)
class OrderAck:
    client_order_id: str
    broker_order_id: str
    accepted_at: pd.Timestamp


@dataclass(frozen=True)
class OrderStatus:
    client_order_id: str
    state: OrderState
    filled_qty: float = 0.0
    avg_fill_price: float | None = None

    @property
    def is_terminal(self) -> bool:
        # What recovery actually asks: is this order still capable of moving the book?
        return self.state in TERMINAL_ORDER_STATES


@dataclass(frozen=True)
class BrokerFill:
    # Deliberately not `backtest.portfolio.Fill`. That one carries the cost model's
    # itemised spread and impact, which are our estimates; this one carries what the venue
    # reported. Merging them would let a modelled number masquerade as an observed one.
    client_order_id: str
    ticker: str
    qty: float           # positive; direction lives in `side`
    side: Side
    fill_price: float
    filled_at: pd.Timestamp
    commission: float = 0.0

    @property
    def signed_qty(self) -> float:
        return self.qty * self.side.sign


# `runtime_checkable` so a fake can be asserted against this in a test. It checks that
# the methods exist, not that they agree on arguments — enough to catch a Protocol that
# grew a method the adapters did not.
@runtime_checkable
class Broker(Protocol):
    # `order_status` and `cancel` are not extras. `order_status` answers "did the order I
    # may have sent before I crashed actually land?"; `cancel` is how the kill switch pulls
    # working orders instead of merely declining to add new ones.

    # Every method may raise BrokerError, and a BrokerError from `submit` means the outcome
    # is unknown rather than negative.

    def positions(self) -> dict[str, float]:          # ticker -> signed shares
        ...

    def account(self) -> Account:
        ...

    def submit(self, intent: OrderIntent) -> OrderAck:
        ...

    def order_status(self, client_order_id: str) -> OrderStatus | None:
        ...

    def cancel(self, client_order_id: str) -> None:
        ...

    def fills(self, since=None) -> list[BrokerFill]:
        ...
