import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

from stock_agent.execution.broker import (
    Account,
    BrokerError,
    BrokerFill,
    DuplicateOrderError,
    OrderAck,
    OrderIntent,
    OrderState,
    OrderStatus,
    Side,
)

# The only file in the repo that knows Alpaca exists. Everything above it is written in the
# vocabulary in broker.py, which is why this module is thin enough to read in one sitting:
# six methods, each one request, plus the mapping from the venue's words to ours.

# No SDK, on purpose. `alpaca-py` would be a reasonable choice, but the Protocol is six
# calls and the part that has to be right is not the HTTP — it is the mapping below, which
# an SDK would hide rather than remove. stdlib keeps the dependency list at zero and keeps
# every venue assumption in one readable place.

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

# Alpaca's order lifecycle, mapped onto the five states the order store understands.
# Anything absent from this table is treated as PENDING — see `_state_of` for why that is
# the safe default and why it is still alert-worthy.
_TERMINAL_STATES = {
    "filled": OrderState.FILLED,
    "canceled": OrderState.CANCELED,
    "cancelled": OrderState.CANCELED,
    "expired": OrderState.CANCELED,
    "done_for_day": OrderState.CANCELED,
    "rejected": OrderState.REJECTED,
    "suspended": OrderState.REJECTED,
}
_OPEN_STATES = {
    "partially_filled": OrderState.PARTIALLY_FILLED,
    "new": OrderState.PENDING,
    "accepted": OrderState.PENDING,
    "pending_new": OrderState.PENDING,
    "accepted_for_bidding": OrderState.PENDING,
    "pending_cancel": OrderState.PENDING,
    "pending_replace": OrderState.PENDING,
    "replaced": OrderState.PENDING,
    "calculated": OrderState.PENDING,
    "stopped": OrderState.PENDING,
}


@dataclass
class AlpacaBroker:
    # `paper` defaults to True and the live endpoint needs more than flipping it — see
    # `__post_init__`. Phase 4's own card says passing its gate is a plumbing result and
    # not permission for live capital, so the default has to be the one that cannot lose
    # money even if every other guard in the repo is wrong.
    paper: bool = True

    # Empty means "read the environment when the adapter is built", not "no key". Taking
    # the default from `config` at import time would make the keys depend on whether
    # `load_dotenv()` had run before this module was first imported, which is an
    # import-order bug waiting for the one session that imports things in a new order.
    api_key: str = ""
    secret_key: str = ""
    timeout: float = 10.0

    # Set only by an operator who has read what it does. Pointing this adapter at real
    # money is a decision that deserves a name in a diff, not a flag in a cron line.
    i_understand_this_is_real_money: bool = False

    base_url: str = field(init=False)

    # Statuses the venue returned that this adapter's table does not name. Collected on the
    # instance rather than logged in place, so the session runner can raise them as alerts
    # without this module picking a logging mechanism for the whole repo.
    unknown_statuses: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.paper and not self.i_understand_this_is_real_money:
            raise ValueError(
                "AlpacaBroker(paper=False) also needs "
                "i_understand_this_is_real_money=True. Phase 3 failed its gate, so there "
                "is no measured edge for this to trade, and passing Phase 4 is a plumbing "
                "result rather than permission for capital.")

        self.api_key = self.api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = self.secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are empty. Put them in the "
                "environment or a .env file at the repository root; paper keys from a "
                "paper account are what Phase 4's thirty sessions are meant to use.")
        self.base_url = PAPER_BASE_URL if self.paper else LIVE_BASE_URL

    # ----- the Protocol -------------------------------------------------------------

    def positions(self) -> dict[str, float]:
        # Signed, and the sign comes from the venue rather than from a side we remember.
        # Alpaca reports a short as a negative qty, which is what the tripwire's
        # unexpected-position check reads.
        rows = self._request("GET", "/v2/positions")
        return {str(row["symbol"]): float(row["qty"]) for row in rows if float(row["qty"]) != 0}

    def account(self) -> Account:
        row = self._request("GET", "/v2/account")

        # `trading_blocked` is the venue's own kill switch and it is not ours to override.
        # Letting a session proceed to submit against a blocked account would turn one
        # clear refusal into a sequence of per-order rejections.
        if row.get("trading_blocked") or row.get("account_blocked"):
            raise BrokerError(
                f"Alpaca reports the account as blocked (status {row.get('status')!r}). "
                "Nothing will be submitted.")

        return Account(cash=float(row["cash"]), equity=float(row["equity"]),
                       buying_power=float(row["buying_power"]))

    def submit(self, intent: OrderIntent) -> OrderAck:
        # `client_order_id` is the idempotency key and it is computed from the intent, so a
        # resubmit of the same intent is refused by the venue rather than filled twice.
        payload = {
            "symbol": intent.ticker,
            "qty": f"{intent.qty:f}",
            "side": _venue_side(intent.side),
            "type": "market",
            "time_in_force": "day",
            "client_order_id": intent.client_order_id,
        }
        row = self._request("POST", "/v2/orders", payload)
        return OrderAck(
            client_order_id=intent.client_order_id,
            broker_order_id=str(row["id"]),
            accepted_at=_timestamp(row.get("submitted_at") or row.get("created_at")),
        )

    def order_status(self, client_order_id: str) -> OrderStatus | None:
        # The question recovery asks after a crash: did the order I may have sent land? A
        # 404 is the one honest "no", and it is the only case that returns None.
        row = self._request("GET", "/v2/orders:by_client_order_id",
                            query={"client_order_id": client_order_id}, none_on_404=True)
        if row is None:
            return None

        filled_qty = float(row.get("filled_qty") or 0.0)
        avg_price = row.get("filled_avg_price")
        return OrderStatus(
            client_order_id=client_order_id,
            state=self._state_of(str(row.get("status", "")), client_order_id),
            filled_qty=filled_qty,
            avg_fill_price=float(avg_price) if avg_price not in (None, "") else None,
        )

    def cancel(self, client_order_id: str) -> None:
        # Two calls, because Alpaca cancels by its own id. An order it no longer knows is
        # not an error here: cancel is what the kill switch calls to pull working orders,
        # and "already gone" is the outcome it wanted.
        row = self._request("GET", "/v2/orders:by_client_order_id",
                            query={"client_order_id": client_order_id}, none_on_404=True)
        if row is None:
            return
        self._request("DELETE", f"/v2/orders/{row['id']}", none_on_404=True)

    def fills(self, since=None) -> list[BrokerFill]:
        query = {"activity_types": "FILL"}
        if since is not None:
            query["after"] = pd.Timestamp(since).isoformat()

        rows = self._request("GET", "/v2/account/activities", query=query)
        return [self._fill(row) for row in rows if row.get("order_id")]

    # ----- plumbing -----------------------------------------------------------------

    def _fill(self, row: dict) -> BrokerFill:
        # The activities feed reports `side` as the venue saw it, which collapses our four
        # sides to two. Reconstructing the original is not possible and not needed: a fill
        # is an observation, and what it has to carry is quantity, price and direction.
        side = Side.BUY if str(row.get("side", "")).startswith("buy") else Side.SELL
        return BrokerFill(
            client_order_id=str(row.get("client_order_id") or row["order_id"]),
            ticker=str(row["symbol"]),
            qty=abs(float(row["qty"])),
            side=side,
            fill_price=float(row["price"]),
            filled_at=_timestamp(row.get("transaction_time")),
        )

    def _state_of(self, venue_status: str, client_order_id: str) -> OrderState:
        status = venue_status.strip().lower()
        if status in _TERMINAL_STATES:
            return _TERMINAL_STATES[status]
        if status in _OPEN_STATES:
            return _OPEN_STATES[status]

        # An unrecognised status maps to PENDING, and the choice is deliberate. PENDING is
        # not terminal, so recovery keeps asking about the order and the reconciler keeps
        # the name deferred — the conservative reading. Mapping an unknown word to FILLED
        # would have a session size against a position that may not exist; raising would
        # halt trading because a venue added a vocabulary word. So: proceed safely, and
        # make sure a human hears about it.
        self.unknown_statuses.append(
            f"{client_order_id}: unrecognised Alpaca status {venue_status!r}, "
            "treated as pending")
        return OrderState.PENDING

    def drain_unknown_statuses(self) -> tuple[str, ...]:
        drained = tuple(self.unknown_statuses)
        self.unknown_statuses.clear()
        return drained

    def _request(self, method: str, path: str, body: dict | None = None,
                 query: dict | None = None, none_on_404: bool = False):
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"

        request = urllib.request.Request(
            url, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
            })

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            # A refusal the venue stated. Readable, and distinguishable from silence.
            detail = _error_detail(exc)
            if exc.code == 404 and none_on_404:
                return None
            if _is_duplicate(exc.code, detail):
                raise DuplicateOrderError(
                    f"Alpaca already holds this client_order_id: {detail}") from exc
            raise BrokerError(f"{method} {path} failed with {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            # Silence, and the one case the whole write-ahead design exists for. The
            # outcome is UNKNOWN, not negative: the order may be live at the venue. Saying
            # so as a BrokerError is what makes `submit_intent` record nothing and re-raise,
            # leaving the next session's `order_status` to settle it.
            raise BrokerError(
                f"{method} {path} gave no usable answer ({exc!r}). The outcome is unknown "
                "and must not be read as a failure to submit.") from exc


def _venue_side(side: Side) -> str:
    # Alpaca takes two sides and infers short-selling and covering from the position. Our
    # four-way vocabulary collapses here and nowhere else, which is exactly the branch the
    # Phase 4 sketch predicted: widening `Side` stayed cheap because the venue mapping is
    # one function, not a signature everything upstream depends on.
    return "buy" if side.sign > 0 else "sell"


def _is_duplicate(code: int, detail: str) -> bool:
    # Alpaca signals a duplicate client_order_id with a 422/403 and a message naming it.
    # Matched on the message as well as the code because the code alone has moved between
    # API revisions, and a duplicate misread as a generic failure is the one error that
    # would make a crash-recovered session submit twice.
    lowered = detail.lower()
    return code in (403, 409, 422) and (
        "client_order_id" in lowered and ("exist" in lowered or "unique" in lowered
                                          or "duplicate" in lowered))


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read())
    except (json.JSONDecodeError, OSError, ValueError):
        return exc.reason if isinstance(exc.reason, str) else repr(exc.reason)
    if isinstance(payload, dict):
        return str(payload.get("message") or payload)
    return str(payload)


def _timestamp(value) -> pd.Timestamp:
    # A missing timestamp becomes now rather than NaT: an ack with no usable time is still
    # an ack, and NaT would propagate into the order log as a field nothing can compare.
    if value in (None, ""):
        return pd.Timestamp.now("UTC")
    return pd.Timestamp(value)
