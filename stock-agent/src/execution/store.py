import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.execution.broker import (TERMINAL_ORDER_STATES, OrderAck, OrderIntent, OrderState,
                                  OrderStatus, Side)

# An append-only event log, folded into a view of what we believe about each order. Two
# state machines exist and they are not the same: the broker's OrderState is what the venue
# says, LocalState is what our side knows. The gap between them is the entire problem.

# Append-only rather than a rewritten snapshot, because a crash during a rewrite loses the
# whole file — at precisely the moment the file is the only thing that knows an order was
# sent. Appending can lose at most the last line, and that line is repairable.

# The store never talks to a broker. It is a log, not a client; `recovery.py` owns every
# interaction that needs both, including the write-before-submit ordering.


class LocalState(str, Enum):
    # INTENDED is the dangerous state and the reason the log exists: an intent was written
    # and the outcome is unknown. It is never a resting state — recovery must resolve it
    # before the next session generates anything.
    INTENDED = "intended"
    SUBMITTED = "submitted"      # known to the broker, still working
    RESOLVED = "resolved"        # terminal outcome observed and recorded
    ABANDONED = "abandoned"      # the broker never had it; nothing happened


OPEN_STATES = frozenset({LocalState.INTENDED, LocalState.SUBMITTED})


@dataclass
class OrderRecord:
    intent: OrderIntent
    state: LocalState = LocalState.INTENDED
    broker_order_id: str | None = None
    broker_state: OrderState | None = None
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    note: str = ""

    # Positive evidence that the broker holds this order, from an ack *or* from a status
    # that came back. Without it, an order observed as working but never acked would look
    # like one that never landed, and recovery would write it off.
    known_to_broker: bool = False

    @property
    def client_order_id(self) -> str:
        return self.intent.client_order_id

    @property
    def is_open(self) -> bool:
        return self.state in OPEN_STATES


@dataclass
class OrderStore:
    path: Path
    _records: dict[str, OrderRecord] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()
        self._records = self._load()

    # --- Writes ----------------------------------------------------------------------

    def record_intent(self, intent: OrderIntent) -> OrderRecord:
        # Refuses a repeat rather than overwriting. A second intent under an id the log
        # already knows means recovery did not resolve the first one, and silently
        # replacing the record is how the evidence of that disappears.
        coid = intent.client_order_id
        if coid in self._records:
            raise ValueError(
                f"{coid} is already in the order log as {self._records[coid].state.value} — "
                "resolve it through recovery before intending it again.")

        self._append({"event": "intended", "client_order_id": coid,
                      "session_date": f"{intent.session_date:%Y-%m-%d}", "ticker": intent.ticker,
                      "qty": intent.qty, "side": intent.side.value})
        return self._records[coid]

    def record_ack(self, ack: OrderAck) -> OrderRecord:
        self._append({"event": "submitted", "client_order_id": ack.client_order_id,
                      "broker_order_id": ack.broker_order_id})
        return self._records[ack.client_order_id]

    def record_observation(self, status: OrderStatus) -> OrderRecord:
        self._append({"event": "observed", "client_order_id": status.client_order_id,
                      "state": status.state.value, "filled_qty": status.filled_qty,
                      "avg_fill_price": status.avg_fill_price})
        return self._records[status.client_order_id]

    def record_abandoned(self, client_order_id: str, reason: str) -> OrderRecord:
        self._append({"event": "abandoned", "client_order_id": client_order_id,
                      "reason": reason})
        return self._records[client_order_id]

    # --- Reads -----------------------------------------------------------------------

    def get(self, client_order_id: str) -> OrderRecord | None:
        return self._records.get(client_order_id)

    def records(self) -> dict[str, OrderRecord]:
        return dict(self._records)

    def unresolved(self) -> list[OrderRecord]:
        # What recovery has to work through before a session may generate anything.
        return [r for r in self._records.values() if r.is_open]

    # --- Internals -------------------------------------------------------------------

    def _append(self, event: dict) -> None:
        # fsync, not just flush. A write-ahead log that sits in the OS buffer is not
        # ahead of anything — the crash it is meant to survive takes the buffer with it.
        event = {"at": pd.Timestamp.now("UTC").isoformat(), **event}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _apply(event, self._records)

    def _load(self) -> dict[str, OrderRecord]:
        # The same `_apply` the live path uses, so the in-memory view after a write and the
        # view after a restart cannot disagree. Two fold implementations is how a log that
        # replays correctly in a test replays differently in production.
        records: dict[str, OrderRecord] = {}
        if not self.path.exists():
            return records

        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}:{number} is not readable JSON. A torn tail is "
                                 "repaired on open, so damage here is corruption in the "
                                 "middle of the log and needs a human.") from exc
            _apply(event, records)
        return records

    def _repair_tail(self) -> None:
        # A crash between write and fsync can leave a line with no newline. Appending after
        # it would splice the next event onto the fragment and corrupt a line that parsed
        # fine before, so the fragment is dropped on open instead.
        if not self.path.exists() or self.path.stat().st_size == 0:
            return

        raw = self.path.read_bytes()
        if raw.endswith(b"\n"):
            return
        keep = raw.rfind(b"\n") + 1
        self.path.write_bytes(raw[:keep])


def _apply(event: dict, records: dict[str, OrderRecord]) -> None:
    kind = event["event"]
    coid = event["client_order_id"]

    if kind == "intended":
        intent = OrderIntent(session_date=event["session_date"], ticker=event["ticker"],
                             qty=event["qty"], side=Side(event["side"]))
        # The id is derived, so a change to the digest recipe silently re-keys every log
        # written before it and the next session resubmits the lot. Checking the replay
        # against the id that was stored turns that into a refusal to start.
        if intent.client_order_id != coid:
            raise ValueError(
                f"{coid} replays as {intent.client_order_id}. The client_order_id recipe "
                "has changed since this log was written, so nothing in it can be matched "
                "against the broker.")
        records[coid] = OrderRecord(intent=intent)
        return

    record = records.get(coid)
    if record is None:
        raise ValueError(f"{coid}: a {kind!r} event with no intent before it — the log is "
                         "out of order, which means something wrote to it out of band.")

    if kind == "submitted":
        record.broker_order_id = event["broker_order_id"]
        record.known_to_broker = True
        if record.state is LocalState.INTENDED:
            record.state = LocalState.SUBMITTED
    elif kind == "observed":
        record.broker_state = OrderState(event["state"])
        record.filled_qty = event["filled_qty"]
        record.avg_fill_price = event["avg_fill_price"]
        record.known_to_broker = True
        record.state = (LocalState.RESOLVED if record.broker_state in TERMINAL_ORDER_STATES
                        else LocalState.SUBMITTED)
    elif kind == "abandoned":
        record.state = LocalState.ABANDONED
        record.note = event.get("reason", "")
    else:
        raise ValueError(f"{coid}: unknown event {kind!r} in the order log")
