import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.execution.broker import Broker, BrokerError, DuplicateOrderError, OrderIntent
from src.execution.store import LocalState, OrderRecord, OrderStore

# The two places the order log and the broker have to meet, kept together because they are
# one contract read from both ends: `submit_intent` writes before it sends, and `recover`
# is what makes that write worth having.

# Ordering is the whole of it. Recovery runs before a session generates anything, or the
# session sizes against a position that an unresolved order is still about to change, and
# places a second order on top of the first.


@dataclass(frozen=True)
class RecoveryReport:
    resolved: tuple[OrderRecord, ...] = ()
    abandoned: tuple[OrderRecord, ...] = ()
    working: tuple[OrderRecord, ...] = ()

    @property
    def blocked_tickers(self) -> frozenset[str]:
        # Names a session must not trade: an order is still live on them, so any new size
        # would be computed against a position that is still moving. The reconciler takes
        # this as an exclusion rather than the store deciding policy on its behalf.
        return frozenset(record.intent.ticker for record in self.working)

    @property
    def is_clean(self) -> bool:
        return not self.working


def submit_intent(store: OrderStore, broker: Broker, intent: OrderIntent) -> OrderRecord:
    # Write-ahead, and the order matters more than anything else in this module. Submitting
    # first leaves an order at the broker that our side has no memory of, and the next
    # session places it again.
    store.record_intent(intent)

    try:
        ack = broker.submit(intent)
    except DuplicateOrderError:
        # The broker already holds this exact intent, so recovery missed it or a retry
        # landed twice. Ask what it actually did — assuming either outcome here is how a
        # filled order gets treated as a failure, or the reverse.
        status = broker.order_status(intent.client_order_id)
        if status is None:
            raise BrokerError(
                f"{intent.client_order_id} was refused as a duplicate and the broker then "
                "reported no such order. Those cannot both be true; this needs a human "
                "before anything else is sent.") from None
        return store.record_observation(status)
    except BrokerError:
        # The outcome is unknown, so nothing is written. Recording a failure here is how a
        # filled order becomes invisible; the record stays INTENDED and recovery resolves
        # it against the broker on the next run.
        raise

    return store.record_ack(ack)


def recover(store: OrderStore, broker: Broker) -> RecoveryReport:
    # Also the fill poll. "Ask the broker about every order we have not resolved" is the
    # same operation at startup and at the end of a session, and having one implementation
    # is what keeps the two from disagreeing about what a partial fill means.
    resolved, abandoned, working = [], [], []

    for record in store.unresolved():
        coid = record.client_order_id
        status = broker.order_status(coid)

        if status is None:
            # Safe only because we have no evidence the broker ever had it: the process
            # died between writing the intent and the submit landing.
            if record.known_to_broker:
                raise BrokerError(
                    f"{coid} was acknowledged as {record.broker_order_id} and the broker "
                    "now reports no such order. Refusing to guess whether it filled.")
            abandoned.append(store.record_abandoned(coid, "the broker never had it"))
            continue

        updated = store.record_observation(status)
        (resolved if updated.state is LocalState.RESOLVED else working).append(updated)

    return RecoveryReport(resolved=tuple(resolved), abandoned=tuple(abandoned),
                          working=tuple(working))
