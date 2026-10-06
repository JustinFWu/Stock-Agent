"""
The operator entry point for a live session.

    stock-agent-session                          # dry run against the paper account
    stock-agent-session --submit                 # the same, armed to send orders
    stock-agent-session --status                 # kill switch and the last few sessions
    stock-agent-session --reset-kill-switch NAME # clear a trip, by hand, with a name on it

Deliberately NOT a flag on `stock-agent`. The research CLI reads the bar cache and prints
numbers; this one can place orders at a venue. Sharing an entry point between the two means
one mistyped argument separates "re-run the backtest" from "trade the account", and that is
not a difference to leave to a typo.

`--submit` is required to send anything. The default runs the whole sequence — recovery,
the broker's positions, reconciliation, the tripwires and the veto — and stops before
`submit()`, because everything that can be wrong about a session is wrong before the order
goes out.
"""

import argparse
import sys

from dotenv import load_dotenv

# Before `config` is read, so a .env file at the repository root supplies the Alpaca keys.
# `AlpacaBroker` also re-reads the environment when it is constructed, so this is the
# convenience rather than the mechanism.
load_dotenv()

from stock_agent.config import KILL_SWITCH_PATH, ORDER_LOG_PATH, SESSION_LOG_PATH
from stock_agent.data.universe import UNIVERSE
from stock_agent.execution.fake import FakeBroker
from stock_agent.execution.killswitch import KillSwitch
from stock_agent.execution.session import run_session
from stock_agent.execution.store import OrderStore
from stock_agent.strategy.factory import STRATEGIES, VOL_SOURCES, build_strategy

BROKERS = ("alpaca-paper", "alpaca-live", "fake")


def main() -> int:
    args = _parse_args()

    switch = KillSwitch(KILL_SWITCH_PATH)

    if args.reset_kill_switch:
        state = switch.reset(args.reset_kill_switch, note=args.note)
        print(f"kill switch: {state.describe()}")
        return 0

    if args.status:
        return _print_status(switch)

    broker = _build_broker(args.broker)
    report = run_session(
        broker=broker,
        store=OrderStore(ORDER_LOG_PATH),
        kill_switch=switch,
        strategy=build_strategy(args.strategy, args.vol_source, args.vol_target),
        universe=UNIVERSE,
        as_of=args.as_of,
        submit=args.submit,
        session_log=SESSION_LOG_PATH,
    )

    print(report.describe())

    # A non-zero exit is what a cron line or a systemd timer notices. "Completed but not
    # clean" is a failure for that purpose: the gate counts clean sessions, so anything an
    # operator would want to look at has to be visible without reading the log.
    return 0 if report.is_clean else 1


def _build_broker(name: str):
    if name == "fake":
        # For smoke-testing the runner itself without a venue. It holds nothing and prices
        # nothing, so a session against it reconciles to a full set of buys and refuses
        # them on cash — which is a useful thing to be able to watch happen.
        return FakeBroker(cash=100_000.0, prices={})

    # Imported here rather than at module scope so `--status` and `--reset-kill-switch`
    # work on a machine with no API keys configured.
    from stock_agent.execution.alpaca import AlpacaBroker

    if name == "alpaca-live":
        return AlpacaBroker(paper=False, i_understand_this_is_real_money=True)
    return AlpacaBroker(paper=True)


def _print_status(switch: KillSwitch) -> int:
    print(f"kill switch: {switch.state.describe()}")
    print(f"order log:   {ORDER_LOG_PATH}")

    store = OrderStore(ORDER_LOG_PATH)
    open_orders = store.unresolved()
    print(f"open orders: {len(open_orders)}")
    for record in open_orders:
        print(f"  {record.client_order_id}  {record.state.value}")

    sessions = _tail_sessions(5)
    print(f"sessions:    {len(sessions)} shown from {SESSION_LOG_PATH}")
    for row in sessions:
        flag = "clean" if row.get("clean") else "ATTENTION"
        print(f"  {row.get('session_date')}  {row.get('outcome'):<9} {flag:<9} "
              f"sent {len(row.get('sent', []))}")
    return 0 if not switch.is_tripped else 1


def _tail_sessions(count: int) -> list[dict]:
    import json

    if not SESSION_LOG_PATH.exists():
        return []
    rows = []
    for line in SESSION_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn final line is possible and is not worth refusing to print a status
            # over. The order log repairs its own tail because correctness depends on it;
            # this file is a record, not a source of truth.
            continue
    return rows[-count:]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--broker", choices=BROKERS, default="alpaca-paper",
                        help="Which venue to run against (default alpaca-paper)")
    parser.add_argument("--submit", action="store_true",
                        help="Actually send the orders the veto allows. Off by default.")
    parser.add_argument("--strategy", choices=STRATEGIES, default="vol-momentum",
                        help="Which strategy forms the target (default vol-momentum)")
    parser.add_argument("--vol-source", choices=VOL_SOURCES, default="panel",
                        help="Where vol-momentum gets its forecast (default panel)")
    parser.add_argument("--vol-target", type=float, default=None,
                        help="Annualised portfolio vol target (default from config)")
    parser.add_argument("--as-of", default=None,
                        help="Replay a specific session instead of the latest. This skips "
                             "the staleness and coverage guards, which is why it is "
                             "explicit.")
    parser.add_argument("--status", action="store_true",
                        help="Print the kill switch, open orders and recent sessions")
    parser.add_argument("--reset-kill-switch", metavar="OPERATOR", default=None,
                        help="Clear a trip. Requires the name of whoever is clearing it.")
    parser.add_argument("--note", default="",
                        help="Why the kill switch was reset, recorded with it")

    args = parser.parse_args()
    if args.vol_target is None:
        from stock_agent.config import VOL_TARGET_ANNUAL
        args.vol_target = VOL_TARGET_ANNUAL
    return args


if __name__ == "__main__":
    sys.exit(main())
