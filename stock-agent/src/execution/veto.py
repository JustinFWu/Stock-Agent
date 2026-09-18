import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MAX_GROSS, MAX_SECTOR_WEIGHT, MAX_WEIGHT
from src.backtest.costs import MAX_CREDIBLE_PARTICIPATION
from src.data.universe import UniverseSpec
from src.execution.broker import Account, OrderIntent

# The hard, auditable, non-negotiable layer the roadmap's architecture describes, and the
# thing the repo did not have. Every risk limit before this one was enforced at weight
# formation — once, inside `target_weights`, on a vector. Nothing ever checked the book
# that actually came back.

# That gap is invisible in a backtest and load-bearing live, because the realised book and
# the target diverge by two mechanisms that are there on purpose — buys scaled down when
# cash is short, blocked names deferred to the next open — plus ordinary drift between
# monthly rebalances. A name can be well through 10% with nothing objecting.

# Two properties, fixed here rather than discovered later:

# It REJECTS, NEVER RESIZES. That is the roadmap's own design test: if a limit becomes a
# penalty term inside the optimizer it has stopped being a risk control and become a
# suggestion. A veto that quietly shrinks an order is negotiating. This is also where the
# live path deliberately differs from the backtester — `_affordable_scale` scales buys down
# to fit the cash, because in a backtest that is an accounting convenience. Live, an order
# the account cannot pay for means our arithmetic and the venue's disagree, and that is
# worth an alert rather than a smaller order.

# It is an ASSERTION, NOT A FILTER. `target_weights()` already returns compliant weights,
# so in normal operation nothing here should ever fire. When it does, something upstream is
# wrong — a partial fill, a deferred name, drift since the last rebalance. A rejection is an
# alert-worthy event, not routine flow control, and that is what makes the layer worth
# having given the caps are nominally applied already.


class Rejection(str, Enum):
    KILL_SWITCH = "kill_switch"
    UNPRICED = "unpriced"
    WOULD_OPEN_SHORT = "would_open_short"
    ORDER_TOO_LARGE = "order_too_large"
    PER_NAME_CAP = "per_name_cap"
    SECTOR_CAP = "sector_cap"
    GROSS_CAP = "gross_cap"
    INSUFFICIENT_CASH = "insufficient_cash"


@dataclass(frozen=True)
class Rejected:
    intent: OrderIntent
    reason: Rejection
    detail: str


@dataclass(frozen=True)
class VetoResult:
    allowed: tuple[OrderIntent, ...] = ()
    rejected: tuple[Rejected, ...] = ()

    @property
    def is_clean(self) -> bool:
        # The expected answer on every ordinary session. False is the alert.
        return not self.rejected

    def describe(self) -> str:
        if self.is_clean:
            return f"veto: {len(self.allowed)} intent(s) allowed, nothing rejected"
        lines = [f"veto: {len(self.allowed)} allowed, {len(self.rejected)} REJECTED"]
        lines += [f"  {r.intent.ticker:<6} {r.intent.side.value:<13} "
                  f"{r.reason.value:<18} {r.detail}" for r in self.rejected]
        return "\n".join(lines)


def veto(
    intents,
    *,
    positions: dict[str, float],
    account: Account,
    marks: pd.Series,
    universe: UniverseSpec,
    kill_switch_tripped: bool = False,
    adv_notional: pd.Series | None = None,
    max_weight: float = MAX_WEIGHT,
    max_gross: float = MAX_GROSS,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
    max_participation: float = MAX_CREDIBLE_PARTICIPATION,
) -> VetoResult:
    # Checked against the POST-TRADE book computed from live broker positions, not against
    # the intents in isolation. "Would this order leave the account in a state the risk
    # limits forbid" is the only question that means anything; "is this order itself large"
    # is not, because the position it lands on top of is what breaches a cap.
    intents = tuple(intents)
    if kill_switch_tripped:
        return VetoResult(rejected=tuple(
            Rejected(i, Rejection.KILL_SWITCH,
                     "the kill switch is tripped and clears only by hand") for i in intents))

    _check_limits(max_weight, max_gross, max_sector_weight, max_participation)
    nav = float(account.equity)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError(f"cannot veto against equity of {account.equity!r}")

    held = {t: float(q) for t, q in positions.items() if q != 0}
    rejected: list[Rejected] = []

    # Pass one: what can be judged one intent at a time. A name is checked against its own
    # post-trade position, so a sell that reduces an overweight holding passes even while
    # the holding is still over the cap — rejecting it would pin the breach in place.
    survivors = []
    for intent in intents:
        fault = _individual_fault(intent, held, marks, nav, adv_notional,
                                 max_weight, max_participation)
        (rejected.append(fault) if fault is not None else survivors.append(intent))

    # Pass two: what only the whole book can answer, recomputed on the survivors. Order
    # matters — a rejection in pass one removes a sell, and the gross the remaining buys
    # land on is then a different number.
    survivors, group_faults = _aggregate_faults(
        survivors, held, marks, nav, account, universe,
        max_gross, max_sector_weight)
    rejected.extend(group_faults)

    return VetoResult(allowed=tuple(survivors), rejected=tuple(rejected))


def _individual_fault(intent: OrderIntent, held: dict[str, float], marks: pd.Series,
                      nav: float, adv_notional, max_weight: float,
                      max_participation: float) -> Rejected | None:
    price = float(marks.get(intent.ticker, np.nan))
    if not np.isfinite(price) or price <= 0:
        # An order that cannot be valued cannot be checked against any limit, and passing
        # it through unchecked is exactly what this layer exists to stop.
        return Rejected(intent, Rejection.UNPRICED,
                        f"no usable mark for {intent.ticker}, so no limit can be applied")

    before = held.get(intent.ticker, 0.0)
    after = before + intent.signed_qty

    # The real long-only check, and it is on the post-trade quantity rather than on the
    # side. A plain SELL of more than the book holds opens a short just as surely as a
    # SELL_SHORT does, and only this test catches both. Covering an existing short is
    # allowed: `after > before` while still negative is a position getting smaller.
    if after < 0 and after < before:
        return Rejected(intent, Rejection.WOULD_OPEN_SHORT,
                        f"would leave {after:,.4f} shares of {intent.ticker}; v1 is long-only")

    if adv_notional is not None:
        adv = float(adv_notional.get(intent.ticker, np.nan))
        if np.isfinite(adv) and adv > 0:
            participation = intent.qty * price / adv
            if participation >= max_participation:
                # A caveat in the backtester, a refusal here. An order this size against a
                # name's daily volume is more likely a sizing bug than a decision, and the
                # cost model it would be priced with was never fitted this far out.
                return Rejected(intent, Rejection.ORDER_TOO_LARGE,
                                f"{participation:.1%} of average daily volume "
                                f"(limit {max_participation:.0%})")

    weight_after = abs(after) * price / nav
    weight_before = abs(before) * price / nav
    if weight_after > max_weight and weight_after > weight_before:
        return Rejected(intent, Rejection.PER_NAME_CAP,
                        f"{intent.ticker} would reach {weight_after:.2%} of equity "
                        f"(cap {max_weight:.0%}, currently {weight_before:.2%})")

    return None


def _aggregate_faults(intents, held: dict[str, float], marks: pd.Series, nav: float,
                      account: Account, universe: UniverseSpec, max_gross: float,
                      max_sector_weight: float):
    # Rejects every exposure-increasing intent in a breaching group, and nothing else. It
    # is blunt, and deliberately so: choosing which subset to drop in order to land just
    # under a ceiling is resizing by another name. Refusing to add exposure is the response
    # that does not negotiate, and in normal operation the branch never runs.
    if not intents:
        return list(intents), []

    faults: list[Rejected] = []
    surviving = list(intents)

    for check in (_gross_breach, _sector_breach, _cash_breach):
        offenders = check(surviving, held, marks, nav, account, universe,
                          max_gross, max_sector_weight)
        if not offenders:
            continue
        reason, detail, culprits = offenders
        faults.extend(Rejected(i, reason, detail) for i in culprits)
        surviving = [i for i in surviving if i not in culprits]

    return surviving, faults


def _post_trade(intents, held: dict[str, float]) -> dict[str, float]:
    book = dict(held)
    for intent in intents:
        book[intent.ticker] = book.get(intent.ticker, 0.0) + intent.signed_qty
    return book


def _increases_exposure(intent: OrderIntent, held: dict[str, float]) -> bool:
    # Side-agnostic on purpose: BUY_TO_COVER has a positive sign and *reduces* the gross
    # book, so keying off the side would reject the one order that fixes a short.
    before = held.get(intent.ticker, 0.0)
    return abs(before + intent.signed_qty) > abs(before)


def _value(book: dict[str, float], marks: pd.Series) -> dict[str, float]:
    values = {}
    for ticker, qty in book.items():
        price = float(marks.get(ticker, np.nan))
        if np.isfinite(price) and qty != 0:
            values[ticker] = abs(qty) * price
    return values


def _gross_breach(intents, held, marks, nav, account, universe, max_gross, max_sector_weight):
    gross = sum(_value(_post_trade(intents, held), marks).values()) / nav
    if gross <= max_gross:
        return None
    culprits = [i for i in intents if _increases_exposure(i, held)]
    if not culprits:
        return None
    return (Rejection.GROSS_CAP,
            f"the post-trade book would be {gross:.1%} gross (cap {max_gross:.0%})",
            culprits)


def _sector_breach(intents, held, marks, nav, account, universe, max_gross, max_sector_weight):
    book = _post_trade(intents, held)
    values = _value(book, marks)
    for group, members in universe.sector_groups(sorted(values)).items():
        exposure = sum(values[t] for t in members) / nav
        if exposure <= max_sector_weight:
            continue
        culprits = [i for i in intents
                    if i.ticker in set(members) and _increases_exposure(i, held)]
        if culprits:
            return (Rejection.SECTOR_CAP,
                    f"{group} would reach {exposure:.1%} of equity "
                    f"(cap {max_sector_weight:.0%})",
                    culprits)
    return None


def _cash_breach(intents, held, marks, nav, account, universe, max_gross, max_sector_weight):
    # Sale proceeds are counted as available, which assumes they settle in time to fund the
    # same session's buys — true of the margin account Alpaca opens, not of a cash account.
    # Costs are not modelled: the venue's own buying-power figure already carries them, and
    # re-estimating them here would be this layer inventing a number to refuse against.
    spend, proceeds = 0.0, 0.0
    for intent in intents:
        price = float(marks.get(intent.ticker, np.nan))
        if not np.isfinite(price):
            continue
        notional = intent.qty * price
        if intent.signed_qty > 0:
            spend += notional
        else:
            proceeds += notional

    available = float(account.buying_power) + proceeds
    if spend <= available:
        return None
    culprits = [i for i in intents if i.signed_qty > 0]
    if not culprits:
        return None
    return (Rejection.INSUFFICIENT_CASH,
            f"buys need {spend:,.2f} against {available:,.2f} available "
            f"({account.buying_power:,.2f} buying power plus {proceeds:,.2f} of proceeds)",
            culprits)


def _check_limits(max_weight: float, max_gross: float, max_sector_weight: float,
                  max_participation: float) -> None:
    # The same NaN trap `weights.py` and `engine.py` already guard: a NaN ceiling fails
    # every comparison, so the limit is not applied and the result looks like a book that
    # respected it. In a risk veto that failure mode is the whole point of the module.
    for name, value in (("max_weight", max_weight), ("max_gross", max_gross),
                        ("max_sector_weight", max_sector_weight),
                        ("max_participation", max_participation)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")
