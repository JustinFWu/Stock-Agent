import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from stock_agent.config import (
    MAX_DRAWDOWN,
    MAX_LIVE_STALENESS_DAYS,
    MIN_LIVE_COVERAGE,
    NAV_RECONCILIATION_TOLERANCE,
)

# Three separable things, and conflating them is how a kill switch fails. The *tripwire*
# evaluates conditions, the *state* remembers that one fired, and the *enforcement point*
# refuses to trade — which is the veto, in veto.py, not here.

# The state is on disk and clears only by hand. A kill switch held in memory clears itself
# on restart, and the restart is very often caused by whatever tripped it.

# The tripwire and the state share one call (`check`) on purpose, while enforcement stays
# in another module. A tripwire whose result a caller has to remember to act on is not a
# tripwire; an enforcement point that lives inside the thing it enforces cannot be audited.


class TripReason(str, Enum):
    DRAWDOWN = "drawdown"
    STALE_DATA = "stale_data"
    THIN_COVERAGE = "thin_coverage"
    NAV_MISMATCH = "nav_mismatch"
    UNEXPECTED_POSITION = "unexpected_position"


@dataclass(frozen=True)
class Breach:
    reason: TripReason
    detail: str


@dataclass(frozen=True)
class KillSwitchState:
    # `peak_equity` lives here rather than being recomputed because a high-water mark
    # rebuilt from whatever history happens to be on hand is not a high-water mark. It
    # has to survive the restart for the drawdown measurement to mean anything.
    tripped: bool = False
    reason: TripReason | None = None
    detail: str = ""
    tripped_at: str = ""
    peak_equity: float = 0.0
    cleared_by: str = ""
    cleared_at: str = ""

    def describe(self) -> str:
        if not self.tripped:
            return f"armed (peak equity {self.peak_equity:,.2f})"
        return (f"TRIPPED {self.tripped_at} — {self.reason.value if self.reason else '?'}: "
                f"{self.detail}")


def evaluate_tripwires(
    *,
    equity: float,
    peak_equity: float,
    positions: dict[str, float],
    staleness_days: int | None = None,
    coverage: float | None = None,
    nav_disagreement: float | None = None,
    unpriced_positions: tuple[str, ...] = (),
    max_drawdown: float = MAX_DRAWDOWN,
    max_staleness_days: int = MAX_LIVE_STALENESS_DAYS,
    min_coverage: float = MIN_LIVE_COVERAGE,
    nav_tolerance: float = NAV_RECONCILIATION_TOLERANCE,
) -> tuple[Breach, ...]:
    # Pure, and returns every breach rather than the first. Stopping at one would hide the
    # others from the alert, and a drawdown that arrives alongside an unexpected position
    # is a different morning from a drawdown on its own.

    # `None` means not measured, which is not the same as measured-and-fine. A caller with
    # no coverage figure to hand gets no coverage check rather than a silent pass.
    breaches: list[Breach] = []

    if peak_equity > 0:
        drawdown = equity / peak_equity - 1.0
        if drawdown < -max_drawdown:
            breaches.append(Breach(
                TripReason.DRAWDOWN,
                f"equity {equity:,.2f} is {drawdown:.2%} off the {peak_equity:,.2f} peak "
                f"(limit {max_drawdown:.0%})"))

    # `live.py` already refuses to form weights on stale or thin data. It raises, which is
    # a transient failure a retry loop would paper over; this is what makes the same
    # condition persist until a human has looked at it.
    if staleness_days is not None and staleness_days > max_staleness_days:
        breaches.append(Breach(
            TripReason.STALE_DATA,
            f"the most recent session is {staleness_days} days old "
            f"(limit {max_staleness_days})"))

    if coverage is not None and coverage < min_coverage:
        breaches.append(Breach(
            TripReason.THIN_COVERAGE,
            f"only {coverage:.0%} of the universe had a usable bar (need {min_coverage:.0%})"))

    if nav_disagreement is not None and abs(nav_disagreement) > nav_tolerance:
        breaches.append(Breach(
            TripReason.NAV_MISMATCH,
            f"our marks value the book {nav_disagreement:+.2%} away from the broker's "
            f"reported equity (tolerance {nav_tolerance:.2%})"))

    # Nothing in v1 generates a short, so one in the broker's book was not asked for by
    # anything here. That makes it a fact about the system rather than about the market.
    shorts = sorted(t for t, qty in positions.items() if qty < 0)
    if shorts:
        breaches.append(Breach(
            TripReason.UNEXPECTED_POSITION,
            f"the broker reports short positions in {', '.join(shorts)} and nothing in "
            "this system generates a short"))

    # A holding we cannot price is worse than one priced wrongly: it is valued at zero, so
    # it neither shows up in the weights nor moves the NAV gap enough to notice when small.
    if unpriced_positions:
        breaches.append(Breach(
            TripReason.UNEXPECTED_POSITION,
            f"the broker holds {', '.join(unpriced_positions)}, which our data cannot "
            "price — the book being sized is not the book being held"))

    return tuple(breaches)


@dataclass
class KillSwitch:
    path: Path
    max_drawdown: float = MAX_DRAWDOWN
    max_staleness_days: int = MAX_LIVE_STALENESS_DAYS
    min_coverage: float = MIN_LIVE_COVERAGE
    nav_tolerance: float = NAV_RECONCILIATION_TOLERANCE
    _state: KillSwitchState = field(default_factory=KillSwitchState, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def is_tripped(self) -> bool:
        return self._state.tripped

    @property
    def peak_equity(self) -> float:
        return self._state.peak_equity

    def check(self, *, equity: float, positions: dict[str, float],
              staleness_days: int | None = None, coverage: float | None = None,
              nav_disagreement: float | None = None,
              unpriced_positions: tuple[str, ...] = ()) -> tuple[Breach, ...]:
        # Order is load-bearing: the drawdown is measured against the peak as it stood
        # *before* this observation. Updating first would fold today's equity into the
        # high-water mark and measure every drawdown as zero.
        breaches = evaluate_tripwires(
            equity=equity, peak_equity=self._state.peak_equity, positions=positions,
            staleness_days=staleness_days, coverage=coverage,
            nav_disagreement=nav_disagreement, unpriced_positions=unpriced_positions,
            max_drawdown=self.max_drawdown,
            max_staleness_days=self.max_staleness_days, min_coverage=self.min_coverage,
            nav_tolerance=self.nav_tolerance)

        if breaches:
            self.trip(breaches[0].reason, "; ".join(b.detail for b in breaches))

        self._observe_equity(equity)
        return breaches

    def trip(self, reason: TripReason, detail: str) -> KillSwitchState:
        # The first trip wins. A switch that is already down does not need tripping again,
        # and overwriting the reason would replace the original cause with whatever
        # happened next — the consequence, not the diagnosis.
        if self._state.tripped:
            return self._state

        self._write(KillSwitchState(
            tripped=True, reason=reason, detail=detail,
            tripped_at=pd.Timestamp.now("UTC").isoformat(),
            peak_equity=self._state.peak_equity))
        return self._state

    def reset(self, operator: str, note: str = "") -> KillSwitchState:
        # Manual, and nothing in the session path calls this. The operator is required
        # rather than optional because a reset with no name on it is indistinguishable
        # from the automatic clear this whole mechanism exists to prevent.
        if not operator or not operator.strip():
            raise ValueError(
                "resetting the kill switch needs the name of whoever is clearing it — "
                "an unattributed reset is how an automatic one gets added later.")
        if not self._state.tripped:
            return self._state

        self._write(KillSwitchState(
            tripped=False, peak_equity=self._state.peak_equity,
            cleared_by=operator.strip(), cleared_at=pd.Timestamp.now("UTC").isoformat(),
            detail=note))
        return self._state

    def _observe_equity(self, equity: float) -> None:
        # Monotone by construction. A peak that could fall would let a long decline reset
        # its own reference point and read as a series of small, survivable drawdowns.
        if equity > self._state.peak_equity:
            self._write(KillSwitchState(**{**_as_dict(self._state), "peak_equity": float(equity)}))

    def _load(self) -> KillSwitchState:
        if not self.path.exists():
            return KillSwitchState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Deliberately not "assume armed". An unreadable kill switch is a kill switch
            # whose state is unknown, and the safe reading of unknown is stopped.
            raise ValueError(
                f"{self.path} is unreadable, so whether trading is halted is unknown. "
                "Refusing to start; restore or delete it deliberately.") from exc

        reason = raw.get("reason")
        return KillSwitchState(
            tripped=bool(raw.get("tripped", False)),
            reason=TripReason(reason) if reason else None,
            detail=raw.get("detail", ""),
            tripped_at=raw.get("tripped_at", ""),
            peak_equity=float(raw.get("peak_equity", 0.0)),
            cleared_by=raw.get("cleared_by", ""),
            cleared_at=raw.get("cleared_at", ""),
        )

    def _write(self, state: KillSwitchState) -> None:
        # Written whole to a temporary beside the destination and renamed over it, so a
        # crash leaves either the old state or the new one. fsync before the rename for
        # the same reason the order log fsyncs: a trip sitting in the OS buffer has not
        # survived anything, and the crash it is recording may be seconds away.
        payload = _as_dict(state)
        payload["reason"] = state.reason.value if state.reason else None

        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink()
        self._state = state


def _as_dict(state: KillSwitchState) -> dict:
    return {
        "tripped": state.tripped, "reason": state.reason, "detail": state.detail,
        "tripped_at": state.tripped_at, "peak_equity": state.peak_equity,
        "cleared_by": state.cleared_by, "cleared_at": state.cleared_at,
    }
