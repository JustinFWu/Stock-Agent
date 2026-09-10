"""
Weight formation — the one path both the backtest and the live runner take.

This module exists to satisfy the Phase 2 gate: *one* function produces the
backtest weights and the live weights, proven by a test asserting identical
output for a fixed date. The reason that gate matters is that a backtest and a
live system which form positions in two different places will drift apart, and
the drift is invisible — the backtest keeps reporting the strategy you designed
while production trades something else.

The split of responsibility:

  strategy    decides *direction and relative conviction* — which names, how much
              of each, relative to one another — and declares what scale those
              numbers are on. Phase 3's 12-2 momentum plus vol-targeted sizing
              lives behind this interface.
  this module decides *what is legal* — the history requirement, the per-name
              cap, the gross exposure ceiling, and the numeric normalisation.

Constraints live here rather than inside each strategy so a new strategy cannot
forget them, and so the risk limits are auditable in one place.
"""

import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MAX_GROSS, MAX_WEIGHT, MIN_HISTORY_DAYS
from src.data.panel import PricePanel
from src.data.universe import UniverseSpec

# Weights are rounded before they leave this function. Without it, two runs that
# are mathematically identical can differ in the last bit through a different
# summation order, and the parity test that guards this whole design would fail
# for a reason that has nothing to do with the strategy. Ten decimals is far
# finer than any position size that survives contact with a broker.
WEIGHT_PRECISION = 10

# What the numbers in a proposal mean. See `Strategy.proposal_scale`.
PROPOSAL_SCALES = ("absolute", "relative")


class Strategy(Protocol):
    """
    A strategy proposes weights and nothing else.

    It receives a panel already truncated to `as_of`, so it cannot see the
    future even if it tries, and a candidate list already filtered for
    tradability. It returns a Series indexed by ticker; anything missing,
    non-finite or non-positive is treated as "no position".

    `proposal_scale` says what those numbers *mean*. It is required rather than
    defaulted because the two readings cannot be told apart from the numbers:

      "absolute"  already fractions of NAV. A proposal summing to 0.7 means 70%
                  invested and 30% cash, deliberately, and nothing downstream
                  scales it back up. This is how volatility targeting expresses
                  "the market is dangerous right now, hold cash".
      "relative"  unitless conviction. Only the ratios between names carry
                  information; the magnitude is an artefact of whatever the
                  scores were computed from, so the shape is normalised to the
                  gross ceiling before the limits are applied.

    Undeclared, the same conviction *shape* lands at a different invested
    fraction purely because of the magnitude it happened to be computed at —
    raw inverse-volatility scores sum to whatever they sum to — and "70%
    invested because the sizer asked for it" is indistinguishable in the output
    from "70% invested because the scores came out small". One of those is a
    risk decision and the other is an accident, so the strategy has to say which.
    """

    name: str
    proposal_scale: str

    def propose(self, as_of: pd.Timestamp, history: PricePanel,
                candidates: list[str]) -> pd.Series:
        ...


def target_weights(
    as_of,
    panel: PricePanel,
    strategy: Strategy,
    *,
    universe: UniverseSpec,
    min_history: int = MIN_HISTORY_DAYS,
    max_weight: float = MAX_WEIGHT,
    max_gross: float = MAX_GROSS,
) -> pd.Series:
    """
    Portfolio weights for `as_of`, formed only from data available at `as_of`.

    Call this from the backtest with a full-history panel and from the live
    runner with a panel ending today; `panel.as_of` guarantees the strategy sees
    the same shape of information in both cases. The returned Series is sorted by
    ticker and rounded, so two calls that should agree are directly comparable.
    """
    _check_limits(max_weight, max_gross)

    as_of = pd.Timestamp(as_of)
    history = panel.as_of(as_of)

    # Membership and tradability are different questions. `members_asof` answers
    # "was this name in the index then"; `tradable_as_of` answers "did it have a
    # bar then". Today the first is a no-op because the universe carries no
    # membership history — this intersection is where a real one plugs in.
    tradable = history.tradable_as_of(as_of, min_history=min_history)
    candidates = sorted(set(tradable) & set(universe.members_asof(as_of)))
    if not candidates:
        return _empty_weights()

    proposed = strategy.propose(as_of, history, candidates)
    return _apply_constraints(proposed, candidates, _proposal_scale(strategy),
                              max_weight, max_gross)


def _check_limits(max_weight: float, max_gross: float) -> None:
    """
    Reject limits that the constraint step cannot enforce meaningfully.

    A negative ceiling is the sharp case. `_apply_constraints` drops non-positive
    *proposals* and then scales what is left to the gross ceiling, so a negative
    ceiling flips every surviving weight through zero and a long-only path emits
    shorts: with `max_weight=-0.1`, a proposal of 1.0 came back as -0.1. NaN is the
    quiet case — it fails every comparison, so the limit is simply not applied and
    the output looks like a portfolio that respected it.

    Zero is allowed. "Hold nothing" is a legitimate instruction; "hold a negative
    amount of nothing" is not.
    """
    for name, value in (("max_weight", max_weight), ("max_gross", max_gross)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if value < 0:
            raise ValueError(
                f"{name} must not be negative, got {value}. A negative ceiling turns "
                "positive proposals into short positions on a long-only path.")


def _proposal_scale(strategy: Strategy) -> str:
    """
    What scale this strategy's proposals are on, refusing to guess.

    Raising on an undeclared scale is the point. A default would be silently wrong
    for half of all strategies, and wrong in a way that produces a plausible
    portfolio rather than an error — which is how the mistake survives review.
    """
    scale = getattr(strategy, "proposal_scale", None)
    if scale not in PROPOSAL_SCALES:
        raise ValueError(
            f"{getattr(strategy, 'name', type(strategy).__name__)!r} must declare "
            f"proposal_scale as one of {list(PROPOSAL_SCALES)}, got {scale!r}. "
            "'absolute' means the proposal is already fractions of NAV and a sum "
            "below 1 is a deliberate cash position; 'relative' means only the "
            "ratios matter and the shape gets normalised to the gross ceiling."
        )
    return scale


def _apply_constraints(proposed: pd.Series, candidates: list[str], scale: str,
                       max_weight: float, max_gross: float) -> pd.Series:
    """
    Enforce the risk limits and normalise, without ever inventing exposure.

    What happens to the gross depends on what the strategy said its numbers mean.
    An "absolute" proposal is scaled *down* when it breaches the ceiling and never
    up — a strategy that asked for 40% invested meant it. A "relative" proposal is
    normalised to the ceiling in either direction, because its magnitude carries no
    information to preserve: leaving it alone would set the invested fraction from
    the arbitrary units the conviction scores happened to come out in.

    That is not the risk layer inventing exposure, because a relative strategy
    never expressed any. A strategy that wants to hold cash says so on the
    absolute scale, where the request is legible as a request.

    The per-name cap is a clip rather than a redistribution: redistributing a
    capped name's excess would push size into names the strategy wanted less of,
    which is the risk layer overruling the signal instead of bounding it. The clip
    can therefore leave the book below the ceiling, which is correct — the cap is a
    limit, not a target.

    Order matters, and the intuitive order is wrong. Clipping before scaling
    destroys the signal whenever a strategy expresses conviction on a scale other
    than "fractions of the portfolio": a proposal of raw inverse-volatility scores
    is entirely above the cap, so every name clips to exactly `max_weight` and the
    subsequent scaling turns the result into a perfectly equal-weight portfolio —
    silently, with no error and a plausible-looking output. Scaling to the gross
    ceiling first preserves the relative shape; the cap then binds only on names
    that are genuinely oversized.
    """
    weights = pd.Series(proposed, dtype=float).reindex(candidates)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if weights.empty:
        return _empty_weights()

    gross = weights.sum()
    if scale == "relative" or gross > max_gross:
        weights = weights * (max_gross / gross)

    weights = weights.clip(upper=max_weight)

    return weights.sort_index().round(WEIGHT_PRECISION)


def _empty_weights() -> pd.Series:
    """An all-cash portfolio. Same dtype and index type as a populated one."""
    return pd.Series(dtype=float, index=pd.Index([], dtype=object))


class EqualWeightStrategy:
    """
    Equal weight across every tradable name. A plumbing probe, not a signal.

    Phase 2 needs something to push through the engine to prove costs, bands and
    accounting work, and it must not be the real strategy — the roadmap builds
    the backtester before the signal precisely so the signal cannot be tuned
    against a forgiving backtest. This has no parameters to tune, so it cannot be.

    Its returns are not a result. Phase 3 replaces it.
    """

    name = "equal_weight"
    # 1/N already sums to one: fully invested is the statement, not a side effect.
    proposal_scale = "absolute"

    def propose(self, as_of: pd.Timestamp, history: PricePanel,
                candidates: list[str]) -> pd.Series:
        return pd.Series(1.0 / len(candidates), index=candidates)
