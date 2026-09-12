import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MAX_GROSS, MAX_SECTOR_WEIGHT, MAX_WEIGHT, MIN_HISTORY_DAYS
from src.data.panel import PricePanel
from src.data.universe import UniverseSpec

# One function forms both the backtest weights and the live weights — the Phase 2 gate. A
# backtest and a live system that build positions in two places drift apart invisibly, the
# backtest still reporting the strategy you designed while production trades something else.

# Constraints live here, not inside each strategy, so a new strategy cannot forget them and
# the risk limits stay auditable in one place.

# Rounded before leaving: two mathematically identical runs can differ in the last bit
# through a different summation order, failing the parity test for a reason that has
# nothing to do with the strategy. Ten decimals is finer than any broker position size.
WEIGHT_PRECISION = 10

PROPOSAL_SCALES = ("absolute", "relative")


class Strategy(Protocol):
    # Receives a panel already truncated to `as_of` and a candidate list already filtered
    # for tradability, so it cannot see the future. Anything missing, non-finite or
    # non-positive in the returned Series is treated as "no position".

    # `proposal_scale` is required because the two readings cannot be told apart from the
    # numbers. "absolute": already fractions of NAV, so a sum of 0.7 is a deliberate 30%
    # cash position. "relative": unitless conviction, normalised to the gross ceiling.

    # Undeclared, the same conviction *shape* lands at a different invested fraction purely
    # because of the magnitude it was computed at. "70% invested because the sizer asked"
    # and "70% because the scores came out small" are one a risk decision and one an accident.

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
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
) -> pd.Series:
    # Called from the backtest with a full-history panel and from the live runner with a
    # panel ending today; `panel.as_of` guarantees the strategy sees the same shape of
    # information in both cases.
    _check_limits(max_weight, max_gross, max_sector_weight)

    as_of = pd.Timestamp(as_of)
    history = panel.as_of(as_of)

    # Membership and tradability are different questions: `members_asof` answers "was this
    # name in the index then", `tradable_as_of` answers "did it have a bar then". The first
    # is a no-op today — this intersection is where a real membership history plugs in.
    tradable = history.tradable_as_of(as_of, min_history=min_history)
    candidates = sorted(set(tradable) & set(universe.members_asof(as_of)))
    if not candidates:
        return _empty_weights()

    proposed = strategy.propose(as_of, history, candidates)
    return _apply_constraints(proposed, candidates, _proposal_scale(strategy),
                              max_weight, max_gross, max_sector_weight, universe)


def _check_limits(max_weight: float, max_gross: float, max_sector_weight: float) -> None:
    # `_apply_constraints` drops non-positive proposals then scales to the gross ceiling, so
    # a negative ceiling flips every surviving weight through zero and a long-only path emits
    # shorts: max_weight=-0.1 turned a proposal of 1.0 into -0.1.

    # NaN is the quiet case — it fails every comparison, so the limit is not applied and the
    # output looks like a portfolio that respected it. Zero stays legal: "hold nothing" is a
    # legitimate instruction, "hold a negative amount of nothing" is not.
    for name, value in (("max_weight", max_weight), ("max_gross", max_gross),
                        ("max_sector_weight", max_sector_weight)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if value < 0:
            raise ValueError(
                f"{name} must not be negative, got {value}. A negative ceiling turns "
                "positive proposals into short positions on a long-only path.")


def _proposal_scale(strategy: Strategy) -> str:
    # Raising rather than defaulting is the point: a default is silently wrong for half of
    # all strategies, and wrong in a way that produces a plausible portfolio rather than an
    # error — which is how the mistake survives review.
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
                       max_weight: float, max_gross: float, max_sector_weight: float,
                       universe: UniverseSpec) -> pd.Series:
    # An "absolute" proposal is scaled down on a breach and never up — a strategy that asked
    # for 40% invested meant it. A "relative" one is normalised in either direction, because
    # its magnitude carries nothing to preserve; it never expressed exposure to invent.
    weights = pd.Series(proposed, dtype=float).reindex(candidates)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if weights.empty:
        return _empty_weights()

    # Order matters and the intuitive order is wrong. Clipping first destroys the signal for
    # any non-NAV scale: raw inverse-vol scores sit entirely above the cap, so every name
    # clips to max_weight and the later scaling yields a silently equal-weight portfolio.
    gross = weights.sum()
    if scale == "relative" or gross > max_gross:
        weights = weights * (max_gross / gross)

    # A clip, not a redistribution: pushing a capped name's excess into names the strategy
    # wanted less of would be the risk layer overruling the signal instead of bounding it.
    # It can leave the book below the ceiling, which is correct — a cap is a limit, not a target.
    weights = weights.clip(upper=max_weight)
    weights = _cap_sectors(weights, universe, max_sector_weight)

    return weights.sort_index().round(WEIGHT_PRECISION)


def _cap_sectors(weights: pd.Series, universe: UniverseSpec,
                 max_sector_weight: float) -> pd.Series:
    # Scaled within the group rather than clipped per name: the sector cap is a statement about
    # concentration, and flattening the names inside it would overrule the signal as well as bound
    # it. Only ever reduces, so the per-name cap applied above still holds.
    for members in universe.sector_groups(weights.index).values():
        exposure = weights[members].sum()
        if exposure > max_sector_weight:
            weights[members] *= max_sector_weight / exposure
    return weights


def _empty_weights() -> pd.Series:
    # All-cash, with the same dtype and index type as a populated portfolio.
    return pd.Series(dtype=float, index=pd.Index([], dtype=object))


class EqualWeightStrategy:
    # A plumbing probe, not a signal. The roadmap builds the backtester before the signal so
    # the signal cannot be tuned against a forgiving backtest; this has no parameters to
    # tune, so it cannot be. Its returns are not a result — Phase 3 replaces it.

    name = "equal_weight"
    # 1/N already sums to one: fully invested is the statement, not a side effect.
    proposal_scale = "absolute"

    def propose(self, as_of: pd.Timestamp, history: PricePanel,
                candidates: list[str]) -> pd.Series:
        return pd.Series(1.0 / len(candidates), index=candidates)
