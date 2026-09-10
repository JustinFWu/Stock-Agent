"""
The live side of the Phase 2 gate.

This is what a scheduled production run will call to answer "what should the
portfolio look like after today's close". It is deliberately thin: load the
panel, delegate to `target_weights`, return. Every decision of consequence —
which names are eligible, the history requirement, the per-name cap, the gross
ceiling, the rounding — happens inside the shared function, so there is nothing
here that could diverge from the backtest.

Thinness is the feature. The moment this file grows a rule of its own, the
backtest stops describing what production does, and `tests/test_weight_parity.py`
is there to fail loudly when that happens.

Order generation and submission are not here. Phase 4 adds a broker interface
that takes these weights and reconciles them against actual positions; the point
of stopping at weights is that the same target can be produced, logged and
compared long before anything can send an order.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (MAX_GROSS, MAX_LIVE_STALENESS_DAYS, MAX_WEIGHT,
                    MIN_HISTORY_DAYS, MIN_LIVE_COVERAGE)
from src.data.panel import PricePanel, load_price_panel
from src.data.universe import UNIVERSE, UniverseSpec
from src.strategy.weights import Strategy, target_weights


def live_target_weights(
    strategy: Strategy,
    as_of=None,
    panel: PricePanel | None = None,
    *,
    universe: UniverseSpec = UNIVERSE,
    min_history: int = MIN_HISTORY_DAYS,
    max_weight: float = MAX_WEIGHT,
    max_gross: float = MAX_GROSS,
    max_staleness_days: int = MAX_LIVE_STALENESS_DAYS,
    min_coverage: float = MIN_LIVE_COVERAGE,
    today=None,
) -> pd.Series:
    """
    Target weights for a live run, from the cached bar panel.

    `as_of` defaults to the last date in the panel, which in production is the
    session that just closed. Passing an explicit date is what makes the parity
    test possible, and is also how a missed session gets re-run for the day it
    was supposed to trade rather than for today.

    That default is also the dangerous one, so it is the only path that is guarded.
    Calling with no `as_of` means "trade on the latest data", and nothing else in
    the system checks whether the latest data is actually recent — `is_current`
    answers a question about the fetch *window*, not about freshness, and a routine
    `--fetch` skips a ticker whose window still matches however old its last bar is.
    Left unchecked, a cache from months ago produces confident live weights forever.
    Two things are therefore required before an unspecified `as_of` is accepted:

      staleness  the panel's last session must be within `max_staleness_days`.
      coverage   that session must carry bars for at least `min_coverage` of the
                 universe. One updated ticker is enough to advance the panel's last
                 date while every stale name silently drops out of the candidates,
                 which produces a well-formed, fully-invested portfolio of whichever
                 handful of names happened to refresh.

    An explicit `as_of` skips both checks. Naming a session is a deliberate replay
    of history, and refusing to replay 2024 because it is not today would be absurd.

    `panel` is injectable so a caller — the parity test, or a dry run against a
    frozen snapshot — can supply data without touching the cache on disk. The
    `universe` default is the same object the backtest runs on; overriding it here
    and not there is precisely the divergence the parity test exists to catch.
    `today` is injectable for the same reason: a freshness rule that cannot be
    tested without waiting for tomorrow is a freshness rule nobody tests.
    """
    if panel is None:
        panel = load_price_panel(list(universe.tickers))

    replaying = as_of is not None
    as_of = pd.Timestamp(as_of) if replaying else panel.dates[-1]
    if as_of not in panel.dates:
        raise ValueError(f"{as_of.date()} is not a trading day in the panel — "
                         "refusing to guess which session was meant.")

    if not replaying:
        _check_live_data(panel, as_of, universe, min_history,
                         max_staleness_days, min_coverage, today)

    return target_weights(as_of, panel, strategy, universe=universe,
                          min_history=min_history, max_weight=max_weight,
                          max_gross=max_gross)


def _check_live_data(panel: PricePanel, as_of, universe: UniverseSpec, min_history: int,
                     max_staleness_days: int, min_coverage: float, today) -> None:
    """Refuse to trade on data that is old, or on a session most names are missing from."""
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    age_days = (today - as_of.normalize()).days
    if age_days > max_staleness_days:
        raise ValueError(
            f"The panel's most recent session is {as_of.date()}, {age_days} days old "
            f"(limit {max_staleness_days}). Refusing to form live weights from a stale "
            "cache — run `pipeline.py --fetch --refetch`. Pass an explicit as_of to "
            "replay a historical session deliberately.")

    covered = len(panel.as_of(as_of).tradable_as_of(as_of, min_history=min_history))
    expected = len(universe.members_asof(as_of))
    if expected and covered < min_coverage * expected:
        raise ValueError(
            f"Only {covered}/{expected} universe names have a usable bar on "
            f"{as_of.date()} (need {min_coverage:.0%}). A partially refreshed cache "
            "advances the panel's last date while stale names drop silently out of the "
            "candidate list, leaving a portfolio of whichever names happened to update.")
