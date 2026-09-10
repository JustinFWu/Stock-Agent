import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (MAX_GROSS, MAX_LIVE_STALENESS_DAYS, MAX_WEIGHT,
                    MIN_HISTORY_DAYS, MIN_LIVE_COVERAGE)
from src.data.panel import PricePanel, load_price_panel
from src.data.universe import UNIVERSE, UniverseSpec
from src.strategy.weights import Strategy, target_weights

# Thinness is the feature. Every decision of consequence — eligibility, history, the caps,
# the rounding — happens inside `target_weights`, so nothing here can diverge from the
# backtest. tests/test_weight_parity.py fails loudly the moment this grows a rule of its own.

# Order generation and submission are Phase 4. Stopping at weights means the same target can
# be produced, logged and compared long before anything can send an order.


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
    # `as_of` defaults to the panel's last date — in production, the session that just
    # closed. An explicit date is what makes the parity test possible, and is how a missed
    # session gets re-run for the day it should have traded rather than for today.

    # That default is the dangerous one, so it is the only guarded path: nothing else checks
    # freshness (`is_current` asks about the fetch *window*), so an old cache would produce
    # confident live weights forever. An explicit `as_of` is a deliberate replay and skips both.

    # `panel` and `today` are injectable so the guards can be tested without touching the
    # cache or waiting for tomorrow.
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
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    age_days = (today - as_of.normalize()).days
    if age_days > max_staleness_days:
        raise ValueError(
            f"The panel's most recent session is {as_of.date()}, {age_days} days old "
            f"(limit {max_staleness_days}). Refusing to form live weights from a stale "
            "cache — run `pipeline.py --fetch --refetch`. Pass an explicit as_of to "
            "replay a historical session deliberately.")

    # One updated ticker is enough to advance the panel's last date while every stale name
    # drops silently out of the candidates, producing a well-formed, fully-invested
    # portfolio of whichever handful of names happened to refresh.
    covered = len(panel.as_of(as_of).tradable_as_of(as_of, min_history=min_history))
    expected = len(universe.members_asof(as_of))
    if expected and covered < min_coverage * expected:
        raise ValueError(
            f"Only {covered}/{expected} universe names have a usable bar on "
            f"{as_of.date()} (need {min_coverage:.0%}). A partially refreshed cache "
            "advances the panel's last date while stale names drop silently out of the "
            "candidate list, leaving a portfolio of whichever names happened to update.")
