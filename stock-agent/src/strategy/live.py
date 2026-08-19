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
from config import MAX_GROSS, MAX_WEIGHT, MIN_HISTORY_DAYS
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
) -> pd.Series:
    """
    Target weights for a live run, from the cached bar panel.

    `as_of` defaults to the last date in the panel, which in production is the
    session that just closed. Passing an explicit date is what makes the parity
    test possible, and is also how a missed session gets re-run for the day it
    was supposed to trade rather than for today.

    `panel` is injectable so a caller — the parity test, or a dry run against a
    frozen snapshot — can supply data without touching the cache on disk. The
    `universe` default is the same object the backtest runs on; overriding it here
    and not there is precisely the divergence the parity test exists to catch.
    """
    if panel is None:
        panel = load_price_panel(list(universe.tickers))

    as_of = pd.Timestamp(as_of) if as_of is not None else panel.dates[-1]
    if as_of not in panel.dates:
        raise ValueError(f"{as_of.date()} is not a trading day in the panel — "
                         "refusing to guess which session was meant.")

    return target_weights(as_of, panel, strategy, universe=universe,
                          min_history=min_history, max_weight=max_weight,
                          max_gross=max_gross)
