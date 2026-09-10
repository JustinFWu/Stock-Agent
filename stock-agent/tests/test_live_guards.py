"""
The guards on the live weight path.

`fetcher.is_current` answers "was this file fetched for the window I want", and
deliberately not "is it recent" — which left nothing in the system answering the
second question. A cache from months ago therefore produced confident, entirely
stale live weights indefinitely, and a routine `--fetch` skipped it because the
window still matched.

Partial freshness is the subtler half. One refreshed ticker advances the panel's
last date while every stale name silently drops out of the candidate list, and the
result is a well-formed, fully-invested portfolio of whichever handful of names
happened to update. Nothing about that output looks wrong.

Both checks apply only when the caller does not name a session. Naming one is a
deliberate replay of history and must stay unguarded, or the parity gate could not
run at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import make_panel
from src.data.universe import UniverseSpec
from src.strategy.live import live_target_weights
from src.strategy.weights import EqualWeightStrategy

TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE")
UNIVERSE = UniverseSpec(id="synthetic-live", tickers=TICKERS, point_in_time=True, caveats=())

LAST_SESSION = pd.Timestamp("2024-06-28")


def panel_ending(last_session=LAST_SESSION, sessions: int = 400):
    """A fully-covered panel whose final bar is `last_session`."""
    dates = pd.bdate_range(end=last_session, periods=sessions)
    prices = pd.DataFrame(
        {t: 100.0 * (1 + 0.0003 * (i + 1)) ** np.arange(len(dates)) for i, t in enumerate(TICKERS)},
        index=dates,
    )
    return make_panel(prices)


def weights(panel, today, **kwargs):
    return live_target_weights(EqualWeightStrategy(), panel=panel, universe=UNIVERSE,
                               min_history=50, today=today, **kwargs)


def test_fresh_data_produces_weights():
    panel = panel_ending()
    result = weights(panel, today="2024-07-01")  # the next business day
    assert len(result) == len(TICKERS)


def test_a_stale_cache_is_refused():
    """The failure mode: months-old bars producing today's orders."""
    panel = panel_ending()
    with pytest.raises(ValueError, match="stale"):
        weights(panel, today="2024-09-05")


def test_the_staleness_limit_spans_a_long_weekend():
    """
    A Friday close must still be usable on the following Tuesday, or the guard fires
    on every public holiday and gets switched off by whoever is on call.
    """
    friday = pd.Timestamp("2024-06-28")
    panel = panel_ending(friday)

    weights(panel, today="2024-07-02")  # Tuesday: four days, at the limit
    with pytest.raises(ValueError, match="stale"):
        weights(panel, today="2024-07-03")


def test_an_explicit_session_is_a_deliberate_replay_and_is_not_guarded():
    """
    Naming a date means "re-run the decision for that session". Refusing because the
    session is not today would make historical replay — and the parity gate — impossible.
    """
    panel = panel_ending()
    result = weights(panel, today="2025-01-01", as_of=LAST_SESSION)
    assert len(result) == len(TICKERS)


def test_a_partially_refreshed_cache_is_refused():
    """
    One updated ticker advances the panel's last date; the rest drop out of the
    candidate list without a word. The weights that come back are well-formed, fully
    invested, and describe a portfolio nobody chose.
    """
    panel = panel_ending()
    stale_names = list(TICKERS[1:])  # only AAA has a bar on the final session
    closes = panel.closes.copy()
    closes.loc[closes.index[-1], stale_names] = np.nan
    partial = make_panel(closes)

    with pytest.raises(ValueError, match="usable bar"):
        weights(partial, today="2024-07-01")


def test_coverage_just_above_the_floor_is_accepted():
    """The guard rejects a thin session, not an ordinary one with a single halt."""
    panel = panel_ending()
    closes = panel.closes.copy()
    closes.loc[closes.index[-1], "EEE"] = np.nan  # 4/5 = 80%, exactly the floor
    partial = make_panel(closes)

    result = weights(partial, today="2024-07-01")
    assert "EEE" not in result.index
    assert len(result) == len(TICKERS) - 1
