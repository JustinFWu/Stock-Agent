"""
Shared fixtures.

Most of these build synthetic panels rather than reading the bar cache. A test
that depends on downloaded data fails for reasons that have nothing to do with
the code, and a deterministic panel makes it possible to assert an exact number
instead of a plausible range. The two tests that genuinely need real bars — the
parity gate and the end-to-end smoke run — say so and skip when the cache is
empty.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.data.panel import PricePanel


def make_panel(prices: pd.DataFrame, volume: float = 1e9) -> PricePanel:
    """
    A panel where the open equals the close and the bar has no range.

    Flat bars make expected values computable by hand: with open == close, a
    rebalance that fills at the next open pays exactly the previous close, so any
    difference in the equity curve is the engine's doing and not the data's.
    """
    return PricePanel(
        opens=prices.copy(),
        highs=prices.copy(),
        lows=prices.copy(),
        closes=prices.copy(),
        volumes=pd.DataFrame(volume, index=prices.index, columns=prices.columns),
    )


@pytest.fixture
def flat_panel() -> PricePanel:
    """Three names, constant price, 500 trading days. Nothing should ever move."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    prices = pd.DataFrame(100.0, index=dates, columns=["AAA", "BBB", "CCC"])
    return make_panel(prices)


@pytest.fixture
def drifting_panel() -> PricePanel:
    """
    Three names compounding at different fixed daily rates.

    Deterministic drift is what exercises the no-trade band: weights pull apart
    at a known speed, so whether a band is respected becomes a checkable fact
    rather than a judgement call.
    """
    dates = pd.bdate_range("2020-01-01", periods=500)
    rates = {"AAA": 0.0010, "BBB": 0.0000, "CCC": -0.0005}
    prices = pd.DataFrame(
        {name: 100.0 * (1 + rate) ** np.arange(len(dates)) for name, rate in rates.items()},
        index=dates,
    )
    return make_panel(prices)


@pytest.fixture
def late_lister_panel() -> PricePanel:
    """One name whose bars begin partway through, for testing listing-date handling."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    prices = pd.DataFrame(100.0, index=dates, columns=["OLD", "NEW"])
    prices.loc[prices.index[:300], "NEW"] = np.nan
    return make_panel(prices)
