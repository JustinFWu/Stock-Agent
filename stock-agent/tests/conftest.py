# Synthetic panels rather than the bar cache: a test that depends on downloaded data fails for
# reasons unrelated to the code, and a deterministic panel makes exact assertions possible. The
# two that need real bars — the parity gate and the smoke run — skip when the cache is empty.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.data.panel import PricePanel


def make_panel(prices: pd.DataFrame, volume: float = 1e9) -> PricePanel:
    # Flat bars make expected values computable by hand: with open == close, a rebalance filling at
    # the next open pays exactly the previous close, so any difference in the curve is the engine's.
    return PricePanel(
        opens=prices.copy(),
        highs=prices.copy(),
        lows=prices.copy(),
        closes=prices.copy(),
        volumes=pd.DataFrame(volume, index=prices.index, columns=prices.columns),
    )


@pytest.fixture
def flat_panel() -> PricePanel:
    # Three names, constant price, 500 trading days. Nothing should ever move.
    dates = pd.bdate_range("2020-01-01", periods=500)
    prices = pd.DataFrame(100.0, index=dates, columns=["AAA", "BBB", "CCC"])
    return make_panel(prices)


@pytest.fixture
def drifting_panel() -> PricePanel:
    # Deterministic drift is what exercises the no-trade band: weights pull apart at a known speed,
    # so whether a band is respected is a checkable fact rather than a judgement call.
    dates = pd.bdate_range("2020-01-01", periods=500)
    rates = {"AAA": 0.0010, "BBB": 0.0000, "CCC": -0.0005}
    prices = pd.DataFrame(
        {name: 100.0 * (1 + rate) ** np.arange(len(dates)) for name, rate in rates.items()},
        index=dates,
    )
    return make_panel(prices)


@pytest.fixture
def late_lister_panel() -> PricePanel:
    # One name whose bars begin partway through, for testing listing-date handling.
    dates = pd.bdate_range("2020-01-01", periods=500)
    prices = pd.DataFrame(100.0, index=dates, columns=["OLD", "NEW"])
    prices.loc[prices.index[:300], "NEW"] = np.nan
    return make_panel(prices)


def make_bars(periods: int = 500, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    # High at or above open and close, low at or below both: several estimators take log(High/Low)
    # and log(Close/Open) and would return a plausible number from an impossible bar. The seeded
    # generator keeps every derived assertion exact rather than approximate.
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)

    steps = rng.normal(0.0003, 0.012, periods)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = close * np.exp(rng.normal(0.0, 0.004, periods))

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    high = body_high * (1.0 + np.abs(rng.normal(0.0, 0.003, periods)))
    low = body_low * (1.0 - np.abs(rng.normal(0.0, 0.003, periods)))

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1_000_000, 5_000_000, periods).astype(float)},
        index=dates,
    )
