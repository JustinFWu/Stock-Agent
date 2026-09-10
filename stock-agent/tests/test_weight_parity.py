# The Phase 2 gate, pre-committed: one function produces both the backtest and the live weights,
# asserted byte-identical via SHA-256 rather than a tolerance. A tolerance would pass a live path
# that had acquired a different rounding rule, sort order, or idea of which names are eligible.

# The comparison only means something because the two sides get differently shaped inputs: the
# backtest gets the whole panel and relies on `panel.as_of`, the live path gets one already ending
# at the decision date. That holds only if the strategy actually reads the panel — see the probe.

# Runs on a synthetic panel, which is a correction rather than a convenience: the fixture used to
# call `load_price_panel` before counting tickers, so on a clean checkout the principal gate of
# Phase 2 raised before its advertised skip and was unrunnable in CI.

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from config import MAX_GROSS, MAX_WEIGHT, MIN_HISTORY_DAYS
from src.backtest.engine import run_backtest
from src.data.panel import PricePanel, load_price_panel
from src.data.universe import UNIVERSE, UniverseSpec
from src.strategy.live import live_target_weights
from src.strategy.weights import target_weights

REAL_GATE_DATE = "2024-06-28"  # a month-end trading day, so the backtest rebalances on it

PROBE_WINDOW = 63

# Long enough that MIN_HISTORY_DAYS is satisfied well before the gate date, so the
# synthetic gate exercises the same history requirement production runs under.
SYNTHETIC_SESSIONS = 700
SYNTHETIC_TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE")
SYNTHETIC_GATE_MONTH = pd.Period("2022-06")


class HistoryProbeStrategy:
    # A test instrument. It only has to make the weights a sensitive function of exactly which bars
    # are visible, so a leak of future data changes the answer. Inverse-vol does that and is a
    # plausible shape for real sizing, keeping the gate close to what Phase 3 will run.

    name = "history_probe"
    # Raw inverse-vol scores: only the ratios mean anything, and their magnitude
    # is whatever the volatilities of the day happen to give.
    proposal_scale = "relative"

    def propose(self, as_of: pd.Timestamp, history: PricePanel,
                candidates: list[str]) -> pd.Series:
        closes = history.closes[candidates].tail(PROBE_WINDOW + 1)
        vol = np.log(closes / closes.shift(1)).std()
        return (1.0 / vol.replace(0.0, np.nan)).dropna()


def digest(weights: pd.Series) -> str:
    # SHA-256 of the serialised weights. Index, order and values all contribute.
    payload = weights.to_json(orient="index", double_precision=15).encode()
    return hashlib.sha256(payload).hexdigest()


def truncate_independently(panel: PricePanel, date) -> PricePanel:
    # Using `as_of` here would make the tests circular: both sides would inherit the same
    # truncation, so disabling it would move both together and the digests would still match.
    # Verified — with `as_of` replaced by the identity this file passed until this helper existed.
    date = pd.Timestamp(date)
    return PricePanel(
        opens=panel.opens.loc[:date],
        highs=panel.highs.loc[:date],
        lows=panel.lows.loc[:date],
        closes=panel.closes.loc[:date],
        volumes=panel.volumes.loc[:date],
    )


# --------------------------------------------------------------------------- #
# The synthetic gate. Deterministic, history-sensitive, and needs nothing on disk.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def synthetic_panel() -> PricePanel:
    # Different volatilities matter: with identical ones the inverse-vol probe returns near-equal
    # weights, and a test expecting "everything equal" cannot tell a preserved ranking from a
    # destroyed one.
    rng = np.random.default_rng(20240628)
    dates = pd.bdate_range("2020-01-01", periods=SYNTHETIC_SESSIONS)

    closes = {}
    for i, ticker in enumerate(SYNTHETIC_TICKERS):
        daily_vol = 0.008 + 0.004 * i
        steps = rng.normal(0.0002, daily_vol, len(dates))
        closes[ticker] = 100.0 * np.exp(np.cumsum(steps))

    closes = pd.DataFrame(closes, index=dates)
    opens = closes.shift(1).fillna(closes.iloc[0])
    highs = pd.concat([opens, closes]).groupby(level=0).max() * 1.002
    lows = pd.concat([opens, closes]).groupby(level=0).min() * 0.998

    return PricePanel(opens=opens, highs=highs, lows=lows, closes=closes,
                      volumes=pd.DataFrame(1e9, index=dates, columns=list(SYNTHETIC_TICKERS)))


@pytest.fixture(scope="module")
def synthetic_universe() -> UniverseSpec:
    return UniverseSpec(id="synthetic-parity", tickers=SYNTHETIC_TICKERS,
                        point_in_time=True, caveats=())


@pytest.fixture(scope="module")
def synthetic_gate_date(synthetic_panel) -> pd.Timestamp:
    # The last trading day of the gate month — the day the monthly schedule rebalances on.
    dates = synthetic_panel.dates
    in_month = dates[dates.to_period("M") == SYNTHETIC_GATE_MONTH]
    assert len(in_month), "gate month is outside the synthetic panel"
    return in_month[-1]


def test_backtest_and_live_weights_are_byte_identical(
        synthetic_panel, synthetic_universe, synthetic_gate_date):
    # The gate. Same date, same strategy, two independent call paths.
    strategy = HistoryProbeStrategy()

    result = run_backtest(synthetic_panel, strategy, universe=synthetic_universe,
                          start="2021-06-01", rebalance="M")
    assert synthetic_gate_date in result.targets.index, \
        "chose a date the backtest never rebalanced on"
    from_backtest = result.targets.loc[synthetic_gate_date].dropna()
    assert not from_backtest.empty, "the gate date produced no positions to compare"

    # The live path sees only history up to the decision date, as production does.
    from_live = live_target_weights(
        strategy, as_of=synthetic_gate_date, universe=synthetic_universe,
        panel=truncate_independently(synthetic_panel, synthetic_gate_date))

    assert digest(from_backtest) == digest(from_live), (
        "backtest and live weight paths diverged — the backtest no longer describes "
        "what production would trade"
    )


def test_live_path_ignores_future_bars(synthetic_panel, synthetic_universe, synthetic_gate_date):
    # If truncating the panel ever changes the answer, something downstream is reading bars after
    # `as_of`, and every backtest result in the repo is contaminated by look-ahead.
    strategy = HistoryProbeStrategy()

    with_future = live_target_weights(strategy, as_of=synthetic_gate_date,
                                      universe=synthetic_universe, panel=synthetic_panel)
    without_future = live_target_weights(
        strategy, as_of=synthetic_gate_date, universe=synthetic_universe,
        panel=truncate_independently(synthetic_panel, synthetic_gate_date))

    assert digest(with_future) == digest(without_future)


def test_live_and_shared_path_use_the_same_limits(
        synthetic_panel, synthetic_universe, synthetic_gate_date):
    # Calling the shared function directly with the config constants must produce the same weights
    # as the live wrapper's defaults, or the wrapper has drifted.
    strategy = HistoryProbeStrategy()

    direct = target_weights(synthetic_gate_date, synthetic_panel, strategy,
                            universe=synthetic_universe, min_history=MIN_HISTORY_DAYS,
                            max_weight=MAX_WEIGHT, max_gross=MAX_GROSS)
    wrapped = live_target_weights(strategy, as_of=synthetic_gate_date,
                                  universe=synthetic_universe, panel=synthetic_panel)

    assert digest(direct) == digest(wrapped)


def test_live_refuses_a_non_trading_day(synthetic_panel, synthetic_universe):
    # A weekend is a caller mistake, not a date to silently round.
    saturday = pd.Timestamp("2022-06-25")
    assert saturday not in synthetic_panel.dates

    with pytest.raises(ValueError, match="not a trading day"):
        live_target_weights(HistoryProbeStrategy(), as_of=saturday,
                            universe=synthetic_universe, panel=synthetic_panel)


# The same gate against real bars: the only coverage proving the paths agree on actual market
# data, and the only one needing a populated cache, so it skips rather than failing a clean
# checkout.

@pytest.fixture(scope="module")
def real_panel():
    try:
        panel = load_price_panel(list(UNIVERSE.tickers))
    except ValueError:
        # The loader raises when nothing is cached at all — the skip below has to
        # come after this, not instead of it, or the gate errors on a fresh clone.
        pytest.skip("bar cache is empty — run `pipeline.py --fetch` first")
    if len(panel.tickers) < 10:
        pytest.skip("bar cache is too sparse to run the real-data gate")
    if pd.Timestamp(REAL_GATE_DATE) not in panel.dates:
        pytest.skip(f"cache does not cover {REAL_GATE_DATE}")
    return panel


def test_real_data_backtest_and_live_weights_are_byte_identical(real_panel):
    # The gate again, on real bars, where the price paths are not of our choosing.
    strategy = HistoryProbeStrategy()
    gate_date = pd.Timestamp(REAL_GATE_DATE)

    result = run_backtest(real_panel, strategy, universe=UNIVERSE, start="2024-01-01",
                          end="2024-12-31", rebalance="M")
    assert gate_date in result.targets.index, "chose a date the backtest never rebalanced on"
    from_backtest = result.targets.loc[gate_date].dropna()

    from_live = live_target_weights(strategy, as_of=gate_date,
                                    panel=truncate_independently(real_panel, gate_date))

    assert digest(from_backtest) == digest(from_live)


def test_real_data_live_path_ignores_future_bars(real_panel):
    # The look-ahead firewall, checked against real bars.
    strategy = HistoryProbeStrategy()
    gate_date = pd.Timestamp(REAL_GATE_DATE)

    with_future = live_target_weights(strategy, as_of=gate_date, panel=real_panel)
    without_future = live_target_weights(strategy, as_of=gate_date,
                                         panel=truncate_independently(real_panel, gate_date))

    assert digest(with_future) == digest(without_future)
