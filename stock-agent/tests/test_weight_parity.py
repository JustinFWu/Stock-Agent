"""
The Phase 2 gate.

Pre-committed wording from the roadmap: *one function produces both the backtest
weights and the live weights, enforced by a test asserting byte-identical output
for a fixed date. If the paths can drift, the backtest is fiction.*

Byte-identical is taken literally here — the two Series are serialised and their
SHA-256 digests compared, not checked with a tolerance. A tolerance is exactly
the wrong instrument: it would pass a live path that had quietly acquired a
different rounding rule, a different sort order, or a different idea of which
names are eligible, and each of those is a real divergence between what was
tested and what will trade.

The comparison is only meaningful because the two sides are given *differently
shaped* inputs. The backtest hands over the whole panel, including bars after the
decision date, and relies on `panel.as_of` to truncate. The live path is handed a
panel that ends at the decision date, the way production sees the world.

That argument only holds if the strategy actually reads the panel. An earlier
version of these tests used `EqualWeightStrategy`, which returns 1/N and never
touches `history` — and with a history-blind probe, every test here passed with
`PricePanel.as_of` replaced by the identity function. The firewall could be
deleted outright and this file would not notice. The probe below therefore
derives its weights from trailing prices, so a decision date's weights depend on
which bars were visible; removing truncation changes the digest and the gate
fails, which is the property these tests were written to buy.
"""

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
from src.data.universe import UNIVERSE
from src.strategy.live import live_target_weights
from src.strategy.weights import target_weights

GATE_DATE = "2024-06-28"  # a month-end trading day, so the backtest rebalances on it

PROBE_WINDOW = 63


class HistoryProbeStrategy:
    """
    Inverse trailing volatility over the last quarter. A test instrument.

    Its only requirement is that the weights be a sensitive function of exactly
    which bars are visible, so that a leak of future data changes the answer.
    Inverse-vol satisfies that and is a plausible shape for real sizing, which
    keeps the parity test close to what Phase 3 will actually run through here.
    """

    name = "history_probe"

    def propose(self, as_of: pd.Timestamp, history: PricePanel,
                candidates: list[str]) -> pd.Series:
        closes = history.closes[candidates].tail(PROBE_WINDOW + 1)
        vol = np.log(closes / closes.shift(1)).std()
        return (1.0 / vol.replace(0.0, np.nan)).dropna()


def digest(weights: pd.Series) -> str:
    """SHA-256 of the serialised weights. Index, order and values all contribute."""
    payload = weights.to_json(orient="index", double_precision=15).encode()
    return hashlib.sha256(payload).hexdigest()


def truncate_independently(panel: PricePanel, date) -> PricePanel:
    """
    Cut a panel down to `date` without calling `PricePanel.as_of`.

    Using `as_of` to build the live-side panel would make these tests circular:
    both sides would inherit the same truncation, so disabling it would move both
    together and the digests would still match. Verified — with `as_of` replaced by
    the identity function this file passed until this helper existed. Slicing the
    frames here gives the live path a genuinely truncated panel, so the backtest
    path is the only one relying on the firewall and a leak shows up as a mismatch.
    """
    date = pd.Timestamp(date)
    return PricePanel(
        opens=panel.opens.loc[:date],
        highs=panel.highs.loc[:date],
        lows=panel.lows.loc[:date],
        closes=panel.closes.loc[:date],
        volumes=panel.volumes.loc[:date],
    )


@pytest.fixture(scope="module")
def real_panel():
    panel = load_price_panel(list(UNIVERSE.tickers))
    if len(panel.tickers) < 10:
        pytest.skip("bar cache is empty — run `pipeline.py --fetch` first")
    return panel


def test_backtest_and_live_weights_are_byte_identical(real_panel):
    """The gate. Same date, same strategy, two independent call paths."""
    strategy = HistoryProbeStrategy()
    gate_date = pd.Timestamp(GATE_DATE)

    result = run_backtest(real_panel, strategy, universe=UNIVERSE, start="2024-01-01",
                          end="2024-12-31", rebalance="M")
    assert gate_date in result.targets.index, "chose a date the backtest never rebalanced on"
    from_backtest = result.targets.loc[gate_date].dropna()

    # The live path sees only history up to the decision date, as production does.
    from_live = live_target_weights(strategy, as_of=gate_date,
                                    panel=truncate_independently(real_panel, gate_date))

    assert digest(from_backtest) == digest(from_live), (
        "backtest and live weight paths diverged — the backtest no longer describes "
        "what production would trade"
    )


def test_live_path_ignores_future_bars(real_panel):
    """
    Truncating the panel must not change the answer.

    If it ever does, something downstream is reading bars after `as_of`, and every
    backtest result in the repo is contaminated by look-ahead.
    """
    strategy = HistoryProbeStrategy()
    gate_date = pd.Timestamp(GATE_DATE)

    with_future = live_target_weights(strategy, as_of=gate_date, panel=real_panel)
    without_future = live_target_weights(strategy, as_of=gate_date,
                                         panel=truncate_independently(real_panel, gate_date))

    assert digest(with_future) == digest(without_future)


def test_live_and_shared_path_use_the_same_limits(real_panel):
    """
    The live wrapper must not hold its own copy of the risk limits.

    Calling the shared function directly with the config constants has to produce
    the same weights as the live wrapper's defaults, or the wrapper has drifted.
    """
    strategy = HistoryProbeStrategy()
    gate_date = pd.Timestamp(GATE_DATE)

    direct = target_weights(gate_date, real_panel, strategy, universe=UNIVERSE,
                            min_history=MIN_HISTORY_DAYS, max_weight=MAX_WEIGHT,
                            max_gross=MAX_GROSS)
    wrapped = live_target_weights(strategy, as_of=gate_date, panel=real_panel)

    assert digest(direct) == digest(wrapped)


def test_live_refuses_a_non_trading_day(real_panel):
    """A weekend is a caller mistake, not a date to silently round."""
    with pytest.raises(ValueError, match="not a trading day"):
        live_target_weights(HistoryProbeStrategy(), as_of="2024-06-29", panel=real_panel)
