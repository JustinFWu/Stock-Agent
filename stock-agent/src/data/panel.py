"""
Aligned price panel: the single view of market data the backtester and the live
runner both read.

Per-ticker frames are what the feature pipeline wants; a portfolio wants the
transpose — every name's price on a given date, side by side. This assembles the
cached bars into wide date x ticker frames, one per OHLCV field, outer-joined on
date so a name that listed in 2012 is simply NaN before then.

Two properties here are load-bearing for an honest backtest:

  * `as_of` truncation. It is the only look-ahead firewall in the system. Weight
    formation never receives the full panel — it receives `panel.as_of(date)`,
    which physically cannot contain a future bar. Live trading passes a panel
    whose last row happens to be today, and the same code runs unchanged.

  * NaN means "not tradable then". A name with no bar on a date did not exist,
    was halted, or has not listed yet. `tradable_as_of` reads that directly off
    the data rather than off a hardcoded list of start dates, so the panel can
    never claim a name was available earlier than its prices say it was.

What this does NOT fix is survivorship: the universe is today's large caps, so
names that died are absent from the panel entirely and no amount of as-of
slicing brings them back. See `universe.py`.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.fetcher import load_bars

FIELDS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class PricePanel:
    """
    Wide OHLCV frames, all sharing one date index and one column set.

    Frozen because slicing returns a new panel rather than mutating this one —
    an engine that could accidentally extend its own panel mid-run would defeat
    the point of as-of truncation.
    """

    opens: pd.DataFrame
    highs: pd.DataFrame
    lows: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.closes.index

    @property
    def tickers(self) -> list[str]:
        return list(self.closes.columns)

    def as_of(self, date) -> "PricePanel":
        """
        A copy of this panel containing only bars on or before `date`.

        Inclusive of `date` itself: a decision made after the close of day t is
        entitled to day t's bar. Execution then happens at the next open, which
        the engine — not this class — is responsible for.
        """
        date = pd.Timestamp(date)
        return self._map(lambda frame: frame.loc[:date])

    def slice_dates(self, start=None, end=None) -> "PricePanel":
        """Restrict to a date window, for running a backtest over part of history."""
        start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else None
        return self._map(lambda frame: frame.loc[start:end])

    def tradable_as_of(self, date, min_history: int = 0) -> list[str]:
        """
        Names with a price on `date` and at least `min_history` prior bars.

        The history requirement is not bookkeeping — a strategy that needs a
        twelve-month formation return will otherwise happily compute one from
        three months of data for a recent listing and rank it against names with
        twenty years. Enforcing it here means every strategy inherits the rule.
        """
        date = pd.Timestamp(date)
        if date not in self.closes.index:
            return []

        history = self.closes.loc[:date]
        live = history.loc[date].notna()
        # Bars strictly before `date`, so `min_history` means what it says. Counting
        # today's bar as history would admit a name one day short of the requirement,
        # which for a thirteen-month momentum window is a name ranked on an
        # incomplete formation period against names measured on a full one.
        deep_enough = history.iloc[:-1].notna().sum() >= min_history
        return sorted(history.columns[live & deep_enough])

    def _map(self, transform) -> "PricePanel":
        """
        Apply `transform` to every field, keeping each frame with its own name.

        Built by keyword rather than by splatting a tuple: a positional rebuild
        silently depends on field declaration order, so reordering the dataclass
        would swap opens for highs with no error and no symptom beyond a wrong
        backtest.
        """
        return PricePanel(
            opens=transform(self.opens),
            highs=transform(self.highs),
            lows=transform(self.lows),
            closes=transform(self.closes),
            volumes=transform(self.volumes),
        )


def load_price_panel(tickers: list[str]) -> PricePanel:
    """
    Assemble a panel from the parquet bar cache.

    Tickers with no cached file are skipped with a warning rather than raising:
    a panel missing one name is still usable, and the alternative is a twenty-
    minute backtest dying on its last ticker. Bars must already be fetched —
    this deliberately does not download, so a backtest can never quietly change
    its own input data mid-run.
    """
    columns = {field: {} for field in FIELDS}
    missing = []

    for ticker in sorted({t.upper() for t in tickers}):
        try:
            bars = load_bars(ticker)
        except FileNotFoundError:
            missing.append(ticker)
            continue
        for field in FIELDS:
            columns[field][ticker] = bars[field]

    if not columns["Close"]:
        raise ValueError("No cached bars for any requested ticker — run `pipeline.py --fetch` first.")
    if missing:
        print(f"  panel: no cached bars for {', '.join(missing)} — excluded")

    frames = {field: pd.DataFrame(series).sort_index() for field, series in columns.items()}

    # One date index for every field, so `closes.loc[d]` and `opens.loc[d]` can never
    # disagree about which day they are describing.
    index = frames["Close"].index
    frames = {field: frame.reindex(index=index, columns=frames["Close"].columns)
              for field, frame in frames.items()}

    return PricePanel(
        opens=frames["Open"],
        highs=frames["High"],
        lows=frames["Low"],
        closes=frames["Close"],
        volumes=frames["Volume"],
    )
