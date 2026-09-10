import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.fetcher import load_bars

FIELDS = ("Open", "High", "Low", "Close", "Volume")

# NaN means "not tradable then" — the name did not exist, was halted, or had not listed.
# Reading that off the data rather than a hardcoded start-date list means the panel can
# never claim a name was available earlier than its prices say. Survivorship: see universe.py.


@dataclass(frozen=True)
class PricePanel:
    # Frozen because slicing returns a new panel: an engine that could extend its own
    # panel mid-run would defeat the point of as-of truncation.

    opens: pd.DataFrame
    highs: pd.DataFrame
    lows: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    # Carried on the panel rather than printed and forgotten: a run on 74 of 82 names is
    # a different experiment from one on all 82, and the result should say so.
    missing: tuple[str, ...] = ()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.closes.index

    @property
    def tickers(self) -> list[str]:
        return list(self.closes.columns)

    def as_of(self, date) -> "PricePanel":
        # The only look-ahead firewall in the system: weight formation receives this,
        # never the full panel. Inclusive of `date` — a decision made after the close of
        # day t is entitled to day t's bar; the engine fills it at the next open.
        date = pd.Timestamp(date)
        return self._map(lambda frame: frame.loc[:date])

    def tradable_as_of(self, date, min_history: int = 0) -> list[str]:
        # Enforced here so every strategy inherits it: otherwise a twelve-month formation
        # return gets computed from three months of data for a recent listing and ranked
        # against names with twenty years.
        date = pd.Timestamp(date)
        if date not in self.closes.index:
            return []

        history = self.closes.loc[:date]
        live = history.loc[date].notna()
        # Bars strictly before `date`, so `min_history` means what it says. Counting
        # today's bar would admit a name one day short of the requirement.
        deep_enough = history.iloc[:-1].notna().sum() >= min_history
        return sorted(history.columns[live & deep_enough])

    def _map(self, transform) -> "PricePanel":
        # By keyword, not by splatting a tuple: a positional rebuild depends silently on
        # field declaration order, so reordering the dataclass would swap opens for highs
        # with no error and no symptom beyond a wrong backtest.
        return PricePanel(
            opens=transform(self.opens),
            highs=transform(self.highs),
            lows=transform(self.lows),
            closes=transform(self.closes),
            volumes=transform(self.volumes),
            missing=self.missing,
        )


def load_price_panel(tickers: list[str]) -> PricePanel:
    # Deliberately does not download: a backtest must not be able to change its own input
    # data mid-run. Missing tickers are skipped rather than raised, so a twenty-minute
    # backtest does not die on its last ticker.
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
        missing=tuple(missing),
    )
