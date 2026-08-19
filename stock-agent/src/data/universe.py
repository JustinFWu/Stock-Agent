"""
Stock universe: which names exist, when, and what is wrong with that answer.

The universe is 82 of today's US large caps grouped by sector ETF. The sector map
also gives the risk layer the grouping it needs for per-sector exposure limits.

Survivorship, stated with a number rather than an adjective. These are the names
that are large caps *now*. Companies that were in the index between 2005 and 2026
and then failed, were acquired, or simply fell out are absent, and the strategies
that would have held them cannot be penalised for it. Measured against RSP — the
equal-weight S&P 500 ETF, which holds the weighting scheme constant and varies
only whether constituents were chosen with hindsight — equal-weight buy-and-hold
of this universe returned 16.8%/yr at Sharpe 0.91 with a -47% drawdown, against
10.1%/yr, Sharpe 0.58 and -60% for RSP over the same window. About 6.7pp of
annual return and 0.33 of Sharpe are available here for free, before any signal.
Eisdorfer (JFM 2008) puts roughly 40% of momentum's measured profit in delisting
returns specifically, which is exactly the part this universe cannot see.

The consequence is not "discount the result a little". It is that any *absolute*
performance figure from this universe is uninterpretable. Only differences
against a baseline run on the same universe mean anything, because the bias
appears in both and largely cancels. That is why `metrics.relative_summary`
exists and why the backtest runner always computes a baseline.

Fixing this properly needs point-in-time constituent data and delisted price
histories — Norgate, Sharadar and EODHD all sell it; yfinance cannot supply it
and, worse, silently serves recycled symbols (querying FB returns an ETF, BBBY
returns Overstock's price path under Bed Bath & Beyond's name). Splicing those in
would replace a known, signed, bounded error with an unbounded unknown that looks
like signal. Until that data is bought, the honest posture is to measure
relatively and disclose loudly.
"""

from dataclasses import dataclass

BENCHMARK = "SPY"

SURVIVORSHIP_CAVEAT = (
    "Universe 'survivor-82' is today's large caps: names that failed, were acquired, "
    "or fell out of the index between 2005 and 2026 are absent. Measured, equal-weight "
    "buy-and-hold of this universe returns 16.8%/yr at Sharpe 0.91 and -47% max "
    "drawdown, against 10.1%/yr, Sharpe 0.58 and -60% for RSP (point-in-time "
    "equal-weight S&P 500). Roughly 6.7pp/yr and 0.33 of Sharpe here is universe "
    "selection, not strategy. Absolute figures from this universe are uninterpretable; "
    "only differences against a baseline run on the same universe are."
)

# Sector membership, via each name's sector ETF.
SECTOR_MAP = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOG": "XLK", "GOOGL": "XLK",
    "META": "XLK", "AVGO": "XLK", "ADBE": "XLK", "CRM": "XLK", "AMD": "XLK",
    "INTC": "XLK", "CSCO": "XLK", "ORCL": "XLK", "QCOM": "XLK", "TXN": "XLK",
    # Healthcare
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "ABBV": "XLV",
    "MRK": "XLV", "TMO": "XLV", "ABT": "XLV", "DHR": "XLV", "AMGN": "XLV",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "BLK": "XLF", "C": "XLF", "SCHW": "XLF", "AXP": "XLF", "V": "XLF",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "NKE": "XLY", "MCD": "XLY",
    "SBUX": "XLY", "LOW": "XLY", "TJX": "XLY", "BKNG": "XLY",
    # Consumer Staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "PM": "XLP", "CL": "XLP", "MDLZ": "XLP",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    # Industrials
    "CAT": "XLI", "BA": "XLI", "HON": "XLI", "UPS": "XLI", "GE": "XLI",
    "RTX": "XLI", "DE": "XLI", "LMT": "XLI",
    # Utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU",
    # Real Estate
    "AMT": "XLRE", "PLD": "XLRE", "CCI": "XLRE", "EQIX": "XLRE",
    # Communication Services
    "DIS": "XLC", "NFLX": "XLC", "CMCSA": "XLC", "T": "XLC", "VZ": "XLC",
    # Materials
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "FCX": "XLB",
}


@dataclass(frozen=True)
class UniverseSpec:
    """
    A named set of tickers, bundled with an honest account of how it was chosen.

    `caveats` is a field on the universe rather than an argument a caller
    remembers to pass, because a disclosure that can be omitted will be omitted —
    including by some future script that summarises results into a table.
    """

    id: str
    tickers: tuple[str, ...]
    point_in_time: bool
    caveats: tuple[str, ...]

    def members_asof(self, date) -> list[str]:
        """
        Index members on `date`.

        Returns everything for every date, because this universe has no membership
        history — it is a snapshot of today. That is the honest behaviour rather
        than a stub: the seam exists so a real point-in-time membership table can
        be dropped in here without touching the engine, the strategy, or the
        weight path.

        This deliberately does not check whether a name had a *bar* on `date`.
        `PricePanel.tradable_as_of` answers that off the price data itself, and
        two places that both decide tradability will eventually disagree.
        """
        return list(self.tickers)


UNIVERSE = UniverseSpec(
    id="survivor-82",
    tickers=tuple(sorted(SECTOR_MAP)),
    point_in_time=False,
    caveats=(SURVIVORSHIP_CAVEAT,),
)
