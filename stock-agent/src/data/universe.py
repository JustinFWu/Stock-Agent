from dataclasses import dataclass, field

BENCHMARK = "SPY"

SURVIVORSHIP_CAVEAT = (
    "Universe 'survivor-82' is today's large caps: names that failed, were acquired, "
    "or fell out of the index between 2005 and 2026 are absent. Measured, an equal-weight "
    "daily-rebalanced run over this universe returns 16.8%/yr at Sharpe 0.91 and -47% max "
    "drawdown, against 10.1%/yr, Sharpe 0.58 and -60% for RSP (point-in-time "
    "equal-weight S&P 500). That ~6.7pp/yr and 0.33 of Sharpe indicates the scale of "
    "the advantage this universe confers, but it is not an isolated measurement of "
    "survivorship: the two portfolios differ in composition and rebalancing as well as "
    "in hindsight. Absolute figures from this universe are uninterpretable; a baseline "
    "run on the same universe removes what both portfolios share, which is narrower "
    "than removing the selection bias."
)

# Live as of Phase 3: `UniverseSpec.sectors` carries this into the weight path, where
# MAX_SECTOR_WEIGHT bounds how much of the book one sector can hold.
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
    # `caveats` is a field rather than an argument a caller remembers to pass, because a
    # disclosure that can be omitted will be omitted — including by some future script
    # that summarises results into a table.

    id: str
    tickers: tuple[str, ...]
    point_in_time: bool
    caveats: tuple[str, ...]
    sectors: dict[str, str] = field(default_factory=dict)

    def sector_groups(self, tickers) -> dict[str, list[str]]:
        # A name with no mapping is its own group, so an unmapped ticker is bounded by the
        # per-name cap rather than escaping the sector cap entirely.
        groups: dict[str, list[str]] = {}
        for ticker in tickers:
            groups.setdefault(self.sectors.get(ticker, ticker), []).append(ticker)
        return groups

    def members_asof(self, date) -> list[str]:
        # Everything, every date: this universe is a snapshot of today with no membership
        # history. The seam exists so a real point-in-time table drops in here without
        # touching the engine, the strategy or the weight path.

        # Deliberately does not check whether a name had a *bar* on `date` —
        # `PricePanel.tradable_as_of` answers that off the price data, and two places
        # that both decide tradability will eventually disagree.
        return list(self.tickers)


# Fixing this needs bought point-in-time constituents and delisted histories. yfinance
# cannot supply them and silently serves recycled symbols (FB returns an ETF, BBBY returns
# Overstock's path), so splicing it in trades a bounded error for one that looks like signal.
UNIVERSE = UniverseSpec(
    id="survivor-82",
    tickers=tuple(sorted(SECTOR_MAP)),
    point_in_time=False,
    caveats=(SURVIVORSHIP_CAVEAT,),
    sectors=dict(SECTOR_MAP),
)
