"""
Stock universe definitions: benchmark, sector membership.

Survivorship warning: these are today's large caps. Names that fell out of the index
over the last twenty years are absent, so any backtest run over this universe is
biased upward. That is tolerable for checking that the machinery works and for live
trading, where you can only trade what exists — it is not evidence a strategy is good.
"""

BENCHMARK = "SPY"

# Sector membership, via each name's sector ETF. Defines the universe, and gives the
# risk layer the grouping it needs for per-sector exposure limits.
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

ALL_TICKERS = sorted(SECTOR_MAP)
