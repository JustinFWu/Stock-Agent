import os
from pathlib import Path

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT / "models" / "saved"

# A fixed date rather than a yfinance `period`: 12-2 momentum burns twelve months on
# formation, yfinance has no "20y" period string, and a fixed date keeps rebuilds
# reproducible instead of sliding with today.
HISTORY_START = "2005-01-01"
DEFAULT_INTERVAL = "1d"

TRADING_DAYS = 252

ATR_PERIOD = 14
RETURN_HORIZONS = [1, 5, 10, 21, 63, 126, 252]

# First three are the HAR-RV daily/weekly/monthly cascade; 63 gives a slower regime reference.
RV_WINDOWS = [1, 5, 21, 63]
EWMA_LAMBDA = 0.94  # RiskMetrics decay for the EWMA variance baseline

FORWARD_DAYS = 5  # forecast horizon: next week's realized volatility

# These live here, not in either caller, because the backtester and the live runner must
# form weights under identical constraints — that is the Phase 2 gate. Two copies of
# "max 10% per name" is how a backtest and production drift apart while both look correct.

# Phase 3 should raise this to ~273: a 12-2 formation window reaches back thirteen months,
# so a name with exactly 252 bars ranks on a truncated, higher-vol window against the rest.
MIN_HISTORY_DAYS = 252
MAX_WEIGHT = 0.10        # per-name cap
MAX_GROSS = 1.0          # long-only, no leverage
NO_TRADE_BAND = 0.005    # ignore drift smaller than 50bp of NAV

# `fetcher.is_current` answers "was this fetched for the window I want", not "is it
# recent". Something must answer the second before weights become orders, or a cache from
# last March produces confident stale weights forever and a routine --fetch skips it.
MAX_LIVE_STALENESS_DAYS = 4   # a Friday close is still fresh the following Tuesday
MIN_LIVE_COVERAGE = 0.8       # fraction of the universe that must have a bar that session

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
