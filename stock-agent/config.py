import os
from pathlib import Path

ROOT = Path(__file__).parent

# Data paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT / "models" / "saved"

# Data settings
# An explicit start date rather than a yfinance `period`: 12-2 momentum burns twelve
# months on formation, so a short window leaves almost nothing to test on. yfinance
# only accepts a fixed set of period strings (no "20y"), and a fixed date also makes
# every rebuild reproducible instead of sliding with today's date.
HISTORY_START = "2005-01-01"
DEFAULT_INTERVAL = "1d"

TRADING_DAYS = 252

# Feature settings
ATR_PERIOD = 14
RETURN_HORIZONS = [1, 5, 10, 21, 63, 126, 252]

# Realized-volatility windows (trading days). The first three are the HAR-RV
# daily/weekly/monthly cascade; 63 gives a slower regime reference.
RV_WINDOWS = [1, 5, 21, 63]
EWMA_LAMBDA = 0.94  # RiskMetrics decay for the EWMA variance baseline

# Label settings
FORWARD_DAYS = 5  # forecast horizon: next week's realized volatility

# Portfolio construction limits.
#
# These live in config rather than in either caller because the backtester and the
# live runner must form weights under identical constraints — that is the Phase 2
# gate. Two copies of "max 10% per name" is exactly how a backtest and production
# drift apart while both look correct in isolation.
# A name needs a year of bars before it can be held. Phase 3 should raise this to
# ~273 (13 months): a 12-2 formation window reaches back thirteen months, so a
# name with exactly 252 bars would be ranked on a truncated, higher-volatility
# window against names measured over the full one.
MIN_HISTORY_DAYS = 252
MAX_WEIGHT = 0.10        # per-name cap
MAX_GROSS = 1.0          # long-only, no leverage
NO_TRADE_BAND = 0.005    # ignore drift smaller than 50bp of NAV

# API keys (set via environment variables)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
