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

# API keys (set via environment variables)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
