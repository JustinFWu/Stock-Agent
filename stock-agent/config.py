import math
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

# Formation runs from t-273 to t-21: twelve months of return ending one month back. The skip
# dodges short-term reversal, and is the whole reason the factor is named 12-2.
MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_SKIP_DAYS = 21
MOMENTUM_TOP_FRACTION = 0.10

# Derived, never typed twice: a name that cannot span the formation window would rank on a
# truncated, higher-vol one against names measured over a full window.
MIN_HISTORY_DAYS = MOMENTUM_LOOKBACK_DAYS + MOMENTUM_SKIP_DAYS

MAX_WEIGHT = 0.10        # per-name cap
MAX_SECTOR_WEIGHT = 0.30
MAX_GROSS = 1.0          # long-only, no leverage
NO_TRADE_BAND = 0.005    # ignore drift smaller than 50bp of NAV

# A book of fewer than MAX_GROSS/MAX_WEIGHT names cannot reach full investment: the per-name
# cap truncates it and the leftover reads as a deliberate cash position it never chose.
# Measured at the decile, that silently held 20% cash and cost momentum 0.12 of information
# ratio against the same-universe baseline.
MIN_SELECTED_NAMES = math.ceil(MAX_GROSS / MAX_WEIGHT)

VOL_TARGET_ANNUAL = 0.10
COV_LOOKBACK_DAYS = 252
VOL_SCALE_CAP = 1.5

# Idle cash earns something, and vol targeting parks a lot of the book there. A flat rate rather
# than a fetched bill series: a second data dependency for a second-order effect.
CASH_ANNUAL_RATE = 0.02

# `fetcher.is_current` answers "was this fetched for the window I want", not "is it
# recent". Something must answer the second before weights become orders, or a cache from
# last March produces confident stale weights forever and a routine --fetch skips it.
MAX_LIVE_STALENESS_DAYS = 4   # a Friday close is still fresh the following Tuesday
MIN_LIVE_COVERAGE = 0.8       # fraction of the universe that must have a bar that session

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
