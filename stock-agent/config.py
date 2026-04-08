from pathlib import Path

ROOT = Path(__file__).parent

# Data paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = ROOT / "models" / "saved"

# Data settings
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

# Feature settings
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_PERIODS = [10, 20, 50]
RETURN_HORIZONS = [1, 5, 10, 20]

# Label settings
FORWARD_DAYS = 5  # predict 1 week ahead

# Model settings
CONFIDENCE_THRESHOLD = 0.65  # no-trade zone below this

# API keys (set via environment variables)
import os
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "")
