"""
Fetches OHLCV data via yfinance and stores as parquet files.
"""

import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DIR, DEFAULT_PERIOD, DEFAULT_INTERVAL


def fetch_and_save(ticker: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Download OHLCV data for a ticker and save to parquet."""
    ticker = ticker.upper()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Fetching {ticker} ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance may return MultiIndex columns for single ticker; flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(path)
    print(f"  Saved {len(df)} rows to {path}")
    return df


def load_bars(ticker: str) -> pd.DataFrame:
    """Load previously fetched OHLCV data from parquet."""
    ticker = ticker.upper()
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data file for {ticker}. Run fetch_and_save('{ticker}') first.")
    return pd.read_parquet(path)
