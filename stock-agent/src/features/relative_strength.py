"""
Computes relative strength of a stock vs market (SPY) and its sector ETF.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DIR
from src.data.universe import BENCHMARK, SECTOR_MAP


def _load(ticker: str) -> pd.Series:
    path = RAW_DIR / f"{ticker.upper()}.parquet"
    df = pd.read_parquet(path)
    return df["Close"]


def add_relative_strength(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add relative strength columns comparing ticker to SPY and its sector ETF.
    Requires SPY (and sector ETF if mapped) to already be fetched.
    """
    df = df.copy()
    close = df["Close"]

    try:
        spy = _load(BENCHMARK)
        spy = spy.reindex(close.index)
        for n in [5, 20]:
            ticker_return = close.pct_change(n)
            spy_return = spy.pct_change(n)
            df[f"rs_vs_spy_{n}d"] = ticker_return - spy_return
    except FileNotFoundError:
        print(f"Warning: {BENCHMARK} not fetched, skipping market relative strength.")

    sector_etf = SECTOR_MAP.get(ticker.upper())
    if sector_etf:
        try:
            sector = _load(sector_etf)
            sector = sector.reindex(close.index)
            for n in [5, 20]:
                ticker_return = close.pct_change(n)
                sector_return = sector.pct_change(n)
                df[f"rs_vs_sector_{n}d"] = ticker_return - sector_return
        except FileNotFoundError:
            print(f"Warning: {sector_etf} not fetched, skipping sector relative strength.")

    return df
