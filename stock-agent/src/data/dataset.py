"""
Dataset assembly.

build_ticker_frame   one ticker's features + forward labels, tagged with a `ticker`
                     column and keeping its DatetimeIndex.
build_pooled_dataset stack many tickers into one frame, caching each ticker's finished
                     frame so a rebuild does not recompute everything.

The pooled frame is a ragged panel — names that listed after HISTORY_START simply have
no rows before their IPO. That is fine everywhere downstream, because training and
evaluation split on dates, not on row position.
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import FEATURES_DIR
from src.data.fetcher import fetch_and_save, load_bars, is_current
from src.features.technical import build_features
from src.features.volatility import build_volatility_features
from src.labels.target import add_labels

CACHE_DIR = FEATURES_DIR / "pooled"


def build_ticker_frame(ticker: str, refetch: bool = False) -> pd.DataFrame:
    """Build one ticker's complete feature + label frame."""
    ticker = ticker.upper()
    if refetch or not is_current(ticker):
        fetch_and_save(ticker)

    df = load_bars(ticker)
    df = build_features(df)
    df = build_volatility_features(df)
    df = add_labels(df)
    df["ticker"] = ticker
    return df


def build_pooled_dataset(
    tickers: list[str],
    force_rebuild: bool = False,
    refetch: bool = False,
) -> pd.DataFrame:
    """
    Build and vertically stack frames for many tickers into one pooled dataset.

    Each finished frame is cached to FEATURES_DIR/pooled/<TICKER>.parquet. A cache is
    only trusted when the underlying bars still match the window in config — otherwise
    changing HISTORY_START would leave the model quietly training on the old window.
    Tickers that fail to build are skipped with a warning rather than aborting the run.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    failures = []
    bar = tqdm(tickers, desc="pooled dataset", unit="ticker")
    for ticker in bar:
        ticker = ticker.upper()
        bar.set_postfix_str(ticker)
        cache = CACHE_DIR / f"{ticker}.parquet"

        # Stale bars invalidate the derived features too, so the raw-data check gates both.
        stale = refetch or not is_current(ticker)
        if cache.exists() and not force_rebuild and not stale:
            frames.append(pd.read_parquet(cache))
            continue

        try:
            df = build_ticker_frame(ticker, refetch=refetch)
        except Exception as e:
            failures.append(ticker)
            bar.write(f"  skip {ticker}: {e}")
            continue

        df.to_parquet(cache)
        frames.append(df)
    bar.close()

    if not frames:
        raise ValueError("No ticker frames could be built for the pooled dataset.")

    pooled = pd.concat(frames).sort_index()
    print(f"  stacked {len(frames)}/{len(tickers)} tickers -> {len(pooled):,} rows, "
          f"{pooled.index.min().date()} to {pooled.index.max().date()}")
    if failures:
        print(f"  failed: {', '.join(failures)}")
    return pooled
