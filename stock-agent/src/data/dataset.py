"""
Dataset assembly for cross-sectional (pooled) training.

build_ticker_frame  -> one ticker's full feature/label frame (the per-ticker work
                       that pipeline.py used to do inline), tagged with a `ticker`
                       column and keeping its DatetimeIndex.
build_pooled_dataset -> stack many tickers into one frame, caching each ticker's
                        finished frame so the (slow) news scoring is paid once.
split_universe      -> deterministic, sector-stratified split into a training
                       universe and a held-out ticker basket you never tune on.
"""

import random
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DIR, FEATURES_DIR
from src.data.fetcher import fetch_and_save, load_bars
from src.data.universe import BENCHMARK, SECTOR_MAP
from src.features.technical import build_features
from src.features.relative_strength import add_relative_strength
from src.labels.target import add_label, add_cross_sectional_labels
from src.news.attach import attach_news_features
from src.news.fetcher import NewsFetchError


def _ensure_fetched(ticker: str, refetch: bool) -> None:
    """Fetch a ticker's bars only if missing (or refetch requested) to avoid re-downloading SPY 90x."""
    path = RAW_DIR / f"{ticker.upper()}.parquet"
    if refetch or not path.exists():
        fetch_and_save(ticker)


def build_ticker_frame(ticker: str, with_news: bool = True, refetch: bool = False) -> pd.DataFrame:
    """
    Build one ticker's complete feature + label frame: technical features, relative
    strength vs SPY/sector, the 5-day-forward label, and (optionally) per-day news
    sentiment. Adds a `ticker` column and keeps the DatetimeIndex so pooled frames
    can be split by date. This is the per-ticker work pipeline.py does inline, made
    reusable so many tickers can be stacked.
    """
    ticker = ticker.upper()
    _ensure_fetched(BENCHMARK, refetch)
    sector_etf = SECTOR_MAP.get(ticker)
    if sector_etf:
        _ensure_fetched(sector_etf, refetch)
    _ensure_fetched(ticker, refetch)

    df = load_bars(ticker)
    df = build_features(df)
    df = add_relative_strength(df, ticker)
    df = add_label(df)
    if with_news:
        df = attach_news_features(df, ticker)
    df["ticker"] = ticker
    return df


def build_pooled_dataset(
    tickers: list[str],
    with_news: bool = True,
    force_rebuild: bool = False,
    refetch: bool = False,
) -> pd.DataFrame:
    """
    Build and vertically stack frames for many tickers into one pooled dataset.

    Each ticker's finished frame is cached to FEATURES_DIR/pooled/<TICKER>_<suffix>.parquet
    (suffix distinguishes the news vs technical-only variants) and reused on later runs
    unless force_rebuild=True — important because the news path hits the LLM per ticker.
    Tickers that fail to build (e.g. no data) are skipped with a warning rather than
    aborting the whole run. A tqdm progress bar tracks how far through the universe the
    build is, with the current ticker shown as the bar's postfix.
    """
    cache_dir = FEATURES_DIR / "pooled"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "news" if with_news else "tech"

    n = len(tickers)
    frames = []
    bar = tqdm(tickers, desc=f"pooled dataset ({suffix})", unit="ticker")
    for t in bar:
        t = t.upper()
        bar.set_postfix_str(t)
        cache = cache_dir / f"{t}_{suffix}.parquet"
        if cache.exists() and not force_rebuild:
            frames.append(pd.read_parquet(cache))
            continue
        try:
            df = build_ticker_frame(t, with_news=with_news, refetch=refetch)
        except NewsFetchError as e:  # connectivity outage — leave uncached so it retries next run
            bar.write(f"  skip {t} (news unavailable, not cached): {e}")
            continue
        except Exception as e:  # skip a bad ticker, keep the rest of the universe
            bar.write(f"  skip {t}: {e}")
            continue
        df.to_parquet(cache)
        frames.append(df)
    bar.close()

    if not frames:
        raise ValueError("No ticker frames could be built for the pooled dataset.")

    print(f"  done: {len(frames)}/{n} tickers stacked")
    pooled = pd.concat(frames).sort_index()

    # Cross-sectional (relative) labels can only be computed on the stacked frame — they
    # rank each ticker against the rest of the universe on the same date. Adding them here
    # (rather than per-ticker in build_ticker_frame) keeps the per-ticker cache reusable
    # and recomputes the cross-section fresh for whatever set of tickers was actually built.
    pooled = add_cross_sectional_labels(pooled)
    return pooled


def split_universe(holdout_per_sector: int = 1, seed: int = 42) -> tuple[list[str], list[str]]:
    """
    Split SECTOR_MAP into (train_tickers, holdout_tickers), holding out
    `holdout_per_sector` names from each sector so the holdout basket spans every
    sector. Deterministic given `seed`. The holdout tickers are never trained on and
    should be evaluated only at the very end — they are the honesty check against
    tuning the model to the training universe's noise.
    """
    rng = random.Random(seed)
    by_sector: dict[str, list[str]] = {}
    for ticker, etf in SECTOR_MAP.items():
        by_sector.setdefault(etf, []).append(ticker)

    train, holdout = [], []
    for etf, names in by_sector.items():
        names = sorted(names)
        rng.shuffle(names)
        holdout.extend(names[:holdout_per_sector])
        train.extend(names[holdout_per_sector:])

    return sorted(train), sorted(holdout)
