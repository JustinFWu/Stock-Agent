"""
End-to-end pipeline script.
Run this to go from zero to a prediction for any ticker.

Usage:
    python pipeline.py AAPL
    python pipeline.py AAPL --validate
"""

import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta, date
from dotenv import load_dotenv

load_dotenv()

from src.data.fetcher import fetch_and_save, load_bars
from src.features.technical import build_features
from src.features.relative_strength import add_relative_strength
from src.labels.target import add_label
from src.models.train import walk_forward_validate, train_final_model
from src.models.predict import predict
from src.data.universe import BENCHMARK, SECTOR_MAP
from src.news.fetcher import get_dated_headlines
from src.news.sentiment import sentiment_by_date


MARKET_CLOSE_HOUR_ET = 16  # 4pm US/Eastern equity close


def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def _is_us_eastern_dst(d: date) -> bool:
    # US DST: 2nd Sunday of March through 1st Sunday of November (stable since 2007).
    return _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)


def _to_eastern_naive(dt_utc: datetime) -> datetime:
    """Convert a tz-aware UTC datetime to naive US/Eastern wall-clock time."""
    offset = timedelta(hours=-4 if _is_us_eastern_dst(dt_utc.date()) else -5)
    return (dt_utc + offset).replace(tzinfo=None)


def _effective_trading_day(publish_utc: datetime, trading_days: pd.DatetimeIndex) -> date | None:
    """
    The trading day whose close first makes a headline actionable, so the feature
    never sees news published after that row's decision point (no look-ahead).
    News at/before the 16:00 ET close on a trading day maps to that day; news after
    close, or on a weekend/holiday, rolls forward to the next trading day present in
    the index. Returns None if the news post-dates the last available bar's close.
    """
    et = _to_eastern_naive(publish_utc)
    target = et.date()
    if et.hour >= MARKET_CLOSE_HOUR_ET:
        target = target + timedelta(days=1)
    pos = trading_days.searchsorted(pd.Timestamp(target))
    if pos >= len(trading_days):
        return None
    return trading_days[pos].date()


def attach_news_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Fetch historical headlines spanning df's date range and write sentiment onto the
    trading day whose close first makes each headline actionable (see
    _effective_trading_day) — aligning news to the exchange calendar and avoiding
    look-ahead. The news_has_news flag (1.0 on days with coverage, else 0.0) lets the
    model distinguish "no coverage" from genuinely neutral coverage, since both
    otherwise collapse to a 0.0 sentiment.
    """
    df["news_sentiment"] = 0.0
    df["news_bullish_pct"] = 0.0
    df["news_bearish_pct"] = 0.0
    df["news_has_news"] = 0.0

    if df.empty:
        return df

    # Reach a few days before the first bar so weekend/holiday news that only
    # becomes actionable on the first trading day is still fetched.
    start = datetime.combine(df.index.min().date(), datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=4)
    end = datetime.combine(df.index.max().date(), datetime.max.time(), tzinfo=timezone.utc)

    dated = get_dated_headlines(ticker, start, end)
    print(f"      {len(dated)} headlines spanning {start.date()} -> {end.date()}")
    if not dated:
        return df

    # Attribute each headline to its effective trading day before scoring.
    attributed: list[tuple[date, str]] = []
    for publish_utc, headline in dated:
        day = _effective_trading_day(publish_utc, df.index)
        if day is not None:
            attributed.append((day, headline))

    if not attributed:
        return df

    per_day = sentiment_by_date(ticker, attributed)
    print(f"      sentiment computed for {len(per_day)} distinct trading days")

    for d, stats in per_day.items():
        ts = pd.Timestamp(d)
        if ts in df.index:
            df.at[ts, "news_sentiment"] = stats["mean_score"]
            df.at[ts, "news_bullish_pct"] = stats["n_bullish"] / stats["count"]
            df.at[ts, "news_bearish_pct"] = stats["n_bearish"] / stats["count"]
            df.at[ts, "news_has_news"] = 1.0

    return df


def run(ticker: str, validate: bool = False):
    ticker = ticker.upper()
    print(f"\n=== Pipeline for {ticker} ===")

    # 1. Fetch data (also fetch SPY + sector ETF for relative strength)
    print("\n[1/5] Fetching data...")
    fetch_and_save(BENCHMARK)
    sector_etf = SECTOR_MAP.get(ticker)
    if sector_etf:
        fetch_and_save(sector_etf)
    fetch_and_save(ticker)

    # 2. Build features
    print("\n[2/5] Building features...")
    df = load_bars(ticker)
    df = build_features(df)
    df = add_relative_strength(df, ticker)
    df = add_label(df)
    print(f"      {len(df)} rows, {len(df.columns)} columns")

    # 3. News sentiment per row
    print("\n[3/5] Fetching historical news + scoring sentiment per day...")
    df = attach_news_features(df, ticker)
    n_with_news = int((df["news_sentiment"] != 0).sum())
    print(f"      {n_with_news}/{len(df)} rows have a non-zero sentiment value")

    # 4. Validate or train
    if validate:
        print("\n[4/5] Walk-forward validation...")
        walk_forward_validate(df)
    else:
        print("\n[4/5] Training model on all data...")
        train_final_model(df, ticker)

    # 5. Predict
    print("\n[5/5] Predicting...")
    latest = df.dropna().iloc[-1]
    result = predict(ticker, latest)
    print(f"\n  Ticker:     {result.ticker}")
    print(f"  Signal:     {result.signal}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Date:       {latest.name.date()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--validate", action="store_true", help="Run walk-forward validation instead of full train")
    args = parser.parse_args()
    run(args.ticker, validate=args.validate)
