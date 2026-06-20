"""
Attach per-day news sentiment features to a price/feature dataframe.

Each headline is mapped to the trading day whose close first makes it actionable
(see _effective_trading_day) so the feature never sees news published after that
row's decision point — no look-ahead. Lives here (rather than in pipeline.py) so the
pooled dataset builder and the predict path can both share it.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta, date

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
