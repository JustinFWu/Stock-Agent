"""
News fetcher — pulls historical headlines for a ticker.
Supports Alpaca News API and Tiingo. Returns (publish_datetime_utc, headline)
pairs so downstream code can attach sentiment to the correct trading day.
"""

import requests
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta, timezone, date

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, TIINGO_API_KEY, MAX_ARTICLES_PER_DAY


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_news_alpaca(ticker: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch Alpaca News articles between start and end (UTC). Paginates."""
    if not ALPACA_API_KEY:
        print("  Warning: ALPACA_API_KEY not set, skipping Alpaca news.")
        return []

    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    params = {
        "symbols": ticker,
        "start": _iso_utc(start),
        "end": _iso_utc(end),
        "limit": 50,
        "sort": "asc",
    }

    articles: list[dict] = []
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        articles.extend(body.get("news", []))
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    return articles


def get_news_tiingo(ticker: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch Tiingo news articles between start and end."""
    if not TIINGO_API_KEY:
        print("  Warning: TIINGO_API_KEY not set, skipping Tiingo news.")
        return []

    url = "https://api.tiingo.com/tiingo/news"
    headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
    params = {
        "tickers": ticker,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "limit": 1000,
        "sortBy": "publishedDate",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _parse_dt(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp to a tz-aware UTC datetime (naive input assumed UTC)."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_dated_headlines(
    ticker: str,
    start: datetime,
    end: datetime,
    max_per_day: int | None = MAX_ARTICLES_PER_DAY,
) -> list[tuple[datetime, str]]:
    """
    Returns [(publish_datetime_utc, headline), ...] between start and end.
    Keeping the publish time (not just the date) lets callers attribute each
    headline to the correct exchange trading day without look-ahead.
    Merges Alpaca + Tiingo when both keys are configured.
    Dedupes identical headlines per day, then caps to max_per_day to bound
    downstream LLM scoring cost. Pass max_per_day=None to disable the cap.
    """
    pairs: list[tuple[datetime, str]] = []

    for article in get_news_alpaca(ticker, start, end):
        dt = _parse_dt(article.get("created_at") or article.get("updated_at"))
        title = article.get("headline") or article.get("title") or ""
        if dt and title:
            pairs.append((dt, title))

    for article in get_news_tiingo(ticker, start, end):
        dt = _parse_dt(article.get("publishedDate"))
        title = article.get("title") or ""
        if dt and title:
            pairs.append((dt, title))

    seen_per_day: dict[date, set[str]] = defaultdict(set)
    kept: list[tuple[datetime, str]] = []
    for dt, title in pairs:
        day = dt.date()
        norm = title.strip().lower()
        if norm in seen_per_day[day]:
            continue
        seen_per_day[day].add(norm)
        if max_per_day is not None and len(seen_per_day[day]) > max_per_day:
            continue
        kept.append((dt, title))

    return kept


def get_headlines(ticker: str, days_back: int = 7) -> list[str]:
    """Recent-only convenience wrapper used by older callers."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    return [h for _, h in get_dated_headlines(ticker, start, end)]
