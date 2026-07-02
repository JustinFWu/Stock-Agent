"""
News fetcher — pulls historical headlines for a ticker.
Supports Alpaca News API and Tiingo. Returns (publish_datetime_utc, headline)
pairs so downstream code can attach sentiment to the correct trading day.
"""

import requests
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone, date

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, TIINGO_API_KEY, MAX_ARTICLES_PER_DAY


# Retry policy for transient failures (DNS/connection drops, timeouts, 429/5xx).
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds; grows 2 -> 4 -> 8 between attempts
_RETRY_STATUS = {429, 500, 502, 503, 504}


class NewsFetchError(RuntimeError):
    """
    Raised when news cannot be fetched because of a network/connectivity failure
    (DNS resolution, dropped connection, timeout) that persisted across retries.

    Deliberately distinct from a successful response that simply contains no
    articles: the latter returns [] so a genuinely news-free ticker still builds,
    while this exception propagates up so build_pooled_dataset skips the ticker
    and does NOT cache a bogus zero-news frame. Without it a mid-run outage
    silently poisons the pooled cache with newsless tickers.
    """


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_with_retry(url: str, *, headers: dict, params: dict, provider: str) -> requests.Response | None:
    """
    GET with backoff on transient failures.

    Retries connection/timeout errors and 429/5xx responses up to _MAX_RETRIES;
    if still failing (i.e. a genuine connectivity outage), raises NewsFetchError so
    the caller skips the ticker rather than treating it as news-free. Non-transient
    HTTP errors (e.g. 401/404) are logged and return None, preserving the old
    "warn and continue with no articles" behaviour for those cases.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in _RETRY_STATUS:
                print(f"  Warning: {provider} news fetch failed ({exc}); continuing with 0 article(s).")
                return None
            last_exc = exc
        except requests.exceptions.RequestException as exc:
            last_exc = exc

        if attempt < _MAX_RETRIES:
            wait = _BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  {provider} news fetch attempt {attempt}/{_MAX_RETRIES} failed "
                  f"({last_exc.__class__.__name__}); retrying in {wait:.0f}s...")
            time.sleep(wait)

    raise NewsFetchError(
        f"{provider} news fetch failed after {_MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


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
        resp = _get_with_retry(url, headers=headers, params=params, provider="Alpaca")
        if resp is None:  # non-transient HTTP error already logged; treat as no news
            break
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

    resp = _get_with_retry(url, headers=headers, params=params, provider="Tiingo")
    if resp is None:  # non-transient HTTP error already logged; continue without Tiingo
        return []
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
