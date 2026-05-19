"""
News fetcher — pulls recent headlines for a ticker.
Supports Alpaca News API (free tier) and Tiingo.
Stub ready for v1; wire up API keys when needed.
"""

import requests
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, TIINGO_API_KEY


def get_news_alpaca(ticker: str, days_back: int = 7) -> list[dict]:
    """
    Fetch recent news from Alpaca Markets News API.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.
    """
    if not ALPACA_API_KEY:
        print("Warning: ALPACA_API_KEY not set, skipping news fetch.")
        return []

    start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    params = {"symbols": ticker, "start": start, "limit": 50}

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("news", [])


def get_news_tiingo(ticker: str, days_back: int = 7) -> list[dict]:
    """
    Fetch recent news from Tiingo.
    Requires TIINGO_API_KEY env var.
    """
    if not TIINGO_API_KEY:
        print("Warning: TIINGO_API_KEY not set, skipping news fetch.")
        return []

    start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/news"
    headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
    params = {"tickers": ticker, "startDate": start, "limit": 50}

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_headlines(ticker: str, days_back: int = 7) -> list[str]:
    """
    Returns a flat list of headline strings.
    Tries Alpaca first, falls back to Tiingo.
    """
    articles = get_news_alpaca(ticker, days_back)
    if not articles:
        articles = get_news_tiingo(ticker, days_back)

    headlines = []
    for article in articles:
        title = article.get("headline") or article.get("title") or ""
        if title:
            headlines.append(title)
    return headlines
