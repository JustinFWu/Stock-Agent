"""
End-to-end pipeline script.
Run this to go from zero to a prediction for any ticker.

Usage:
    python pipeline.py AAPL
    python pipeline.py AAPL --validate
"""

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from src.data.fetcher import fetch_and_save, load_bars
from src.features.technical import build_features
from src.features.relative_strength import add_relative_strength
from src.labels.target import add_label
from src.models.train import walk_forward_validate, train_final_model
from src.models.predict import predict
from src.data.universe import BENCHMARK, SECTOR_MAP
from src.news.fetcher import get_headlines
from src.news.sentiment import aggregate_sentiment


def run(ticker: str, validate: bool = False):
    ticker = ticker.upper()
    print(f"\n=== Pipeline for {ticker} ===")

    # 1. Fetch data (also fetch SPY + sector ETF for relative strength)
    print("\n[1/4] Fetching data...")
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

    # 3. News sentiment
    print("\n[3/5] Fetching news sentiment...")
    headlines = get_headlines(ticker)
    if headlines:
        sentiment = aggregate_sentiment(ticker, headlines)
        print(f"      {sentiment['count']} headlines: mean={sentiment['mean_score']:.2f}, "
              f"bullish={sentiment['n_bullish']}, bearish={sentiment['n_bearish']}")
        # Add sentiment as features to the latest rows
        df["news_sentiment"] = 0.0
        df["news_bullish_pct"] = 0.0
        df["news_bearish_pct"] = 0.0
        if sentiment["count"] > 0:
            df.loc[df.index[-1], "news_sentiment"] = sentiment["mean_score"]
            df.loc[df.index[-1], "news_bullish_pct"] = sentiment["n_bullish"] / sentiment["count"]
            df.loc[df.index[-1], "news_bearish_pct"] = sentiment["n_bearish"] / sentiment["count"]
    else:
        print("      No headlines found, skipping sentiment.")
        df["news_sentiment"] = 0.0
        df["news_bullish_pct"] = 0.0
        df["news_bearish_pct"] = 0.0

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
