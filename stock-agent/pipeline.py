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
from src.data.universe import BENCHMARK


def run(ticker: str, validate: bool = False):
    ticker = ticker.upper()
    print(f"\n=== Pipeline for {ticker} ===")

    # 1. Fetch data (also fetch SPY for relative strength)
    print("\n[1/4] Fetching data...")
    fetch_and_save(BENCHMARK)
    fetch_and_save(ticker)

    # 2. Build features
    print("\n[2/4] Building features...")
    df = load_bars(ticker)
    df = build_features(df)
    df = add_relative_strength(df, ticker)
    df = add_label(df)
    print(f"      {len(df)} rows, {len(df.columns)} columns")

    # 3. Validate or train
    if validate:
        print("\n[3/4] Walk-forward validation...")
        walk_forward_validate(df)
    else:
        print("\n[3/4] Training model on all data...")
        train_final_model(df, ticker)

    # 4. Predict
    print("\n[4/4] Predicting...")
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
