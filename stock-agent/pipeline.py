"""
End-to-end pipeline — one pooled (cross-sectional) model over the whole universe.

Training and validation are universe-level (not per ticker): we build one stacked
dataset across many tickers, train a single pooled model, and validate it with a
date-based walk-forward. Prediction is then pick-a-ticker: that one pooled model
scores whichever ticker you ask about.

Usage:
    python pipeline.py --train          # build pooled dataset, train + save the model
    python pipeline.py --validate       # pooled walk-forward AUC + holdout-ticker check
    python pipeline.py AAPL             # predict AAPL with the pooled model
    python pipeline.py --train --news   # include per-day news sentiment (slow, hits the LLM)
"""

import argparse
import numpy as np
from datetime import date
from dotenv import load_dotenv

load_dotenv()

from config import FORWARD_DAYS
from src.data.dataset import build_pooled_dataset, build_ticker_frame, split_universe
from src.models.train import (
    walk_forward_validate,
    walk_forward_holdout,
    train_pooled_model,
    get_available_features,
)
from src.models.predict import predict


def _build_training_pool(with_news, force_rebuild, refetch, holdout_per_sector):
    """Build the pooled training dataset and return it alongside the held-out tickers."""
    train_tickers, holdout_tickers = split_universe(holdout_per_sector=holdout_per_sector)
    print(f"Train universe: {len(train_tickers)} tickers")
    print(f"Holdout basket: {len(holdout_tickers)} tickers -> {holdout_tickers}")
    train_df = build_pooled_dataset(train_tickers, with_news=with_news,
                                    force_rebuild=force_rebuild, refetch=refetch)
    print(f"  pooled train: {len(train_df)} rows from {train_df['ticker'].nunique()} tickers")
    return train_df, holdout_tickers


def do_validate(with_news, force_rebuild, refetch, holdout_per_sector, n_splits):
    print(f"\n[1/2] Building pooled dataset + walk-forward validation ({n_splits} splits)...")
    train_df, holdout_tickers = _build_training_pool(with_news, force_rebuild, refetch, holdout_per_sector)
    walk_forward_validate(train_df, n_splits=n_splits)

    print("\n[2/2] Holdout check on never-trained tickers...")
    holdout_df = build_pooled_dataset(holdout_tickers, with_news=with_news,
                                      force_rebuild=force_rebuild, refetch=refetch)
    walk_forward_holdout(train_df, holdout_df, n_splits=n_splits)


def do_train(with_news, force_rebuild, refetch, holdout_per_sector):
    print("\n[1/1] Building pooled dataset + training the pooled model...")
    train_df, _ = _build_training_pool(with_news, force_rebuild, refetch, holdout_per_sector)
    train_pooled_model(train_df)


def do_predict(ticker, with_news, refetch):
    ticker = ticker.upper()
    print(f"\n=== Predicting {ticker} with the pooled model ===")
    # A live prediction must not run on stale bars: always pull fresh OHLCV (cheap for a
    # single ticker — the slow part, per-headline LLM scoring, is cached across runs), so
    # the newest bar and any recent news actually reach the model. --refetch stays a no-op
    # override here; the bulk train/validate paths still honour it for speed.
    df = build_ticker_frame(ticker, with_news=with_news, refetch=True)

    # Use the most recent row with complete FEATURES, not labels: add_label leaves the
    # last FORWARD_DAYS rows' labels NaN, and a blanket dropna() would discard them and
    # predict on a stale bar.
    feature_cols = get_available_features(df)
    latest = df.dropna(subset=feature_cols).iloc[-1]

    # Even after refetching, the freshest usable bar can lag today (weekend/holiday, or a
    # yfinance data gap). Warn if it is more than a trading week old so a silent data
    # outage can't masquerade as a current signal.
    bars_old = int(np.busday_count(latest.name.date(), date.today()))
    if bars_old > FORWARD_DAYS:
        print(f"  WARNING: newest usable bar is {latest.name.date()} ({bars_old} trading days old) — "
              f"data may be stale (market gap or fetch issue); treat this signal with caution.")

    result = predict(ticker, latest)
    print(f"\n  Ticker:     {result.ticker}")
    print(f"  Signal:     {result.signal}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Date:       {latest.name.date()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", help="Predict this ticker with the pooled model")
    parser.add_argument("--train", action="store_true", help="Train + save the pooled model")
    parser.add_argument("--validate", action="store_true", help="Pooled walk-forward + holdout check")
    parser.add_argument("--news", action="store_true", help="Include per-day news sentiment (slow, hits the LLM API)")
    parser.add_argument("--rebuild", action="store_true", help="Ignore the per-ticker cache and rebuild every frame")
    parser.add_argument("--refetch", action="store_true", help="Re-download OHLCV instead of reusing cached bars")
    parser.add_argument("--holdout-per-sector", type=int, default=1, help="Tickers held out per sector (default 1)")
    parser.add_argument("--splits", type=int, default=2,
                        help="Walk-forward splits for --validate (default 2; more splits = smaller, noisier folds)")
    args = parser.parse_args()

    if args.validate:
        do_validate(args.news, args.rebuild, args.refetch, args.holdout_per_sector, args.splits)
    elif args.train:
        do_train(args.news, args.rebuild, args.refetch, args.holdout_per_sector)
    elif args.ticker:
        do_predict(args.ticker, args.news, args.refetch)
    else:
        parser.error("give a TICKER to predict, or use --train / --validate")
