"""
Entry point for the volatility-forecasting stage.

    python pipeline.py --fetch           # Phase 0: pull ~20y of adjusted bars for the universe
    python pipeline.py --vol-validate    # Phase 1 gate: xgb vs rw / ewma / har-rv
    python pipeline.py --train-vol       # fit and save the production forecaster

Direction (which names, long or short) is not decided here — that is Phase 3's
momentum signal. This stage answers only "how volatile is each name about to be",
which is what the sizing layer needs.
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from src.data.dataset import build_pooled_dataset
from src.data.fetcher import fetch_and_save, is_current
from src.data.universe import ALL_TICKERS, BENCHMARK
from src.models.vol_forecast import validate_vol_forecast, train_vol_model


def do_fetch(refetch: bool) -> None:
    """Pull bars for the whole universe plus the benchmark, skipping what is already current."""
    tickers = ALL_TICKERS + [BENCHMARK]
    print(f"Fetching {len(tickers)} tickers...")

    fetched, skipped, failed = 0, 0, []
    for ticker in tickers:
        if not refetch and is_current(ticker):
            skipped += 1
            continue
        try:
            df = fetch_and_save(ticker)
            fetched += 1
            print(f"  {ticker:<6} {len(df):>5,} bars  {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:
            failed.append(ticker)
            print(f"  {ticker:<6} FAILED: {e}")

    print(f"\nFetched {fetched}, already current {skipped}, failed {len(failed)}")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")


def do_vol_validate(n_splits: int, force_rebuild: bool, refetch: bool) -> None:
    print("\n[1/2] Building pooled dataset...")
    df = build_pooled_dataset(ALL_TICKERS, force_rebuild=force_rebuild, refetch=refetch)

    print("\n[2/2] Walk-forward volatility forecast comparison...")
    validate_vol_forecast(df, n_splits=n_splits)


def do_train_vol(kind: str, force_rebuild: bool, refetch: bool) -> None:
    print("\n[1/2] Building pooled dataset...")
    df = build_pooled_dataset(ALL_TICKERS, force_rebuild=force_rebuild, refetch=refetch)

    print(f"\n[2/2] Fitting the {kind} forecaster on the full history...")
    train_vol_model(df, kind=kind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true",
                        help="Download bars for the universe (Phase 0)")
    parser.add_argument("--vol-validate", action="store_true",
                        help="Walk-forward vol forecast vs baselines — the Phase 1 gate")
    parser.add_argument("--train-vol", action="store_true",
                        help="Fit and save the production vol forecaster")
    parser.add_argument("--kind", choices=["xgb", "har"], default="xgb",
                        help="Which forecaster --train-vol saves (default xgb; use har if the gate failed)")
    parser.add_argument("--splits", type=int, default=5,
                        help="Walk-forward splits (default 5)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Ignore the per-ticker feature cache and rebuild every frame")
    parser.add_argument("--refetch", action="store_true",
                        help="Re-download bars instead of reusing cached ones")
    args = parser.parse_args()

    if args.fetch:
        do_fetch(args.refetch)
    elif args.vol_validate:
        do_vol_validate(args.splits, args.rebuild, args.refetch)
    elif args.train_vol:
        do_train_vol(args.kind, args.rebuild, args.refetch)
    else:
        parser.error("choose one of --fetch / --vol-validate / --train-vol")
        sys.exit(2)
