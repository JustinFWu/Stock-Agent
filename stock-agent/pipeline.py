"""
Entry point.

    python pipeline.py --fetch           # Phase 0: pull ~20y of adjusted bars for the universe
    python pipeline.py --vol-validate    # Phase 1 gate: xgb vs rw / ewma / har-rv
    python pipeline.py --train-vol       # fit and save the production forecaster
    python pipeline.py --backtest        # Phase 2: run the cost-aware backtester

Direction (which names, long or short) is not decided here — that is Phase 3's
momentum signal. The volatility stage answers only "how volatile is each name
about to be", which is what the sizing layer needs; the backtest currently runs
an equal-weight placeholder, which exercises the machinery without pretending to
be a strategy.
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from config import NO_TRADE_BAND
from src.backtest.costs import ALPACA_COSTS, PESSIMISTIC_COSTS, ZERO_COSTS
from src.backtest.engine import run_backtest
from src.data.dataset import build_pooled_dataset
from src.data.fetcher import fetch_and_save, is_current
from src.backtest.metrics import format_relative_summary, summarize_relative
from src.data.panel import load_price_panel
from src.data.universe import BENCHMARK, UNIVERSE
from src.models.vol_forecast import validate_vol_forecast, train_vol_model
from src.strategy.weights import EqualWeightStrategy


def do_fetch(refetch: bool) -> None:
    """Pull bars for the whole universe plus the benchmark, skipping what is already current."""
    tickers = list(UNIVERSE.tickers) + [BENCHMARK]
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
    df = build_pooled_dataset(list(UNIVERSE.tickers), force_rebuild=force_rebuild, refetch=refetch)

    print("\n[2/2] Walk-forward volatility forecast comparison...")
    validate_vol_forecast(df, n_splits=n_splits)


def do_train_vol(kind: str, force_rebuild: bool, refetch: bool) -> None:
    print("\n[1/2] Building pooled dataset...")
    df = build_pooled_dataset(list(UNIVERSE.tickers), force_rebuild=force_rebuild, refetch=refetch)

    print(f"\n[2/2] Fitting the {kind} forecaster on the full history...")
    train_vol_model(df, kind=kind)


COST_MODELS = {"alpaca": ALPACA_COSTS, "pessimistic": PESSIMISTIC_COSTS, "zero": ZERO_COSTS}


def do_backtest(rebalance: str, cost_model: str, band: float, start: str | None) -> None:
    """
    Run the Phase 2 backtester over cached bars.

    The baseline run is not optional, and that is the point. On a survivorship-
    biased universe an absolute Sharpe means nothing — equal-weight buy-and-hold
    of these names scores about 0.91 with no signal in it — so the runner always
    computes the same-universe baseline and prints the difference. Enforcement by
    construction beats enforcement by discipline.

    Phase 2 has no strategy yet, so the strategy IS the baseline and the relative
    block is trivially zero. That is the correct output for a session whose
    deliverable is the machinery, and it makes the comparison visible from the day
    Phase 3's signal is dropped in.
    """
    strategy = EqualWeightStrategy()
    baseline = EqualWeightStrategy()

    print(f"\n[1/3] Loading price panel for {len(UNIVERSE.tickers)} names...")
    panel = load_price_panel(list(UNIVERSE.tickers))
    print(f"  {len(panel.tickers)} tickers, {panel.dates[0].date()} to {panel.dates[-1].date()}")

    settings = dict(universe=UNIVERSE, start=start, rebalance=rebalance,
                    costs=COST_MODELS[cost_model], no_trade_band=band)

    print(f"\n[2/3] Backtesting {strategy.name} ({cost_model} costs, {rebalance} rebalance)...")
    result = run_backtest(panel, strategy, **settings)

    print(f"\n[3/3] Baseline {baseline.name} on the same universe, engine and costs...")
    baseline_result = run_backtest(panel, baseline, **settings)

    print(result.describe())
    print()
    print(format_relative_summary(summarize_relative(result.equity, baseline_result.equity),
                          baseline.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true",
                        help="Download bars for the universe (Phase 0)")
    parser.add_argument("--vol-validate", action="store_true",
                        help="Walk-forward vol forecast vs baselines — the Phase 1 gate")
    parser.add_argument("--train-vol", action="store_true",
                        help="Fit and save the production vol forecaster")
    parser.add_argument("--backtest", action="store_true",
                        help="Run the cost-aware backtester (Phase 2)")
    parser.add_argument("--rebalance", choices=["D", "W", "M", "Q"], default="M",
                        help="Backtest rebalance frequency (default monthly)")
    parser.add_argument("--costs", choices=sorted(COST_MODELS), default="alpaca",
                        help="Which cost model the backtest charges (default alpaca)")
    parser.add_argument("--band", type=float, default=NO_TRADE_BAND,
                        help=f"No-trade band as a weight fraction (default {NO_TRADE_BAND})")
    parser.add_argument("--start", default=None,
                        help="First tradable date; history before it is still visible to the strategy")
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
    elif args.backtest:
        do_backtest(args.rebalance, args.costs, args.band, args.start)
    else:
        parser.error("choose one of --fetch / --vol-validate / --train-vol / --backtest")
        sys.exit(2)
