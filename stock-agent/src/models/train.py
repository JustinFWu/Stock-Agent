"""
Training pipeline for the pooled cross-sectional RANKING model.

The model is an XGBoost regressor trained on the cross-sectional target
`xs_demeaned_return` (each name's 5-day forward return minus the universe mean that
day). It learns to ORDER names within a date — separate winners from losers — rather
than call absolute up/down. Validation is therefore rank-based (rank IC and long/short
spread), not classification AUC.
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, FORWARD_DAYS

# Feature columns to use (excludes OHLCV raw prices and label)
FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "sma10_vs_sma20", "sma20_vs_sma50",
    "sma_10_slope", "sma_20_slope", "sma_50_slope",
    "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50",
    "rsi", "rsi_overbought", "rsi_oversold",
    "macd_norm", "macd_signal_norm", "macd_hist_norm", "macd_hist_slope_norm",
    "atr_pct",
    "volume_ratio_20d", "volume_ratio_5d",
    "gap", "daily_range", "close_position",
    "rs_vs_sector_5d", "rs_vs_sector_20d",
    "news_sentiment", "news_bullish_pct", "news_bearish_pct", "news_has_news",
]

LABEL_COL = "xs_demeaned_return"  # cross-sectional target: forward return minus the day's universe mean


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return only feature cols that exist in this dataframe."""
    return [c for c in FEATURE_COLS if c in df.columns]


def _build_model() -> XGBRegressor:
    """The single XGBoost configuration shared by validation and the saved model. A
    regressor on the demeaned forward return — its output orders names within a date."""
    return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        objective="reg:squarederror",
                        random_state=42, verbosity=0)


def _date_windows(dates, n_splits: int = 5, embargo: int = FORWARD_DAYS):
    """
    Yield (train_dates, test_dates) for each expanding-window split over the sorted
    UNIQUE dates — never shuffling, time order preserved. Splitting on dates (not row
    position) is what makes this valid for a pooled multi-ticker frame: 'train on the
    past, test on the next block' then holds across every ticker simultaneously, even
    though many rows share each date.

    Embargo: the label looks `embargo` (FORWARD_DAYS) trading days ahead, so the last
    `embargo` train dates would have labels computed from closes inside the test block.
    We drop those dates so train and test never overlap — without this gap, AUC is
    inflated by leakage. (Assumes a shared trading calendar, true for US equities here.)
    """
    dates = np.sort(pd.Index(dates).unique().values)
    fold_size = len(dates) // (n_splits + 1)
    if fold_size == 0:
        return
    for i in range(1, n_splits + 1):
        train_cutoff = i * fold_size - embargo
        if train_cutoff <= 0:
            continue
        train_dates = dates[:train_cutoff]
        test_dates = dates[i * fold_size : (i + 1) * fold_size]
        if len(test_dates) == 0:
            continue
        yield i, train_dates, test_dates


def _walk_forward_folds(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5,
                        embargo: int = FORWARD_DAYS):
    """
    Yield (fold_index, train, test, score) for each date-based expanding-window split
    (see _date_windows). `score` is the regressor's predicted demeaned 5-day return for
    each test row — higher means "expected to out-rank its peers that day". Callers pass
    an already-cleaned df (NaN labels/features dropped) and turn the scores into ranking
    metrics. Works for a single ticker (one row per date) or a pooled frame (many tickers
    per date).
    """
    for i, train_dates, test_dates in _date_windows(df.index, n_splits, embargo):
        train = df[df.index.isin(train_dates)]
        test = df[df.index.isin(test_dates)]
        if len(train) == 0 or len(test) == 0:
            continue

        model = _build_model()
        model.fit(train[feature_cols], train[LABEL_COL])
        score = model.predict(test[feature_cols])
        yield i, train, test, score


def _daily_rank_ic(oos: pd.DataFrame) -> pd.Series:
    """
    Per-date rank IC: Spearman correlation between the model's ranking score and the
    realized forward return across that date's cross-section. Spearman == Pearson on
    ranks, so we rank both columns within the date and correlate (no scipy needed).

    Dates with fewer than 3 names, or with no spread in either column, are skipped
    (correlation is undefined / meaningless there). Returns one IC per usable date,
    indexed by date.
    """
    ics = {}
    for day, g in oos.groupby(level=0):
        if len(g) < 3 or g["score"].nunique() < 2 or g["future_return"].nunique() < 2:
            continue
        ics[day] = g["score"].rank().corr(g["future_return"].rank())
    return pd.Series(ics, dtype=float)


def _daily_long_short(oos: pd.DataFrame, quantile: float = 0.2) -> pd.Series:
    """
    Per-date long-short spread: mean realized forward return of the top-`quantile` names
    by score minus that of the bottom-`quantile`, forming an equal-weight, dollar-neutral
    book each day. This is the economic payoff of the ranking, before costs.

    k = round(n * quantile), floored at 1; dates too small to hold two disjoint baskets
    (2k > n) are skipped. Returns one spread per usable date, indexed by date.
    """
    spreads = {}
    for day, g in oos.groupby(level=0):
        n = len(g)
        k = max(1, int(round(n * quantile)))
        if 2 * k > n:
            continue
        ordered = g.sort_values("score")["future_return"].values
        short_leg = ordered[:k].mean()
        long_leg = ordered[-k:].mean()
        spreads[day] = long_leg - short_leg
    return pd.Series(spreads, dtype=float)


def cross_sectional_validate(
    df: pd.DataFrame,
    n_splits: int = 5,
    quantile: float = 0.2,
    forward_days: int = FORWARD_DAYS,
) -> dict:
    """
    The go/no-go gate: does the model separate winners from losers within each date?

    Reuses the walk-forward folds and the pooled regressor; the per-row score is the
    predicted demeaned 5-day return. Within each date we measure:
      * rank IC   - Spearman(score, realized forward return) across the cross-section
      * L/S spread - top-quantile minus bottom-quantile realized forward return

    Reported both per fold (to see stability, per the memory note) and pooled across all
    OOS dates: mean rank IC, IC information ratio (mean/std — the "stable across folds"
    check), hit rate of positive-IC days, and the long-short book's mean daily spread
    with an annualised Sharpe.

    IMPORTANT: consecutive days' books overlap (a `forward_days`-day horizon means
    `forward_days` overlapping books live at once), so daily spreads are autocorrelated
    and the Sharpe is optimistic — read it as a screen, not a tradeable number.
    """
    if "future_return" not in df.columns:
        raise ValueError("df has no 'future_return'; call add_label() before ranking.")

    df = df.dropna(subset=[LABEL_COL, "future_return"])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    fold_rows = []
    oos_parts = []
    for i, _, test, score in _walk_forward_folds(df, feature_cols, n_splits):
        part = pd.DataFrame(
            {"score": score, "future_return": test["future_return"].values},
            index=test.index,
        )
        oos_parts.append(part)
        fold_ic = _daily_rank_ic(part)
        fold_ls = _daily_long_short(part, quantile)
        if len(fold_ic):
            fold_rows.append({
                "fold": i,
                "mean_ic": float(fold_ic.mean()),
                "ls_spread": float(fold_ls.mean()) if len(fold_ls) else float("nan"),
                "n_days": int(len(fold_ic)),
                "n_rows": len(part),
            })

    if not oos_parts:
        print("No out-of-sample folds to rank.")
        return {"n_days": 0}

    oos = pd.concat(oos_parts)
    ic = _daily_rank_ic(oos)
    ls = _daily_long_short(oos, quantile)

    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    ic_ir = float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else float("nan")
    ic_hit = float((ic > 0).mean()) if len(ic) else float("nan")

    ls_mean = float(ls.mean()) if len(ls) else float("nan")
    ls_std = float(ls.std(ddof=1)) if len(ls) > 1 else 0.0
    periods_per_year = 252 / forward_days
    ls_sharpe = (ls_mean / ls_std) * np.sqrt(periods_per_year) if ls_std > 0 else float("nan")

    if fold_rows:
        print("\n=== Cross-sectional ranking validation (out-of-sample, walk-forward) ===")
        print(pd.DataFrame(fold_rows).to_string(index=False))
    print(f"\n  ranking score:        predicted demeaned {forward_days}-day return (pooled regressor)")
    print(f"  long/short quantile:  top/bottom {quantile:.0%} each date")
    print(f"  rank-IC days:         {len(ic)}")
    print(f"  mean rank IC:         {mean_ic:.4f}   (target > ~0.03)")
    print(f"  IC info ratio:        {ic_ir:.2f}      (mean/std across days - stability)")
    print(f"  positive-IC days:     {ic_hit:.1%}")
    print(f"  mean L/S spread ({forward_days}d): {ls_mean:.4%}")
    print(f"  L/S Sharpe (annual.): {ls_sharpe:.2f}   (optimistic - overlapping books; target > ~1)")

    return {
        "n_days": int(len(ic)),
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "ic_hit_rate": ic_hit,
        "ls_mean_spread": ls_mean,
        "ls_sharpe": ls_sharpe,
        "quantile": quantile,
        "folds": fold_rows,
    }


def walk_forward_holdout(train_df: pd.DataFrame, holdout_df: pd.DataFrame,
                         n_splits: int = 5, embargo: int = FORWARD_DAYS) -> dict:
    """
    The honesty check for a pooled model: does it generalize to tickers it never saw?

    For each date-based fold, fit on the TRAIN-universe rows up to the cutoff and rank the
    HOLDOUT tickers' rows in the next block. So both axes stay clean: the model is tested
    on out-of-sample dates AND out-of-sample tickers, with the same embargo as
    cross_sectional_validate. The metric is the same rank IC — does the ordering hold up
    on names the model never trained on? Pass holdout_df built from tickers excluded from
    train_df (see split_universe). Report only at the very end, after all tuning is done.
    """
    feature_cols = [c for c in get_available_features(train_df) if c in holdout_df.columns]
    train_df = train_df.dropna(subset=[LABEL_COL] + feature_cols)
    holdout_df = holdout_df.dropna(subset=[LABEL_COL, "future_return"] + feature_cols)

    all_dates = train_df.index.union(holdout_df.index)
    results = []
    for i, train_dates, test_dates in _date_windows(all_dates, n_splits, embargo):
        tr = train_df[train_df.index.isin(train_dates)]
        te = holdout_df[holdout_df.index.isin(test_dates)]
        if len(tr) == 0 or len(te) == 0:
            continue
        model = _build_model()
        model.fit(tr[feature_cols], tr[LABEL_COL])
        score = model.predict(te[feature_cols])
        part = pd.DataFrame(
            {"score": score, "future_return": te["future_return"].values},
            index=te.index,
        )
        ic = _daily_rank_ic(part)
        if not len(ic):
            continue
        results.append({
            "fold": i,
            "mean_ic": float(ic.mean()),
            "n_days": int(len(ic)),
            "n_train": len(tr),
            "n_test": len(te),
        })

    if not results:
        print("No evaluable holdout folds.")
        return {"folds": []}

    summary = pd.DataFrame(results)
    print("\n=== Holdout check (never-trained tickers, out-of-sample dates) ===")
    print(summary.to_string(index=False))
    print(f"\nHoldout mean rank IC: {summary['mean_ic'].mean():.4f}   (target > ~0.03)")
    return summary.to_dict("records")


POOLED_MODEL_PATH = MODELS_DIR / "pooled_xgb.joblib"


def train_pooled_model(df: pd.DataFrame) -> Path:
    """
    Train one regressor on the full pooled dataset (target xs_demeaned_return) and save
    it to POOLED_MODEL_PATH. The single model scores every ticker's expected demeaned
    5-day return; that score is only meaningful once ranked against the rest of the
    universe on the same date (see cross_sectional_validate).
    """
    df = df.dropna(subset=[LABEL_COL])
    feature_cols = get_available_features(df)
    df = df.dropna(subset=feature_cols)

    model = _build_model()
    model.fit(df[feature_cols], df[LABEL_COL])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, POOLED_MODEL_PATH)
    print(f"Pooled model saved -> {POOLED_MODEL_PATH}  ({len(df)} rows, {len(feature_cols)} features)")
    return POOLED_MODEL_PATH
