"""
Dataset assembly.

build_ticker_frame   one ticker's features + forward labels, tagged with a `ticker`
                     column and keeping its DatetimeIndex.
build_pooled_dataset stack many tickers into one frame, caching each ticker's finished
                     frame so a rebuild does not recompute everything.

The pooled frame is a ragged panel — names that listed after HISTORY_START simply have
no rows before their IPO. That is fine everywhere downstream, because training and
evaluation split on dates, not on row position.

Cache invalidation is fingerprint-based, and that is the load-bearing part. A derived
frame depends on the raw bars it was built from *and* on every constant that shaped
the features and labels. Trusting it because the raw manifest still names the same
start date checks almost none of that: `--fetch --refetch` replaces the bars and
leaves the manifest looking identical in every field the old check read, so the next
training run happily reuses features derived from data that no longer exists. Changing
FORWARD_DAYS was worse — the model trained on the old labels while its saved payload
advertised the new horizon. Each cached frame therefore records a hash of everything
it depends on, and is reused only when that hash still matches.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (ATR_PERIOD, EWMA_LAMBDA, FEATURES_DIR, FORWARD_DAYS,
                    RETURN_HORIZONS, RV_WINDOWS)
from src.data.fetcher import FetchError, fetch_and_save, is_current, load_bars, manifest_entry
from src.features.technical import build_features
from src.features.volatility import build_volatility_features
from src.labels.target import add_labels
from src.storage import atomic_path

CACHE_DIR = FEATURES_DIR / "pooled"
FINGERPRINT_PATH = CACHE_DIR / "_fingerprints.json"

# Bumped whenever the shape of a built frame changes — a new column, a redefined
# feature, a different label convention. The constants below catch a changed
# *parameter*; this catches a changed *computation*, which no parameter records.
FEATURE_SCHEMA_VERSION = 2

# Data-access failures that mean "this ticker is unavailable" and are a reason to
# skip it. Anything else — a TypeError in a feature, an AttributeError from a
# pandas upgrade — is a defect in this repo, and swallowing it silently trains the
# model on whichever tickers happened to dodge the bug.
SKIPPABLE = (FetchError, FileNotFoundError, KeyError, ValueError)


def _fingerprint(ticker: str) -> str:
    """
    A hash of everything a built frame depends on.

    The raw manifest entry covers the bars themselves — including `fetched_at`, so a
    refetch invalidates derived frames even when the window is unchanged. The config
    constants cover the feature and label definitions. The schema version covers
    changes to the code that no constant would show.
    """
    payload = {
        "raw": manifest_entry(ticker),
        "schema": FEATURE_SCHEMA_VERSION,
        "forward_days": FORWARD_DAYS,
        "rv_windows": list(RV_WINDOWS),
        "ewma_lambda": EWMA_LAMBDA,
        "atr_period": ATR_PERIOD,
        "return_horizons": list(RETURN_HORIZONS),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _load_fingerprints() -> dict:
    if not FINGERPRINT_PATH.exists():
        return {}
    try:
        return json.loads(FINGERPRINT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}  # unreadable index just means every cached frame looks stale


def _save_fingerprints(fingerprints: dict) -> None:
    with atomic_path(FINGERPRINT_PATH) as tmp:
        tmp.write_text(json.dumps(fingerprints, indent=2, sort_keys=True))


def build_ticker_frame(ticker: str, refetch: bool = False) -> pd.DataFrame:
    """Build one ticker's complete feature + label frame."""
    ticker = ticker.upper()
    if refetch or not is_current(ticker):
        fetch_and_save(ticker)

    df = load_bars(ticker)
    df = build_features(df)
    df = build_volatility_features(df)
    df = add_labels(df)
    df["ticker"] = ticker
    return df


def build_pooled_dataset(
    tickers: list[str],
    force_rebuild: bool = False,
    refetch: bool = False,
) -> pd.DataFrame:
    """
    Build and vertically stack frames for many tickers into one pooled dataset.

    Each finished frame is cached to FEATURES_DIR/pooled/<TICKER>.parquet alongside a
    fingerprint of everything it was derived from; a cache is reused only when that
    fingerprint still matches. See the module docstring for why the previous check —
    "does the raw manifest still name this start date" — was not enough.

    Tickers that cannot be read are skipped with a warning. Tickers that fail for any
    *other* reason are not: a bug in a feature builder is not a data problem, and
    catching it here would quietly train the model on whichever names dodged it.

    The returned frame carries a `coverage` entry in `frame.attrs` recording what was
    requested against what actually loaded, so a caller can enforce a policy on it
    instead of reading the log.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprints = _load_fingerprints()

    frames, loaded, failures = [], [], []
    requested = [t.upper() for t in tickers]

    bar = tqdm(requested, desc="pooled dataset", unit="ticker")
    for ticker in bar:
        bar.set_postfix_str(ticker)
        cache = CACHE_DIR / f"{ticker}.parquet"

        try:
            reusable = (cache.exists() and not force_rebuild and not refetch
                        and is_current(ticker)
                        and fingerprints.get(ticker) == _fingerprint(ticker))
            if reusable:
                frames.append(pd.read_parquet(cache))
                loaded.append(ticker)
                continue

            df = build_ticker_frame(ticker, refetch=refetch)
            with atomic_path(cache) as tmp:
                df.to_parquet(tmp)
            # Recomputed after the build: a refetch inside `build_ticker_frame`
            # rewrites the manifest, and the fingerprint has to describe the bars
            # this frame was actually derived from.
            fingerprints[ticker] = _fingerprint(ticker)
            frames.append(df)
            loaded.append(ticker)
        except SKIPPABLE as e:
            failures.append(ticker)
            bar.write(f"  skip {ticker}: {type(e).__name__}: {e}")
    bar.close()

    _save_fingerprints(fingerprints)

    if not frames:
        raise ValueError("No ticker frames could be built for the pooled dataset.")

    pooled = pd.concat(frames).sort_index()
    pooled.attrs["coverage"] = {
        "requested": len(requested),
        "loaded": sorted(loaded),
        "failed": sorted(failures),
    }
    print(f"  stacked {len(frames)}/{len(requested)} tickers -> {len(pooled):,} rows, "
          f"{pooled.index.min().date()} to {pooled.index.max().date()}")
    if failures:
        print(f"  failed: {', '.join(sorted(failures))}")
    return pooled
