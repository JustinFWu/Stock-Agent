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

# Cache invalidation is fingerprint-based because the old check — "does the raw manifest
# still name this start date" — passed after `--fetch --refetch` replaced the bars, and a
# changed FORWARD_DAYS trained the model on old labels while advertising the new horizon.

# Bumped whenever the shape of a built frame changes — a new column, a redefined
# feature, a different label convention. The constants below catch a changed
# *parameter*; this catches a changed *computation*, which no parameter records.
FEATURE_SCHEMA_VERSION = 2

# Failures meaning "this ticker is unavailable", and a reason to skip it. Anything else — a
# TypeError in a feature, an AttributeError from a pandas upgrade — is a defect here, and
# swallowing it trains the model on whichever tickers happened to dodge the bug.
SKIPPABLE = (FetchError, FileNotFoundError, KeyError, ValueError)


def _fingerprint(ticker: str) -> str:
    # The manifest entry covers the bars, including `fetched_at`, so a refetch invalidates
    # derived frames even when the window is unchanged. The constants cover the feature and
    # label definitions; the schema version covers code changes no constant would show.
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
    # The pooled frame is ragged — names that listed after HISTORY_START have no rows
    # before their IPO — which is fine downstream because training and evaluation split
    # on dates, not row position.

    # `coverage` in frame.attrs records requested against loaded, so a caller can enforce
    # a policy on it instead of reading the log.
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
