import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DIR, HISTORY_START, DEFAULT_INTERVAL
from src.storage import atomic_path

MANIFEST_PATH = RAW_DIR / "_manifest.json"

FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 2.0

# The manifest is a read-modify-write with no lock, so overlapping fetchers can drop
# each other's entries. Fetching is a single foreground command today; documented limit.


class FetchError(RuntimeError):
    pass


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}  # unreadable manifest just means everything looks stale


def _save_manifest(manifest: dict) -> None:
    with atomic_path(MANIFEST_PATH) as tmp:
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def manifest_entry(ticker: str) -> dict:
    # Public because the derived-feature cache fingerprints it: a frame built from one
    # download must stop being trusted the moment that download is replaced, and
    # `fetched_at` is what makes that visible.
    return _load_manifest().get(ticker.upper(), {})


def _download(ticker: str, start: str, interval: str) -> pd.DataFrame:
    # Retries because yfinance fails transiently often enough over ~90 tickers.
    last_error = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            # auto_adjust: unadjusted closes would put a fake -50% return on every split
            # date, wrecking both the momentum formation return and the vol estimate.
            df = yf.download(ticker, start=start, interval=interval,
                             auto_adjust=True, progress=False, threads=False)
            if not df.empty:
                return df
            last_error = ValueError("empty frame returned")
        except Exception as e:
            last_error = e
        if attempt < FETCH_ATTEMPTS:
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
    raise FetchError(f"{ticker}: no data after {FETCH_ATTEMPTS} attempts ({last_error})")


def fetch_and_save(ticker: str, start: str = HISTORY_START,
                   interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    ticker = ticker.upper()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = _download(ticker, start, interval)

    # yfinance returns MultiIndex columns for a single ticker in recent versions.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[~df.index.duplicated(keep="last")].sort_index()

    path = RAW_DIR / f"{ticker}.parquet"
    with atomic_path(path) as tmp:
        df.to_parquet(tmp)

    # Without this record a change to HISTORY_START would be silently ignored and the
    # model would train on whatever window happened to be on disk.
    manifest = _load_manifest()
    manifest[ticker] = {
        "start": start,
        "interval": interval,
        "rows": len(df),
        "first_bar": str(df.index[0].date()),
        "last_bar": str(df.index[-1].date()),
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
    }
    _save_manifest(manifest)
    return df


def is_current(ticker: str, start: str = HISTORY_START,
               interval: str = DEFAULT_INTERVAL) -> bool:
    # Answers only "was this file built for the window I want". How *recent* the last bar
    # is stays an execution-time concern, handled where live signals are formed.
    ticker = ticker.upper()
    if not (RAW_DIR / f"{ticker}.parquet").exists():
        return False
    entry = _load_manifest().get(ticker)
    return bool(entry) and entry.get("start") == start and entry.get("interval") == interval


def load_bars(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data file for {ticker}. Run fetch_and_save('{ticker}') first.")
    return pd.read_parquet(path)
