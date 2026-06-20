"""
LLM-assisted news sentiment scoring via Claude API.
Caches per-headline scores to disk so reruns don't re-bill the API.
"""

import anthropic
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ANTHROPIC_API_KEY, DATA_DIR

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

CACHE_PATH = DATA_DIR / "sentiment_cache.json"

MODEL = "claude-haiku-4-5-20251001"  # cheap + fast for bulk scoring
PROMPT_TEMPLATE = (
    "You are a financial analyst. Rate the sentiment of this headline for {ticker} stock "
    "on a scale from -1.0 (very bearish) to +1.0 (very bullish). "
    "Reply with ONLY a number, nothing else.\n\n"
    "Headline: {headline}"
)
# Tag cached scores with the model + prompt so editing either invalidates stale entries
# instead of silently serving scores produced by a different scoring function.
_SCORER_VERSION = hashlib.sha1(f"{MODEL}|{PROMPT_TEMPLATE}".encode("utf-8")).hexdigest()[:8]


def _load_cache() -> dict[str, float]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict[str, float]) -> None:
    # Write to a temp file in the same dir, then atomically replace, so an interrupted
    # write can't truncate the cache and wipe every accumulated score.
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(CACHE_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, CACHE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _cache_key(ticker: str, headline: str) -> str:
    return hashlib.sha1(f"{_SCORER_VERSION}|{ticker.upper()}|{headline}".encode("utf-8")).hexdigest()


def _parse_score(message) -> float | None:
    """Extract a clamped [-1, 1] score from a response, or None if it can't be parsed."""
    try:
        raw = message.content[0].text.strip()
    except (IndexError, AttributeError):
        return None  # empty content list or a non-text block (e.g. refusal)
    try:
        return max(-1.0, min(1.0, float(raw)))
    except ValueError:
        return None


def score_headline(ticker: str, headline: str, cache: dict[str, float] | None = None) -> float:
    """
    Score a single headline in [-1, 1]. Uses (and updates) cache if provided.
    Empty or unparseable responses return a neutral 0.0 but are NOT cached, so a
    transient failure is retried next run instead of being pinned to 0.0 forever.
    """
    if cache is not None:
        key = _cache_key(ticker, headline)
        if key in cache:
            return cache[key]

    prompt = PROMPT_TEMPLATE.format(ticker=ticker, headline=headline)

    message = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    score = _parse_score(message)
    if score is None:
        return 0.0

    if cache is not None:
        cache[_cache_key(ticker, headline)] = score

    return score


def sentiment_by_date(ticker: str, dated_headlines: list[tuple[date, str]]) -> dict[date, dict]:
    """
    Group headlines by date, score each, return per-date aggregate stats.
    Persists cache once at the end to amortize disk writes.
    """
    if not dated_headlines:
        return {}

    cache = _load_cache()
    cache_size_before = len(cache)
    by_date: dict[date, list[float]] = defaultdict(list)

    for d, headline in dated_headlines:
        by_date[d].append(score_headline(ticker, headline, cache))

    if len(cache) != cache_size_before:
        _save_cache(cache)

    return {
        d: {
            "mean_score": sum(scores) / len(scores),
            "n_bullish": sum(1 for s in scores if s > 0.2),
            "n_bearish": sum(1 for s in scores if s < -0.2),
            "count": len(scores),
        }
        for d, scores in by_date.items()
    }
