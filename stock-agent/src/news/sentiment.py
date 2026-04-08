"""
LLM-assisted news sentiment scoring via Claude API.
Returns a numeric score per headline and an aggregate.
"""

import anthropic
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def score_headline(ticker: str, headline: str) -> float:
    """
    Ask Claude to score a headline's sentiment for a ticker.
    Returns float in [-1.0, 1.0]: -1 = very bearish, +1 = very bullish.
    """
    prompt = (
        f"You are a financial analyst. Rate the sentiment of this headline for {ticker} stock "
        f"on a scale from -1.0 (very bearish) to +1.0 (very bullish). "
        f"Reply with ONLY a number, nothing else.\n\n"
        f"Headline: {headline}"
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap + fast for bulk scoring
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    try:
        score = float(raw)
        return max(-1.0, min(1.0, score))  # clamp to [-1, 1]
    except ValueError:
        return 0.0


def aggregate_sentiment(ticker: str, headlines: list[str]) -> dict:
    """
    Score all headlines and return aggregate stats.
    Returns dict with mean_score, n_bullish, n_bearish, n_neutral.
    """
    if not headlines:
        return {"mean_score": 0.0, "n_bullish": 0, "n_bearish": 0, "n_neutral": 0, "count": 0}

    scores = [score_headline(ticker, h) for h in headlines]
    return {
        "mean_score": sum(scores) / len(scores),
        "n_bullish": sum(1 for s in scores if s > 0.2),
        "n_bearish": sum(1 for s in scores if s < -0.2),
        "n_neutral": sum(1 for s in scores if -0.2 <= s <= 0.2),
        "count": len(scores),
    }
