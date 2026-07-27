"""
FastAPI app — exposes prediction endpoints.
Run with: uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.fetcher import fetch_and_save, load_bars
from src.features.technical import build_features
from src.features.relative_strength import add_relative_strength
from src.data.dataset import build_pooled_dataset, split_universe
from src.models.predict import predict
from src.models.train import cross_sectional_validate, walk_forward_holdout, train_pooled_model

app = FastAPI(title="Stock Trend Agent", version="0.1.0")


class PredictionResponse(BaseModel):
    ticker: str
    signal: str
    confidence: float


class TrainResponse(BaseModel):
    status: str
    n_tickers: int
    n_rows: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def get_prediction(ticker: str):
    """Get the current trend prediction for a ticker."""
    ticker = ticker.upper()
    try:
        df = load_bars(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}. Call /refresh first.")

    df = build_features(df)
    df = add_relative_strength(df, ticker)

    latest = df.dropna().iloc[-1]

    try:
        result = predict(ticker, latest)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No pooled model. Call POST /train first.")

    return PredictionResponse(ticker=ticker, signal=result.signal, confidence=round(result.confidence, 4))


@app.post("/refresh/{ticker}")
def refresh_data(ticker: str):
    """Re-fetch latest price data for a ticker."""
    ticker = ticker.upper()
    try:
        path = fetch_and_save(ticker)
        return {"ticker": ticker, "status": "refreshed", "path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse)
def train_model(news: bool = False):
    """Build the pooled dataset across the training universe and train the single pooled model."""
    try:
        train_tickers, _ = split_universe()
        df = build_pooled_dataset(train_tickers, with_news=news)
        train_pooled_model(df)
        return TrainResponse(status="trained", n_tickers=int(df["ticker"].nunique()), n_rows=len(df))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate")
def validate_model(news: bool = False):
    """Pooled walk-forward validation over the training universe, plus the holdout-ticker check."""
    try:
        train_tickers, holdout_tickers = split_universe()
        train_df = build_pooled_dataset(train_tickers, with_news=news)
        folds = cross_sectional_validate(train_df)
        holdout_df = build_pooled_dataset(holdout_tickers, with_news=news)
        holdout = walk_forward_holdout(train_df, holdout_df)
        return {"folds": folds, "holdout": holdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
