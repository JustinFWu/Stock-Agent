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
from src.labels.target import add_label
from src.models.predict import predict
from src.models.train import walk_forward_validate, train_final_model

app = FastAPI(title="Stock Trend Agent", version="0.1.0")


class PredictionResponse(BaseModel):
    ticker: str
    signal: str
    confidence: float


class TrainResponse(BaseModel):
    ticker: str
    status: str


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
        raise HTTPException(status_code=404, detail=f"No model for {ticker}. Call /train first.")

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


@app.post("/train/{ticker}", response_model=TrainResponse)
def train_model(ticker: str):
    """Train and save a model for a ticker."""
    ticker = ticker.upper()
    try:
        df = load_bars(ticker)
        df = build_features(df)
        df = add_relative_strength(df, ticker)
        df = add_label(df)
        train_final_model(df, ticker)
        return TrainResponse(ticker=ticker, status="trained")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate/{ticker}")
def validate_model(ticker: str):
    """Run walk-forward validation for a ticker and return metrics."""
    ticker = ticker.upper()
    try:
        df = load_bars(ticker)
        df = build_features(df)
        df = add_relative_strength(df, ticker)
        df = add_label(df)
        results = walk_forward_validate(df)
        return {"ticker": ticker, "folds": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
