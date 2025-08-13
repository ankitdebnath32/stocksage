from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from app.services.prediction_service import (
    train_xgboost_classifier,
    predict_next_day_direction,
)
from app.utils.data_fetcher import get_financial_news, get_stock_data
from app.services.sentiment_service import annotate_news_sentiment
from app.utils.preprocess import aggregate_daily_sentiment

router = APIRouter()

class TrainRequest(BaseModel):
    ticker: str
    news_query: Optional[str] = None
    news_days: int = 90
    period: str = "12mo"

class PredictRequest(BaseModel):
    ticker: str
    period: str = "2mo"
    news_days: int = 7

@router.post("/train")
def train_endpoint(req: TrainRequest):
    try:
        # 1) Fetch news
        raw_news = get_financial_news(api_key=None, query=req.news_query or req.ticker, days=req.news_days)
        if raw_news.empty:
            raise HTTPException(status_code=400, detail="No news fetched for given query")

        # 2) Annotate sentiment (uses FinBERT) - this adds 'sentiment' column
        news_with_sentiment = annotate_news_sentiment(raw_news)

        # 3) Train model
        result = train_xgboost_classifier(req.ticker, news_with_sentiment, period=req.period)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        # Fetch recent price history
        price_df = get_stock_data(req.ticker, period=req.period)
        if price_df.empty:
            raise HTTPException(status_code=400, detail="No price data fetched for ticker")

        # Fetch recent news and annotate
        raw_news = get_financial_news(api_key=None, query=req.ticker, days=req.news_days)
        news_with_sentiment = annotate_news_sentiment(raw_news)
        sentiment_daily = aggregate_daily_sentiment(news_with_sentiment, date_col='publishedAt', sentiment_col='sentiment')

        pred = predict_next_day_direction(req.ticker, price_df, sentiment_daily)
        return {"status": "ok", "prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
