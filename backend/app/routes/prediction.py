from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
 
from ..services.prediction_service import train_xgboost_classifier, predict_next_day_direction, load_model_for_ticker
from ..utils.data_fetcher import get_financial_news, get_stock_data
 
router = APIRouter()
 
class TrainRequest(BaseModel):
    ticker: str
    news_query: Optional[str] = None
    news_days: int = 30
    period: str = "12mo"
 
class PredictRequest(BaseModel):
    ticker: str
    period: str = "2mo"  # window of recent price data to build features
    news_days: int = 7
 
@router.post("/train")
def train_endpoint(req: TrainRequest):
    try:
        # Fetch news (Phase 1) - ensure your data_fetcher returns publishedAt and sentiment (or that you run sentiment pipeline)
        raw_news = get_financial_news(api_key="YOUR_NEWS_API_KEY", query=req.news_query or req.ticker, days=req.news_days)
        # NOTE: raw_news here must be already enriched with 'sentiment' column (e.g., run sentiment_service)
        # For simplicity in this endpoint we assume sentiment is present. In production hook pipeline to annotate.
        result = train_xgboost_classifier(req.ticker, raw_news, period=req.period)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@router.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        # Fetch recent price history
        price_df = get_stock_data(req.ticker, period=req.period)
        # Fetch recent news and run sentiment pipeline (in production you should call the sentiment service)
        raw_news = get_financial_news(api_key="YOUR_NEWS_API_KEY", query=req.ticker, days=req.news_days)
        # For now we assume raw_news has 'sentiment' labels (if not, run sentiment_service on raw_news first).
        # Aggregate sentiment
        from ..utils.preprocess import aggregate_daily_sentiment
        sentiment_daily = aggregate_daily_sentiment(raw_news, date_col='publishedAt', sentiment_col='sentiment')
        pred = predict_next_day_direction(req.ticker, price_df, sentiment_daily)
        return {"status": "ok", "prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))