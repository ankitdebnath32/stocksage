"""
Quick script to train and save XGBoost models for a given ticker.
Usage:
    python train_ticker.py AAPL
"""
 
import sys
import pandas as pd
from ..services.prediction_service import train_xgboost_classifier
from ..utils.data_fetcher import get_financial_news
 
def load_news_for_ticker(ticker, days=90):
    # This function should call your data_fetcher and sentiment_service pipeline to return
    # a DataFrame with 'publishedAt' and 'sentiment' columns.
    # For demonstration this raises an error to remind to plug pipeline.
    raise NotImplementedError("Please implement news fetching + sentiment annotation pipeline and return DataFrame with 'publishedAt' and 'sentiment'")
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_ticker.py <TICKER>")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    # The user must implement or call the sentiment pipeline here:
    news_df = load_news_for_ticker(ticker, days=180)
    result = train_xgboost_classifier(ticker, news_df, period="12mo")
    print("Training result:", result)