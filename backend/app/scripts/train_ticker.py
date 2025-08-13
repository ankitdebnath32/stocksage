"""
Train a model for a ticker end-to-end:
- Fetch news via NewsAPI
- Annotate sentiment with FinBERT
- Fetch price data via yfinance
- Train XGBoost classifier and save model

Usage:
    cd stocksage/backend
    python -m app.scripts.train_ticker AAPL --news_days 180 --period 12mo
"""

import argparse
import sys
import os
import pandas as pd
from app.services.prediction_service import train_xgboost_classifier
from app.utils.data_fetcher import get_financial_news
from app.services.sentiment_service import annotate_news_sentiment

def load_and_annotate_news(query: str, days: int = 180) -> pd.DataFrame:
    df = get_financial_news(api_key=None, query=query, days=days)
    if df is None or df.empty:
        print("No news fetched. Exiting.")
        return pd.DataFrame()
    annotated = annotate_news_sentiment(df)
    return annotated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--news_days", type=int, default=180, help="How many days of news to fetch")
    parser.add_argument("--period", type=str, default="12mo", help="Price history period for yfinance")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"Starting training for {ticker} ...")
    news_df = load_and_annotate_news(query=ticker, days=args.news_days)
    if news_df is None or news_df.empty:
        print("No news to train on. Make sure you set NEWS_API_KEY env var or pass api key in data_fetcher.")
        sys.exit(1)

    result = train_xgboost_classifier(ticker, news_df, period=args.period)
    print("Training finished. Result:")
    print(result)

if __name__ == "__main__":
    main()
