import os
from backend.app.utils.data_fetcher import get_stock_data, get_financial_news
from backend.app.services.sentiment_service import analyze_news_sentiment

API_KEY = os.getenv("NEWS_API_KEY")  # Now reads from environment

if not API_KEY:
    raise EnvironmentError("Please set the NEWS_API_KEY environment variable.")

# Fetch stock data
print(get_stock_data("AAPL").head())

# Fetch news + analyze sentiment
news_df = get_financial_news(API_KEY, query="Tesla stock")
sentiment_df = analyze_news_sentiment(news_df)
print(sentiment_df.head())
