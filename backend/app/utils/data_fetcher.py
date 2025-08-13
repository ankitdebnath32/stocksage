import yfinance as yf
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import os

# You can set NEWS_API_KEY as env var or pass into functions
NEWS_API_KEY = os.getenv("NEWS_API_KEY", None)

def get_stock_data(ticker: str, period: str = "12mo", interval: str = "1d") -> pd.DataFrame:
    """
    Returns yfinance history DataFrame with DatetimeIndex and columns Open, High, Low, Close, Volume
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df is None:
            return pd.DataFrame()
        df = df.reset_index()
        # normalize columns names if present
        return df
    except Exception as e:
        print("Error fetching stock data:", e)
        return pd.DataFrame()

def get_financial_news(api_key: str = None, query: str = "stock market", days: int = 30, page_size: int = 100) -> pd.DataFrame:
    """
    Fetch news articles using NewsAPI.org
    Returns DataFrame with fields: source, author, title, description, url, publishedAt, content
    NOTE: if you don't supply api_key, function will attempt to use NEWS_API_KEY env var.
    """
    key = api_key or NEWS_API_KEY
    if not key:
        # Return empty df instead of raising, so caller can decide
        print("No NewsAPI key provided. Returning empty DataFrame.")
        return pd.DataFrame(columns=['source','author','title','description','url','publishedAt','content'])

    try:
        newsapi = NewsApiClient(api_key=key)
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        all_articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            page_size=page_size
        )
        if not all_articles or 'articles' not in all_articles:
            return pd.DataFrame()
        df = pd.DataFrame(all_articles['articles'])
        # ensure publishedAt present
        if 'publishedAt' in df.columns:
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        return df
    except Exception as e:
        print("Error fetching news:", e)
        return pd.DataFrame()
