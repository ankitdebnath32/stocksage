import yfinance as yf
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
 
# Yahoo Finance Data
def get_stock_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df
 
# News API Data
def get_financial_news(api_key, query="stock market", days=7):
    newsapi = NewsApiClient(api_key=api_key)
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    all_articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        language='en',
        sort_by='relevancy',
        page_size=50
    )
    return pd.DataFrame(all_articles['articles'])