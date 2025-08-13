import pandas as pd
import numpy as np
from typing import Tuple
from .indicators import add_technical_indicators
 
def map_sentiment_label_to_score(label: str):
    # Accepts 'positive','neutral','negative' or 1/0/-1
    if isinstance(label, (int, float)):
        return float(label)
    l = str(label).lower()
    if l in ['positive', 'pos', '1']:
        return 1.0
    if l in ['neutral', 'neu', '0']:
        return 0.0
    if l in ['negative', 'neg', '-1']:
        return -1.0
    return 0.0
 
def aggregate_daily_sentiment(news_sentiment_df: pd.DataFrame, date_col='publishedAt', sentiment_col='sentiment'):
    """
    news_sentiment_df expected columns: publishedAt (ISO str or datetime), sentiment (label or numeric), optional confidence
    Returns df with index = date (yyyy-mm-dd) and column sentiment_score (mean), sentiment_count
    """
    df = news_sentiment_df.copy()
    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['date'] = df[date_col].dt.date
    df['sentiment_score'] = df[sentiment_col].apply(map_sentiment_label_to_score)
    agg = df.groupby('date').agg(
        sentiment_mean = ('sentiment_score', 'mean'),
        sentiment_std = ('sentiment_score', 'std'),
        sentiment_count = ('sentiment_score', 'count')
    ).reset_index()
    agg['date'] = pd.to_datetime(agg['date'])
    agg = agg.set_index('date').rename_axis('date')
    return agg
 
def build_feature_dataset(price_df: pd.DataFrame, sentiment_daily_df: pd.DataFrame, lookahead: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    price_df: DataFrame from yfinance with Date index (datetime) and OHLCV columns
    sentiment_daily_df: indexed by date (datetime) with columns sentiment_mean, sentiment_count...
    lookahead: days ahead to predict (1 -> next-day)
    Returns: X (features), y_class (binary up/down), y_reg (actual next-day return)
    """
    # Add indicators
    price_df = price_df.copy()
    if 'Date' in price_df.columns:
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df.set_index('Date', inplace=True)
    price_df = price_df.sort_index()
    price_with_ind = add_technical_indicators(price_df)
 
    # Align sentiment to price dates
    sentiment = sentiment_daily_df.copy()
    # Reindex sentiment to business days and forward fill (so each trading day has the latest available sentiment)
    sentiment_reindex = sentiment.reindex(price_with_ind.index.date, method='ffill')
    sentiment_reindex.index = pd.to_datetime(sentiment_reindex.index)
    sentiment_reindex.index.name = price_with_ind.index.name or 'Date'
    # Join
    df = price_with_ind.join(sentiment_reindex, how='left')
 
    # If sentiment NaN, fill with 0 (neutral) and count 0
    if 'sentiment_mean' in df.columns:
        df['sentiment_mean'] = df['sentiment_mean'].fillna(0.0)
    else:
        df['sentiment_mean'] = 0.0
    if 'sentiment_count' in df.columns:
        df['sentiment_count'] = df['sentiment_count'].fillna(0)
    else:
        df['sentiment_count'] = 0
 
    # Target: next-day return
    df['FutureClose'] = df['Close'].shift(-lookahead)
    df['FutureReturn'] = (df['FutureClose'] - df['Close']) / df['Close']
    # Binary target: 1 if next-day return > 0 else 0
    df['Target'] = (df['FutureReturn'] > 0).astype(int)
 
    # Drop last lookahead rows with NaN future
    df = df.dropna(subset=['FutureClose'])
 
    # Select features
    feature_cols = [
        'Open','High','Low','Close','Volume',
        'SMA_5','SMA_10','SMA_20',
        'EMA_12','EMA_26',
        'RSI_14','MACD','MACD_signal','MACD_hist',
        'ATR_14','Return','LogReturn',
        'sentiment_mean','sentiment_std','sentiment_count'
    ]
    # Keep only columns present
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y_class = df['Target'].copy()
    y_reg = df['FutureReturn'].copy()
 
    # Optionally add lag features
    X['Close_minus_SMA5'] = X['Close'] - X.get('SMA_5', X['Close'])
    # Fill any remaining NaNs
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return X, y_class, y_reg, df