import pandas as pd
import numpy as np
from typing import Tuple
from app.utils.indicators import add_technical_indicators

def map_sentiment_label_to_score(label: str):
    if isinstance(label, (int, float)):
        return float(label)
    l = str(label).lower()
    if l in ['positive', 'pos', '1', 'positive\n']:
        return 1.0
    if l in ['neutral', 'neu', '0']:
        return 0.0
    if l in ['negative', 'neg', '-1', 'negative\n']:
        return -1.0
    return 0.0

def aggregate_daily_sentiment(news_sentiment_df: pd.DataFrame, date_col='publishedAt', sentiment_col='sentiment'):
    if news_sentiment_df is None or news_sentiment_df.empty:
        return pd.DataFrame(columns=['sentiment_mean','sentiment_std','sentiment_count'])

    df = news_sentiment_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['date'] = df[date_col].dt.date
    df['sentiment_score'] = df[sentiment_col].apply(map_sentiment_label_to_score)
    agg = df.groupby('date').agg(
        sentiment_mean = ('sentiment_score', 'mean'),
        sentiment_std = ('sentiment_score', 'std'),
        sentiment_count = ('sentiment_score', 'count')
    ).reset_index()
    if agg.empty:
        return pd.DataFrame(columns=['sentiment_mean','sentiment_std','sentiment_count'])
    agg['date'] = pd.to_datetime(agg['date'])
    agg = agg.set_index('date').rename_axis('date')
    return agg

def build_feature_dataset(price_df: pd.DataFrame, sentiment_daily_df: pd.DataFrame, lookahead: int = 1) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    price_df = price_df.copy()
    if 'Date' in price_df.columns:
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df = price_df.set_index('Date')
    price_df = price_df.sort_index()
    price_with_ind = add_technical_indicators(price_df)

    # Align sentiment to price dates
    sentiment = sentiment_daily_df.copy()
    if sentiment is None or sentiment.empty:
        # create neutral sentiment index aligned to price dates
        idx = price_with_ind.index
        sentiment_reindex = pd.DataFrame(index=idx)
        sentiment_reindex['sentiment_mean'] = 0.0
        sentiment_reindex['sentiment_std'] = 0.0
        sentiment_reindex['sentiment_count'] = 0
        sentiment_reindex.index.name = price_with_ind.index.name or 'Date'
    else:
        # reindex sentiment to trading dates using forward fill
        sentiment_reindex = sentiment.reindex(price_with_ind.index.date, method='ffill')
        sentiment_reindex.index = pd.to_datetime(sentiment_reindex.index)
        sentiment_reindex.index.name = price_with_ind.index.name or 'Date'

    df = price_with_ind.join(sentiment_reindex, how='left')

    # Fill defaults
    df['sentiment_mean'] = df.get('sentiment_mean', 0.0).fillna(0.0)
    df['sentiment_std'] = df.get('sentiment_std', 0.0).fillna(0.0)
    df['sentiment_count'] = df.get('sentiment_count', 0).fillna(0)

    # Target: future close and return
    df['FutureClose'] = df['Close'].shift(-lookahead)
    df['FutureReturn'] = (df['FutureClose'] - df['Close']) / df['Close']
    df['Target'] = (df['FutureReturn'] > 0).astype(int)

    df = df.dropna(subset=['FutureClose'])

    feature_cols = [
        'Open','High','Low','Close','Volume',
        'SMA_5','SMA_10','SMA_20',
        'EMA_12','EMA_26',
        'RSI_14','MACD','MACD_signal','MACD_hist',
        'ATR_14','Return','LogReturn',
        'sentiment_mean','sentiment_std','sentiment_count'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y_class = df['Target'].copy()
    y_reg = df['FutureReturn'].copy()

    X['Close_minus_SMA5'] = X['Close'] - X.get('SMA_5', X['Close'])
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return X, y_class, y_reg, df
