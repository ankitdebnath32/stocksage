import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
 
def add_technical_indicators(df):
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    # SMA
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    # EMA
    df['ema_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()
    df['ema_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    # RSI
    df['rsi_14'] = RSIIndicator(df['Close'], window=14).rsi()
    # MACD diff
    macd = MACD(df['Close'])
    df['macd_diff'] = macd.macd_diff()
    # ATR
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['atr_14'] = atr.average_true_range()
    # Rolling volatility
    df['vol_7'] = df['return'].rolling(window=7).std()
    df['vol_14'] = df['return'].rolling(window=14).std()
    # Lag features
    for lag in [1,2,3]:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
    return df
 
def add_sentiment_features(df, sentiment_daily_df):
    """
    sentiment_daily_df: DataFrame with columns ['date','sentiment_score','num_articles']
    that has daily aggregated sentiment for the ticker.
    """
    df = df.copy()
    df = df.merge(sentiment_daily_df, left_on=df.index.date, right_on='date', how='left')
    # Normalize or fill na
    df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
    df['num_articles'] = df['num_articles'].fillna(0)
    # rolling sentiment
    df['sentiment_roll3'] = df['sentiment_score'].rolling(window=3).mean().fillna(0)
    df['sentiment_roll7'] = df['sentiment_score'].rolling(window=7).mean().fillna(0)
    df.set_index('key_0', inplace=True)  # restore original index merge behavior if needed
    return df
 
def prepare_dataset(price_df, sentiment_daily_df, threshold=0.005):
    df = price_df.copy()
    df = add_technical_indicators(df)
    # Assume price_df index is datetime and sentiment_daily_df.date is datetime.date
    sentiment_daily_df['date'] = pd.to_datetime(sentiment_daily_df['date']).dt.date
    df = df.reset_index().rename(columns={'index':'datetime'})
    df['date_only'] = df['datetime'].dt.date
    df = df.merge(sentiment_daily_df, left_on='date_only', right_on='date', how='left')
    df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
    df['num_articles'] = df['num_articles'].fillna(0)
    # Target
    df['next_close'] = df['Close'].shift(-1)
    df['next_return'] = (df['next_close'] - df['Close']) / df['Close']
    df['target'] = (df['next_return'] >= threshold).astype(int)
    # Drop final row with NaN target
    df = df.dropna(subset=['target'])
    # Drop columns not needed
    df = df.set_index('datetime')
    return df
 
# Training function
def train_xgb(df, feature_cols, model_path='models/xgb_stocksage.pkl'):
    # Sort by time
    df = df.sort_index()
    X = df[feature_cols]
    y = df['target']
    # TimeSeries split
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    models = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        roc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
        scores.append({'acc': acc, 'roc': roc})
        models.append(model)
    # Choose last model or retrain on full data
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    final_model.fit(X, y, verbose=False)
    joblib.dump(final_model, model_path)
    return final_model, scores
 
# Example usage:
# price_df = pd.read_csv('data/processed/AAPL.csv', parse_dates=['Date'], index_col='Date')
# sentiment_daily_df = pd.read_csv('data/processed/AAPL_sentiment_daily.csv')
# df = prepare_dataset(price_df, sentiment_daily_df)
# feature_cols = ['sma_5','sma_10','ema_10','ema_20','rsi_14','macd_diff','atr_14','vol_7','sentiment_score','sentiment_roll3','return_lag_1','return_lag_2']
# model, scores = train_xgb(df, feature_cols)
# print(scores)