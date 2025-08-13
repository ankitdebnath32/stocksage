import os
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error

from app.utils.preprocess import build_feature_dataset, aggregate_daily_sentiment
from app.utils.data_fetcher import get_stock_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # points to backend/
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_xgboost_classifier(ticker: str, news_df: pd.DataFrame, period: str = "12mo", test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    # 1) Fetch price data
    price_df = get_stock_data(ticker, period=period)
    if price_df is None or price_df.empty:
        raise ValueError("No price data fetched for ticker: " + ticker)

    # 2) Aggregate daily sentiment
    sentiment_daily = aggregate_daily_sentiment(news_df, date_col='publishedAt', sentiment_col='sentiment')

    # 3) Build dataset
    X, y_class, y_reg, merged_df = build_feature_dataset(price_df, sentiment_daily, lookahead=1)

    if X.shape[0] < 50:
        raise ValueError("Not enough data to train. Need more historical data.")

    # 4) Train/test split (time-aware)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]

    # 5) Train classifier
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob))
    }

    # 7) Save model and feature names
    model_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_clf.joblib")
    joblib.dump({'model': clf, 'features': list(X.columns)}, model_path)

    return {
        'ticker': ticker,
        'model_path': model_path,
        'metrics': metrics,
        'train_rows': len(X_train),
        'test_rows': len(X_test)
    }

def train_xgboost_regressor(ticker: str, news_df: pd.DataFrame, period: str = "12mo", test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    price_df = get_stock_data(ticker, period=period)
    if price_df is None or price_df.empty:
        raise ValueError("No price data fetched for ticker: " + ticker)

    sentiment_daily = aggregate_daily_sentiment(news_df, date_col='publishedAt', sentiment_col='sentiment')
    X, y_class, y_reg, merged_df = build_feature_dataset(price_df, sentiment_daily, lookahead=1)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

    reg = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=random_state
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    model_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_reg.joblib")
    joblib.dump({'model': reg, 'features': list(X.columns)}, model_path)

    return {
        'ticker': ticker,
        'model_path': model_path,
        'rmse': rmse,
        'train_rows': len(X_train),
        'test_rows': len(X_test)
    }

def load_model_for_ticker(ticker: str, mode: str = 'clf') -> Tuple[Any, list]:
    fname = f"{ticker}_xgb_clf.joblib" if mode == 'clf' else f"{ticker}_xgb_reg.joblib"
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found for {ticker} at {path}")
    payload = joblib.load(path)
    return payload['model'], payload['features']

def predict_next_day_direction(ticker: str, latest_price_df: pd.DataFrame, latest_sentiment_daily: pd.DataFrame):
    model, features = load_model_for_ticker(ticker, mode='clf')
    # Build features for the input window
    X, _, _, merged = build_feature_dataset(latest_price_df, latest_sentiment_daily, lookahead=1)
    x_latest = X.iloc[[-1]][features]
    prob = float(model.predict_proba(x_latest)[0,1])
    pred = int(model.predict(x_latest)[0])
    return {'prob_up': prob, 'pred_up': bool(pred)}
