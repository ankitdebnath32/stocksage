import pandas as pd
import numpy as np

def SMA(series: pd.Series, window: int):
    return series.rolling(window=window).mean()

def EMA(series: pd.Series, window: int):
    return series.ewm(span=window, adjust=False).mean()

def RSI(series: pd.Series, window: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series: pd.Series, span_short: int = 12, span_long: int = 26, span_signal: int = 9):
    ema_short = EMA(series, span_short)
    ema_long = EMA(series, span_long)
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def ATR(df: pd.DataFrame, window: int = 14):
    # df must contain High, Low, Close columns
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    df = df.sort_index()

    df['SMA_5'] = SMA(df['Close'], 5)
    df['SMA_10'] = SMA(df['Close'], 10)
    df['SMA_20'] = SMA(df['Close'], 20)

    df['EMA_12'] = EMA(df['Close'], 12)
    df['EMA_26'] = EMA(df['Close'], 26)

    df['RSI_14'] = RSI(df['Close'], 14)

    macd_line, signal_line, macd_hist = MACD(df['Close'])
    df['MACD'] = macd_line
    df['MACD_signal'] = signal_line
    df['MACD_hist'] = macd_hist

    df['ATR_14'] = ATR(df, 14)

    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log1p(df['Return'])

    df = df.fillna(method='ffill').fillna(method='bfill').dropna()
    return df
