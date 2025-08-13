from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from typing import Tuple
import math

MODEL_NAME = "ProsusAI/finbert"  # FinBERT model on HF

# Load tokenizer & model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ["negative", "neutral", "positive"]

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Returns (label, confidence)
    """
    if text is None or (isinstance(text, str) and len(text.strip()) == 0):
        return "neutral", 1.0
    try:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            output = model(**tokens)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)
        confidence = float(scores.max().item())
        idx = int(scores.argmax())
        return labels[idx], confidence
    except Exception as e:
        # fallback neutral on any failure
        return "neutral", 0.0

def annotate_news_sentiment(news_df: pd.DataFrame, title_col: str = 'title', desc_col: str = 'description') -> pd.DataFrame:
    """
    Input: DataFrame from get_financial_news (must include publishedAt/title/description/url)
    Output: same DataFrame with added 'sentiment' and 'sentiment_confidence' columns
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=['publishedAt', 'title', 'description', 'sentiment', 'sentiment_confidence'])

    df = news_df.copy()
    texts = []
    for idx, row in df.iterrows():
        t = (row.get(title_col) or "") + " " + (row.get(desc_col) or "")
        texts.append(t.strip())

    sentiments = []
    confidences = []
    for text in texts:
        label, conf = analyze_sentiment(text)
        sentiments.append(label)
        confidences.append(conf)

    df['sentiment'] = sentiments
    df['sentiment_confidence'] = confidences
    # Normalize publishedAt to datetime
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    return df
