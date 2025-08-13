from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
 
MODEL_NAME = "ProsusAI/finbert"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
 
labels = ["negative", "neutral", "positive"]
 
def analyze_sentiment(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    output = model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=-1)
    sentiment = labels[scores.argmax()]
    confidence = scores.max().item()
    return sentiment, confidence
 
def analyze_news_sentiment(news_df):
    results = []
    for _, row in news_df.iterrows():
        sentiment, confidence = analyze_sentiment(row['title'] + " " + (row.get('description') or ""))
        results.append({
            "title": row['title'],
            "sentiment": sentiment,
            "confidence": confidence,
            "url": row['url']
        })
    return pd.DataFrame(results)