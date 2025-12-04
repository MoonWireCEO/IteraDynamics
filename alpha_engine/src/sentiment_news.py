from pathlib import Path
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import os

# Feature flag: Set to False to disable news sentiment (returns neutral)
NEWS_SENTIMENT_ENABLED = os.getenv("NEWS_SENTIMENT_ENABLED", "false").lower() == "true"

# News API (stubbed out - no free securities news API available)
# Future: Could integrate with NewsAPI, Bloomberg, or other services
API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = ""  # Placeholder for future news service integration

# Updated for securities tickers
ASSET_KEYWORDS = {
    "SPY": ["s&p 500", "spy", "s&p"],
    "QQQ": ["nasdaq", "qqq", "tech"],
    "XLF": ["financial", "xlf", "banks"],
    "XLK": ["technology", "xlk"],
    "ES": ["e-mini", "es", "futures"]
}

def fetch_securities_news():
    # Stubbed out - no free securities news API currently integrated
    # Returns empty list to allow sentiment analysis to return neutral scores
    try:
        if not BASE_URL or not API_KEY:
            return []
        # Future implementation would go here
        return []
    except Exception as e:
        print(f"Exception fetching securities news data: {e}")
        return []

def analyze_sentiment(posts):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {symbol: [] for symbol in ASSET_KEYWORDS.keys()}

    for title in posts:
        clean_title = title.lower().strip()
        score = analyzer.polarity_scores(clean_title)["compound"]

        for symbol, keywords in ASSET_KEYWORDS.items():
            if any(k in clean_title for k in keywords):
                sentiment_scores[symbol].append(score)

    return {
        symbol: round(sum(scores)/len(scores), 4) if scores else 0.0
        for symbol, scores in sentiment_scores.items()
    }

def fetch_news_sentiment_scores():
    # Return neutral sentiment if feature is disabled
    if not NEWS_SENTIMENT_ENABLED:
        print(f"[{datetime.utcnow()}] News sentiment disabled, returning neutral")
        return {symbol: 0.0 for symbol in ASSET_KEYWORDS.keys()}

    print(f"[{datetime.utcnow()}] Fetching news sentiment...")
    posts = fetch_securities_news()
    sentiment = analyze_sentiment(posts)
    print(f"News sentiment scores: {sentiment}")
    return sentiment


