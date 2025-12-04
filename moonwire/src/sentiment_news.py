from pathlib import Path
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import os

# You should set this in your environment or Render secrets
API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
BASE_URL = "https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&public=true"

ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"]
}

def fetch_crypto_news():
    try:
        url = BASE_URL.format(api_key=API_KEY)
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"CryptoPanic fetch failed: {response.status_code}")
            return []
        data = response.json()
        return [post["title"] for post in data.get("results", [])]
    except Exception as e:
        print(f"Exception fetching CryptoPanic data: {e}")
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
    print(f"[{datetime.utcnow()}] Fetching CryptoPanic sentiment...")
    posts = fetch_crypto_news()
    sentiment = analyze_sentiment(posts)
    print(f"News sentiment scores: {sentiment}")
    return sentiment


