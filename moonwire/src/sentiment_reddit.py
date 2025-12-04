import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import re

REDDIT_URL = "https://www.reddit.com/r/cryptocurrency/hot.json?limit=50"
HEADERS = {"User-Agent": "MoonWireSentimentBot/0.1"}
ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"]
}

def fetch_reddit_posts():
    try:
        response = requests.get(REDDIT_URL, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Reddit fetch failed: {response.status_code}")
            return []
        data = response.json()
        posts = data.get("data", {}).get("children", [])
        return [p["data"]["title"] for p in posts]
    except Exception as e:
        print(f"Exception fetching Reddit posts: {e}")
        return []

def clean_text(text):
    return re.sub(r"[\n\r]", " ", text).strip().lower()

def analyze_sentiment(posts):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {symbol: [] for symbol in ASSET_KEYWORDS.keys()}

    for title in posts:
        clean_title = clean_text(title)
        score = analyzer.polarity_scores(clean_title)["compound"]

        for symbol, keywords in ASSET_KEYWORDS.items():
            if any(k in clean_title for k in keywords):
                sentiment_scores[symbol].append(score)

    return {
        symbol: round(sum(scores)/len(scores), 4) if scores else 0.0
        for symbol, scores in sentiment_scores.items()
    }

def fetch_sentiment_scores():
    print(f"[{datetime.utcnow()}] Fetching Reddit sentiment...")
    posts = fetch_reddit_posts()
    sentiment = analyze_sentiment(posts)
    print(f"Sentiment scores: {sentiment}")
    return sentiment
