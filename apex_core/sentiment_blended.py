from src.sentiment_reddit import fetch_sentiment_scores as fetch_reddit_scores
from src.sentiment_news import fetch_news_sentiment_scores as fetch_news_scores
from src.cache_instance import cache
from datetime import datetime

def blend_sentiment_scores():
    print(f"[{datetime.utcnow()}] Blending Reddit + News sentiment...")
    
    reddit_scores = fetch_reddit_scores()
    news_scores = fetch_news_scores()
    blended_scores = {}

    for asset in reddit_scores.keys():
        r = reddit_scores.get(asset, 0.0)
        n = news_scores.get(asset, 0.0)
        blended = round((r + n) / 2, 4)
        blended_scores[asset] = blended
        cache.set_signal(f"{asset}_sentiment", blended)
        print(f"[Blended Sentiment] {asset}: Reddit={r}, News={n}, Final={blended}")

    return blended_scores
