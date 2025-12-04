import os
import logging
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.cache_instance import cache
import ssl

# Feature flag: Set to False to disable Twitter sentiment (returns neutral)
TWITTER_SENTIMENT_ENABLED = os.getenv("TWITTER_SENTIMENT_ENABLED", "false").lower() == "true"

# Only import Twitter libraries if enabled
if TWITTER_SENTIMENT_ENABLED:
    try:
        import snscrape.modules.twitter as sntwitter
        import tweepy
        ssl._create_default_https_context = ssl._create_unverified_context
    except ImportError:
        logging.warning("Twitter libraries not available, sentiment will be disabled")
        TWITTER_SENTIMENT_ENABLED = False

logging.basicConfig(level=logging.INFO)

MOCK_TWEETS = [
    "SPY looks strong today, markets trending up.",
    "Technology sector showing positive momentum.",
    "Financial sector is performing well this week.",
    "Market sentiment appears neutral on major indices.",
    "Overall outlook seems positive for equities."
]

def fetch_from_snscrape(query: str, limit: int = 10):
    try:
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            tweets.append(tweet.content)
        return tweets
    except Exception as e:
        logging.error({
            "event": "twitter_fetch_error",
            "method": "snscrape",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
        return []

def fetch_from_twitter_api(query: str, limit: int = 10):
    try:
        bearer = os.getenv("TWITTER_BEARER_TOKEN")
        client = tweepy.Client(bearer_token=bearer)
        resp = client.search_recent_tweets(
            query=query,
            tweet_fields=["created_at", "lang"],
            max_results=max(limit, 10)
        )

        logging.info({
            "event": "twitter_api_response_debug",
            "query": query,
            "raw_response": str(resp.data),
            "timestamp": datetime.utcnow().isoformat()
        })

        return [t.text for t in resp.data or []]

    except tweepy.TooManyRequests:
        logging.warning({
            "event": "twitter_fetch_error",
            "method": "api",
            "error": "RateLimitExceeded",
            "timestamp": datetime.utcnow().isoformat()
        })
        return []
    except Exception as e:
        logging.error({
            "event": "twitter_fetch_error",
            "method": "api",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
        return []

def score_tweets(tweets: list[str]):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)["compound"] for t in tweets]
    return round(sum(scores) / len(scores), 4) if scores else 0.0

def fetch_tweets_and_analyze(asset: str, method="api", limit=10):
    # Return neutral sentiment if feature is disabled
    if not TWITTER_SENTIMENT_ENABLED:
        timestamp = datetime.utcnow().isoformat()
        logging.info({
            "event": "twitter_sentiment_disabled",
            "asset": asset,
            "timestamp": timestamp
        })

        cache.set_signal(f"{asset}_twitter_sentiment", {
            "sentiment": 0.0,
            "source": "Twitter (disabled)",
            "timestamp": timestamp
        })

        return {
            "asset": asset,
            "average_sentiment": 0.0,
            "timestamp": timestamp,
            "source": "twitter_disabled",
            "sample_tweets": []
        }

    query = f"${asset} OR #{asset}"

    tweets = []
    if method == "snscrape":
        tweets = fetch_from_snscrape(query, limit)
    elif method == "api":
        tweets = fetch_from_twitter_api(query, limit)

    if not tweets:
        logging.warning({
            "event": "twitter_fallback_used",
            "reason": "No tweets returned from either method",
            "timestamp": datetime.utcnow().isoformat()
        })
        tweets = MOCK_TWEETS

    avg_sentiment = score_tweets(tweets)
    timestamp = datetime.utcnow().isoformat()

    cache.set_signal(f"{asset}_twitter_sentiment", {
        "sentiment": avg_sentiment,
        "source": "Twitter",
        "timestamp": timestamp
    })

    return {
        "asset": asset,
        "average_sentiment": avg_sentiment,
        "timestamp": timestamp,
        "source": "twitter",
        "sample_tweets": tweets
    }