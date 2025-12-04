# src/twitter_sentiment_api.py

from fastapi import APIRouter, Query
from src.cache_instance import cache
from datetime import datetime
import json

router = APIRouter()

def get_cached_sentiment(asset: str):
    key = f"{asset}_twitter_sentiment"
    cached = cache.get_signal(key)
    if cached:
        return cached
    return {
        "asset": asset,
        "average_sentiment": None,
        "timestamp": None,
        "source": "twitter",
        "sample_tweets": []
    }

@router.get("/sentiment/twitter")
def get_twitter_sentiment(asset: str = Query(...)):
    """
    Return the latest sentiment score for a single asset.
    """
    return get_cached_sentiment(asset)

@router.get("/sentiment/twitter/batch")
def get_batch_twitter_sentiment(assets: str = Query(...)):
    """
    Return sentiment scores for multiple assets (comma-separated).
    Example: ?assets=BTC,ETH,SOL
    """
    asset_list = [a.strip().upper() for a in assets.split(",")]
    return [get_cached_sentiment(asset) for asset in asset_list]
