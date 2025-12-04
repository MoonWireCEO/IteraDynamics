# src/dashboard.py

from fastapi import APIRouter
from src.cache_instance import cache
from src.sentiment_blended import blend_sentiment_scores

router = APIRouter()

@router.get("/dashboard")
def dashboard():
    sentiment_scores = blend_sentiment_scores()
    output = {}

    for key in cache._store:
        # Skip internal storage keys
        if key.endswith("_signals") or key.endswith("_sentiment") or key.endswith("_history"):
            continue

        asset = key
        output[asset] = {
            "sentiment": sentiment_scores.get(asset, 0),
            "signals": cache.get_signal(f"{asset}_signals"),
            "history": cache.get_signal(f"{asset}_history") or []
        }

    return output
    
@router.get("/history")
def get_history():
    output = {}
    for key in cache._store:
        if key.endswith("_history"):
            asset = key.replace("_history", "")
            output[asset] = cache.get_signal(key)
    return output
