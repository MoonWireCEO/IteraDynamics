# src/history.py

from fastapi import APIRouter
from src.cache_instance import cache

router = APIRouter()

@router.get("/history")
def get_history():
    output = {}
    for key in cache._store:
        if key.endswith("_history"):
            asset = key.replace("_history", "")
            output[asset] = cache.get_signal(key)
    return output