import requests
import time

# Simple in-memory cache to avoid excessive API hits
_price_cache = {}
CACHE_TTL = 300  # seconds (5 minutes)

# Supported securities tickers (Yahoo Finance symbols)
SUPPORTED_TICKERS = {
    "SPY": "SPY",      # S&P 500 ETF
    "QQQ": "QQQ",      # Nasdaq-100 ETF
    "XLF": "XLF",      # Financial Sector ETF
    "XLK": "XLK",      # Technology Sector ETF
    "ES": "ES=F"       # E-mini S&P 500 Futures
}

HEADERS = {
    "User-Agent": "alphaengine-signal-platform/1.0"
}

def get_price_usd(asset: str):
    asset = asset.upper()
    yahoo_symbol = SUPPORTED_TICKERS.get(asset)
    if not yahoo_symbol:
        print(f"[Price Fetcher] Unsupported asset: {asset}")
        return None

    now = time.time()
    if asset in _price_cache:
        cached_price, timestamp = _price_cache[asset]
        if now - timestamp < CACHE_TTL:
            return cached_price

    try:
        # Yahoo Finance API v8 endpoint
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Extract current price from Yahoo Finance response
        result = data.get("chart", {}).get("result", [])
        if result and len(result) > 0:
            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice")
            if price:
                _price_cache[asset] = (price, now)
                return price

        print(f"[Price Fetcher] No price data available for {asset}")
        return None

    except Exception as e:
        print(f"[Price Fetcher] Error fetching price for {asset}: {e}")
        return None

def bulk_price_fetch(assets):
    now = time.time()
    assets_to_fetch = []
    result = {}

    for asset in assets:
        asset = asset.upper()
        if asset in _price_cache:
            cached_price, timestamp = _price_cache[asset]
            if now - timestamp < CACHE_TTL:
                result[asset] = cached_price
                continue

        if asset in SUPPORTED_TICKERS:
            assets_to_fetch.append(asset)

    if not assets_to_fetch:
        return result

    # Fetch each asset individually (Yahoo Finance API doesn't have a great bulk endpoint)
    for asset in assets_to_fetch:
        price = get_price_usd(asset)
        if price:
            result[asset] = price

    # Fallback mock data for securities if no data available
    if not result:
        print("[Price Fetcher] Using fallback mock price data")
        result = {
            "SPY": 450.00,
            "QQQ": 380.00,
            "XLF": 38.50,
            "XLK": 185.00,
            "ES": 4500.00
        }

    return result