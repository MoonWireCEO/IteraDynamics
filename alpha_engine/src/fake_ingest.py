import random
import time

def ingest_market_data(cache):
    assets = ['SPY', 'QQQ', 'XLF', 'XLK', 'ES']

    # Realistic price ranges for each security
    price_ranges = {
        'SPY': (440, 460),   # S&P 500 ETF
        'QQQ': (370, 390),   # Nasdaq-100 ETF
        'XLF': (37, 40),     # Financial Sector ETF
        'XLK': (180, 190),   # Technology Sector ETF
        'ES': (4450, 4550)   # E-mini S&P 500 Futures
    }

    for asset in assets:
        price_range = price_ranges.get(asset, (100, 200))
        price_now = random.uniform(price_range[0], price_range[1])
        price_1hr_ago = price_now * random.uniform(0.997, 1.003)  # Smaller moves for securities
        volume_now = random.uniform(5000000, 20000000)  # Higher volume for ETFs
        volume_avg = volume_now * random.uniform(0.9, 1.1)
        sentiment_now = random.uniform(0.4, 0.8)
        sentiment_6hr_ago = sentiment_now * random.uniform(0.7, 1.3)

        cache.set_signal(asset, {
            'price_now': price_now,
            'price_1hr_ago': price_1hr_ago,
            'volume_now': volume_now,
            'volume_avg': volume_avg,
            'sentiment_now': sentiment_now,
            'sentiment_6hr_ago': sentiment_6hr_ago
        })
        print(f"[Ingested] {asset}: price {price_now:.2f}, volume {volume_now:.0f}, sentiment {sentiment_now:.2f}")
