import random
import time

def ingest_market_data(cache):
    assets = ['BTC', 'ETH', 'SOL']
    for asset in assets:
        price_now = random.uniform(25000, 30000) if asset == 'BTC' else random.uniform(1500, 2000) if asset == 'ETH' else random.uniform(20, 30)
        price_1hr_ago = price_now * random.uniform(0.97, 1.03)
        volume_now = random.uniform(1000000, 5000000)
        volume_avg = volume_now * random.uniform(0.8, 1.2)
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
