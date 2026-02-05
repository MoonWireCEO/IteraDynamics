# src/auto_loop.py

import time
from src.ingest_discovery import ingest_market_data
from src.signal_generator import generate_signals
from src.dispatcher import dispatch_alerts
from src.cache_instance import cache

def auto_loop(interval=600):
    print("✅ MoonWire Auto-Loop Started...")

    while True:
        try:
            print("💬 Ingesting market data...")
            ingest_market_data(cache)

            print("🧠 Generating signals...")
            signals = generate_signals()

            print("📣 Dispatching alerts...")
            for signal in signals:
                dispatch_alerts(
                    asset=signal['asset'],
                    signal=signal,
                    cache=cache  # Added the missing argument here
                )

            print(f"✅ Cycle complete. Sleeping for {interval} seconds...\n")

        except Exception as e:
            print(f"❌ Error in auto-loop: {str(e)}")

        print(f"⏳ Sleeping for {interval} seconds...\n")
        time.sleep(interval)