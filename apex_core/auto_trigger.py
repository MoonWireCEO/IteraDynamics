import time
from src.fake_ingest import ingest_market_data
from src.signal_generator import generate_signals
from src.dispatcher import dispatch_alerts

def auto_run(cache):
    while True:
        ingest_market_data(cache)
        generate_signals(cache)
        dispatch_alerts(cache)
        time.sleep(60)  # wait 60 seconds before next cycle
