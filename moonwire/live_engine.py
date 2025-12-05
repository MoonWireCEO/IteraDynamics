import time
import logging
import signal
import sys
import requests
import joblib # Needed to save the dummy model if real one is missing
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# --- ARCHITECTURE IMPORTS ---
from apex_core.paper_broker import PaperBroker
from moonwire.strategies.ml_adapter import MLStrategyAdapter

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MoonWire_Live")

# ---------------------------------------------------------------------
# DATA FEED (COINBASE)
# ---------------------------------------------------------------------
class CoinbasePublicFeed:
    """Fetches REAL live prices from Coinbase Public API (US-Friendly)."""
    def fetch_latest_price(self, symbol: str) -> float:
        # Coinbase format: BTC-USD
        pair = f"{symbol.upper()}-USD"
        url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
        
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            # Structure: {'data': {'base': 'BTC', 'currency': 'USD', 'amount': '95000.00'}}
            price = float(data['data']['amount'])
            
            logger.info(f"🔌 Feed: {symbol} @ ${price:,.2f}")
            return price
        except Exception as e:
            logger.error(f"Data Feed Error: {e}")
            # Fallback to prevent crash loop, using a safe 'hold' number logic in strategy
            return 0.0

# ---------------------------------------------------------------------
# THE LIVE ENGINE (HEARTBEAT)
# ---------------------------------------------------------------------
class LiveEngine:
    def __init__(self, symbol: str, broker: PaperBroker, strategy: object, feed: object):
        self.symbol = symbol.upper()
        self.broker = broker
        self.feed = feed
        self.strategy = strategy
        self.running = True

        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum, frame):
        logger.info("\n🛑 Shutdown signal received. Saving state...")
        self.running = False

    def run_loop(self, interval_seconds: int = 60):
        logger.info(f"🚀 Starting Engine on {self.symbol}")
        logger.info(f"💰 Balance: ${self.broker.get_balance():,.2f}")
        logger.info(f"🧠 Strategy: {self.strategy.__class__.__name__}")
        logger.info(f"Press Ctrl+C to stop.\n")

        while self.running:
            try:
                # 1. FETCH DATA
                price = self.feed.fetch_latest_price(self.symbol)
                
                if price <= 0:
                    time.sleep(5)
                    continue

                # 2. STRATEGY INFERENCE
                # The adapter handles memory, feature engineering, and governance inside here
                signal = self.strategy.analyze(price)
                
                # 3. EXECUTION LOGIC
                current_pos = self.broker.get_position(self.symbol)
                
                if signal == "BUY" and current_pos == 0:
                    logger.info(f"🟢 SIGNAL: BUY {self.symbol} @ {price:.2f}")
                    # Example Size: Buy 0.1 BTC (You can make this dynamic later)
                    self.broker.submit_order(self.symbol, "BUY", 0.1, price)
                    
                elif signal == "SELL" and current_pos > 0:
                    logger.info(f"🔴 SIGNAL: SELL {self.symbol} @ {price:.2f}")
                    # Size: Sell entire position
                    self.broker.submit_order(self.symbol, "SELL", current_pos, price)
                
                else:
                    # Optional: Log 'HOLD' if you want verbose output, otherwise keep silent
                    # logger.info(f"Feature Check: HOLD (Position: {current_pos})")
                    pass

                # 4. SLEEP
                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"⚠️ Crash in loop: {e}")
                time.sleep(10) # Safety backoff

        logger.info("✅ Engine stopped safely.")

# ---------------------------------------------------------------------
# HELPER: GENERATE DUMMY MODEL (If real one missing)
# ---------------------------------------------------------------------
def ensure_model_exists(path: Path):
    """Creates a dummy sklearn model if the real one isn't found."""
    if not path.exists():
        logger.warning(f"⚠️ Model file not found at {path}")
        logger.warning("🛠️ Generating DUMMY model for testing purposes...")
        
        from sklearn.linear_model import LogisticRegression
        # Create a dummy model trained on random data just so .predict_proba works
        X_dummy = np.random.rand(10, 12) # 10 rows, 12 features (matches feature_builder)
        y_dummy = np.random.randint(0, 2, 10)
        
        model = LogisticRegression()
        model.fit(X_dummy, y_dummy)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"✅ Dummy model saved to {path}")

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("--- CHECKPOINT 1: Setup Paths ---")
        # 1. Setup Paths
        state_path = Path(__file__).parent / "paper_state.json"
        
        model_path = Path("apex_core/models/btc_model_v1.pkl")
        ensure_model_exists(model_path)

        print("--- CHECKPOINT 2: Initialize Broker ---")
        # 2. Initialize Components
        broker = PaperBroker(
            state_file=state_path,
            initial_capital=10000.0,
            slippage_bps=5
        )
        
        # Force save state to ensure file exists (if new)
        if not state_path.exists():
            broker._save_state()

        print("--- CHECKPOINT 3: Initialize Feed ---")
        # Initialize Feed
        feed = CoinbasePublicFeed()

        print("--- CHECKPOINT 4: Initialize Strategy ---")
        # Initialize Strategy Adapter (Connecting to Apex Core Brain)
        strategy = MLStrategyAdapter(
            symbol="BTC", 
            model_path=str(model_path),
            lookback_window=48 
        )

        print("--- CHECKPOINT 5: Launch Engine ---")
        # 3. Launch Engine
        engine = LiveEngine(
            symbol="BTC", 
            broker=broker, 
            strategy=strategy, 
            feed=feed
        )
        
        print("--- CHECKPOINT 6: Starting Loop ---")
        # Run loop (check every 60 seconds)
        engine.run_loop(interval_seconds=60)

    except Exception as e:
        print(f"\n🔥 CRITICAL CRASH: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...") # Keeps window open so you can read the error