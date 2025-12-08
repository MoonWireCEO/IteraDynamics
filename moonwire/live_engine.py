import time
import logging
import signal
import sys
import requests
import joblib 
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from datetime import datetime, timezone

# --- ARCHITECTURE IMPORTS ---
from apex_core.paper_broker import PaperBroker
from moonwire.strategies.ml_adapter import MLStrategyAdapter
from apex_core.train_model import train_brain

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MoonWire_Live")

# ---------------------------------------------------------------------
# DATA FEED
# ---------------------------------------------------------------------
class CoinbasePublicFeed:
    def fetch_latest_price(self, symbol: str) -> float:
        pair = f"{symbol.upper()}-USD"
        url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            price = float(data['data']['amount'])
            logger.info(f"🔌 Feed: {symbol} @ ${price:,.2f}")
            return price
        except Exception as e:
            logger.error(f"Data Feed Error: {e}")
            return 0.0

# ---------------------------------------------------------------------
# THE LIVE ENGINE
# ---------------------------------------------------------------------
class LiveEngine:
    def __init__(self, symbol: str, broker: PaperBroker, strategy: object, feed: object):
        self.symbol = symbol.upper()
        self.broker = broker
        self.feed = feed
        self.strategy = strategy
        self.running = True
        
        # CSV Recorder Setup
        self.csv_path = Path(__file__).parent / "flight_recorder.csv"
        self._init_csv()

        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _init_csv(self):
        """Creates the CSV header if file doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Price", "Signal", "RSI", "BB_Pos", "Vol_Z", "Confidence", "Balance"])

    def log_decision(self, price, packet, balance):
        """Writes a single row to the CSV flight recorder. Handles file locks."""
        try:
            # Try to open. If Excel has it locked, this will throw PermissionError.
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    price,
                    packet["signal"],
                    packet["rsi"],
                    packet["bb_pos"],
                    packet["vol_z"],
                    packet["confidence"],
                    balance
                ])
        except PermissionError:
            # Gentle warning, no crash
            logger.warning("⚠️ CSV Locked by user. Skipping log entry (Close Excel to resume logging).")
        except Exception as e:
            logger.error(f"⚠️ CSV Write Failed: {e}")

    def _handle_exit(self, signum, frame):
        logger.info("\n🛑 Shutdown signal received. Saving state...")
        self.broker.save_state()
        self.running = False

    def preload_history(self):
        logger.info("⚡ HOT START: Pre-loading last 24 hours of 1-MINUTE data...")
        try:
            import yfinance as yf
            history = yf.download(f"{self.symbol}-USD", period="1d", interval="1m", progress=False)
            if isinstance(history.columns, pd.MultiIndex):
                history.columns = history.columns.droplevel(1)
            history = history.rename(columns={"Close": "close"})
            recent = history.tail(50)
            count = 0
            for _, row in recent.iterrows():
                self.strategy.prices.append(row["close"])
                count += 1
            logger.info(f"✅ Loaded {count} minute-candles. Engine is WARM.")
        except Exception as e:
            logger.error(f"⚠️ Pre-load failed: {e}. Falling back to Cold Start.")

    def run_loop(self, interval_seconds: int = 60):
        logger.info(f"🚀 Starting Engine on {self.symbol}")
        logger.info(f"💰 Balance: ${self.broker.cash:,.2f}")
        logger.info(f"🧠 Strategy: {self.strategy.__class__.__name__}")
        
        self.preload_history()
        logger.info(f"Press Ctrl+C to stop.\n")

        while self.running:
            try:
                price = self.feed.fetch_latest_price(self.symbol)
                if price <= 0:
                    time.sleep(5)
                    continue

                # --- STRATEGY ---
                packet = self.strategy.analyze(price)
                
                # Unpack
                signal = packet["signal"]
                rsi = packet["rsi"]
                bb = packet["bb_pos"]
                vol = packet["vol_z"]
                conf = packet["confidence"]
                
                # --- INTELLIGENT LOGGING ---
                current_pos = self.broker.positions 
                
                # Determine what to show the human (UX)
                display_signal = signal
                if signal == "SELL" and current_pos == 0:
                    display_signal = "AVOID" # We agree it's bearish, but we can't sell.
                
                log_msg = f"🧠 READ: {display_signal} | RSI: {rsi} | BB: {bb} | VolZ: {vol} | Conf: {conf}%"
                
                if signal == "HOLD":
                    logger.info(log_msg)
                elif display_signal == "AVOID":
                    logger.info(f"🛡️ {log_msg}") # Shield emoji for 'Stay Away'
                else:
                    logger.info(f"🚨 {log_msg}")

                # --- CSV RECORDING ---
                # We save the RAW signal to CSV (SELL) for data integrity
                current_balance = self.broker.get_portfolio_value(price)
                self.log_decision(price, packet, current_balance)

                # --- EXECUTION ---
                if signal == "BUY" and current_pos == 0:
                    logger.info(f"🟢 EXECUTING: BUY {self.symbol} @ {price:.2f}")
                    self.broker.execute_trade("BUY", 0.1, price)
                    
                elif signal == "SELL" and current_pos > 0:
                    logger.info(f"🔴 EXECUTING: SELL {self.symbol} @ {price:.2f}")
                    self.broker.execute_trade("SELL", current_pos, price)

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"⚠️ Crash in loop: {e}")
                time.sleep(10)

        logger.info("✅ Engine stopped safely.")

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("\n=== 🔄 ITERA DYNAMICS AUTO-START SEQUENCE ===\n")
        state_path = Path(__file__).parent / "paper_state.json"
        model_path = Path("apex_core/models/btc_model_v1.pkl")

        print("--- CHECKPOINT 1: Training Fresh Brain (High Frequency) ---")
        train_brain() 
        
        print("\n--- CHECKPOINT 2: Initialize Broker ---")
        broker = PaperBroker(initial_cash=10000.0, state_file=str(state_path))
        if not state_path.exists(): broker.save_state()

        print("--- CHECKPOINT 3: Initialize Feed ---")
        feed = CoinbasePublicFeed()

        print("--- CHECKPOINT 4: Initialize Strategy ---")
        strategy = MLStrategyAdapter(symbol="BTC", model_path=str(model_path), lookback_window=48)

        print("--- CHECKPOINT 5: Launch Engine ---")
        engine = LiveEngine(symbol="BTC", broker=broker, strategy=strategy, feed=feed)
        
        print("--- CHECKPOINT 6: Starting Loop ---")
        engine.run_loop(interval_seconds=60)

    except Exception as e:
        print(f"\n🔥 CRITICAL CRASH: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")