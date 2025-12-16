# apex_core/signal_generator.py
# 🦅 ARGUS LIVE PILOT - V1.2 (SYNCED WITH REAL BROKER)

from __future__ import annotations
import sys
import os
import joblib
import pandas as pd
import pandas_ta as ta
import requests
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv # Added to ensure env is loaded

# --- 🔧 CRITICAL PATH FIX ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent # apex_core -> IteraDynamics_Mono
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load Env (Redundant safety)
load_dotenv(project_root / ".env")

# --- CONFIGURATION ---
MODELS_DIR = project_root / "moonwire/models"
MODEL_FILE = "random_forest.pkl"
DATA_FILE = project_root / "flight_recorder.csv"

# --- ⚠️ LIVE BROKER IMPORT ⚠️ ---
try:
    from src.real_broker import RealBroker
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# Connect to API
try:
    print("   >> 🦅 CONNECTING TO LIVE COINBASE API...")
    _broker = RealBroker() 
except Exception as e:
    print(f"❌ CRITICAL: Broker Connection Failed: {e}")
    sys.exit(1)


def update_market_data(csv_path: Path = DATA_FILE):
    """ Fetches latest candle from Coinbase to keep the CSV alive. """
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {'granularity': 3600} # 1 Hour
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        data.sort(key=lambda x: x[0]) 
        
        # Simple CSV Append Logic
        if not csv_path.exists():
            print("   >> ⚠️ Flight Recorder missing. Creating new...")
            pd.DataFrame(columns=["Timestamp","Open","High","Low","Close","Volume"]).to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)
        if not df.empty:
            last_ts = pd.to_datetime(df['Timestamp']).max()
        else:
            last_ts = datetime.min
        
        new_rows = []
        for candle in data:
            ts = datetime.fromtimestamp(candle[0])
            if ts > last_ts:
                new_rows.append({
                    'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': candle[3], 'High': candle[2], 'Low': candle[1], 'Close': candle[4], 'Volume': candle[5]
                })
        
        if new_rows:
            pd.DataFrame(new_rows).to_csv(csv_path, mode='a', header=False, index=False)
            print(f"   >> ✅ Market Data Updated. Newest: {new_rows[-1]['Timestamp']}")
    except Exception as e:
        print(f"   >> ⚠️ Data Update Glitch: {e}")

def get_latest_features(csv_path: Path = DATA_FILE):
    try:
        df = pd.read_csv(csv_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.sort_values('Timestamp', inplace=True)
        
        # Feature Engineering (V1.0 Standard)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bband = ta.bbands(df['Close'], length=20, std=2)
        
        # Safe BB Column Access
        lower_col = next(c for c in bband.columns if c.startswith("BBL"))
        upper_col = next(c for c in bband.columns if c.startswith("BBU"))
        
        df['BB_Pos'] = (df['Close'] - bband[lower_col]) / (bband[upper_col] - bband[lower_col])
        df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        
        last = df.iloc[-1]
        # Return Dataframe for model, and Float for price
        return pd.DataFrame([[last['RSI'], last['BB_Pos'], last['Vol_Z']]], columns=['RSI', 'BB_Pos', 'Vol_Z']), last['Close']
    except:
        return None, None

def generate_signals():
    print(f"[{datetime.now().time()}] 🦅 ARGUS LIVE EXECUTION CYCLE...")
    
    # 1. Update & Load
    update_market_data()
    try:
        model = joblib.load(MODELS_DIR / MODEL_FILE)
    except FileNotFoundError:
        print(f"❌ Model not found at {MODELS_DIR / MODEL_FILE}")
        return

    features, price = get_latest_features()
    
    if features is None:
        print("   >> ❌ No Data. Skipping.")
        return

    # 2. Predict
    prediction = model.predict(features)[0]
    signal = "BUY" if prediction == 1 else "SELL"
    print(f"   >> [BRAIN] Signal: {signal} (Price: ${price:,.2f})")

    # 3. LIVE EXECUTION
    try:
        # --- FIXED METHOD CALLS ---
        cash = _broker.cash         # Using property instead of get_cash_balance()
        position = _broker.positions # Using property instead of get_position_size()
        
        print(f"   >> [WALLET] Cash: ${cash:.2f} | BTC: {position:.6f}")
        
        # --- CIRCUIT BREAKER ---
        # If account value drops below $85, KILL SWITCH.
        est_value = cash + (position * price)
        if est_value < 85.00:
            print("   >> 🛑 CRITICAL: Account Value < $85. Trading Halted.")
            return

        if signal == "BUY":
            investable_cash = cash - 2.0 
            if investable_cash > 10.0: 
                qty = investable_cash / price
                print(f"   >> 🦅 ROUTING ORDER: BUY {qty:.6f} BTC (~${investable_cash:.2f})")
                # Using execute_trade(action, qty, price)
                _broker.execute_trade("BUY", qty, price)
            else:
                print("   >> [HOLD] BUY Signal, but Insufficient Cash.")

        elif signal == "SELL":
            if (position * price) > 5.0:
                print(f"   >> 🦅 ROUTING ORDER: SELL {position:.6f} BTC")
                # Using execute_trade(action, qty, price)
                _broker.execute_trade("SELL", position, price)
            else:
                print("   >> [HOLD] SELL Signal, but no BTC to sell.")
                
    except Exception as e:
        print(f"   >> ❌ EXECUTION ERROR: {e}")

if __name__ == "__main__":
    generate_signals()