# apex_core/signal_generator.py
# 🦅 ARGUS LIVE PILOT - V3.1 (GUARDRAILS + FULL LOGGING)

from __future__ import annotations
import sys, os, joblib, requests, json, time
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- 🔧 PATH CONFIGURATION ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

MODELS_DIR = project_root / "moonwire/models"
MODEL_FILE = "random_forest.pkl"
DATA_FILE = project_root / "flight_recorder.csv"
STATE_FILE = project_root / "trade_state.json"

# --- ⚠️ BROKER IMPORT ---
try:
    from src.real_broker import RealBroker
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

_broker = RealBroker()

def update_market_data():
    """ Fetches latest candle from Coinbase. """
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        resp = requests.get(url, params={'granularity': 3600}, timeout=10)
        data = resp.json()
        data.sort(key=lambda x: x[0]) 
        
        df = pd.read_csv(DATA_FILE) if DATA_FILE.exists() else pd.DataFrame(columns=["Timestamp","Open","High","Low","Close","Volume"])
        last_ts = pd.to_datetime(df['Timestamp']).max() if not df.empty else datetime.min
        
        new_rows = []
        for c in data:
            ts = datetime.fromtimestamp(c[0])
            if ts > last_ts:
                new_rows.append({
                    'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': c[3], 'High': c[2], 'Low': c[1], 'Close': c[4], 'Volume': c[5]
                })
        
        if new_rows:
            pd.DataFrame(new_rows).to_csv(DATA_FILE, mode='a', header=False, index=False)
            print(f"   >> ✅ Data Updated. Newest: {new_rows[-1]['Timestamp']}")
    except Exception as e:
        print(f"   >> ⚠️ Data Update Glitch: {e}")

def detect_regime(df):
    """ The 6-Regime Matrix logic. """
    sma_50 = ta.sma(df['Close'], length=50)
    sma_200 = ta.sma(df['Close'], length=200)
    vol = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
    vol_t = vol.rolling(100).mean().iloc[-1]
    
    cp, s50, s200 = df['Close'].iloc[-1], sma_50.iloc[-1], (sma_200.iloc[-1] if not sma_200.isnull().all() else sma_50.iloc[-1])
    
    if cp > s200:
        if cp > s50:
            return ("🐂 BULL QUIET", 0.90) if vol.iloc[-1] < vol_t else ("🐎 BULL VOLATILE", 0.50)
        return ("⚠️ PULLBACK (Warning)", 0.0)
    return ("🐯 RECOVERY", 0.25) if cp > s50 else ("🐻 BEAR QUIET", 0.0)

def generate_signals():
    print(f"[{datetime.now().time()}] 🦅 ARGUS EXECUTION CYCLE...")
    update_market_data()
    
    try:
        model = joblib.load(MODELS_DIR / MODEL_FILE)
        df = pd.read_csv(DATA_FILE)
        
        # Feature Engineering
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bband = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Pos'] = (df['Close'] - bband.iloc[:, 0]) / (bband.iloc[:, 2] - bband.iloc[:, 0])
        df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        
        feat = pd.DataFrame([[df['RSI'].iloc[-1], df['BB_Pos'].iloc[-1], df['Vol_Z'].iloc[-1]]], columns=['RSI', 'BB_Pos', 'Vol_Z'])
        price = df['Close'].iloc[-1]
        
        regime, risk_mult = detect_regime(df)
        raw_signal = "BUY" if model.predict(feat)[0] == 1 else "SELL"
        
        # 🦅 RESTORED LOGGING:
        print(f"   >> [BRAIN] Raw Signal: {raw_signal}")
        print(f"   >> [REGIME] {regime} | Risk Multiplier: {risk_mult:.2f}")

        # Save Dashboard Data
        with open(project_root / "cortex.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                "regime": regime, 
                "risk_mult": risk_mult, 
                "raw_signal": raw_signal
            }, f)

        cash, pos = _broker.cash, _broker.positions
        print(f"   >> [WALLET] Cash: ${cash:.2f} | BTC: {pos:.6f}")

        if raw_signal == "BUY" and risk_mult > 0.0:
            target = cash * risk_mult
            if target > 5.0:
                _broker.execute_trade("BUY", target / price, price)
        
        elif raw_signal == "SELL":
            if (pos * price) > 5.0:
                # 🛡️ APPLY GUARDRAILS
                if STATE_FILE.exists():
                    with open(STATE_FILE, "r") as f:
                        state = json.load(f)
                        entry_time = datetime.fromisoformat(state["entry_timestamp"])
                        entry_price = state["entry_price"]
                    
                    # 1. Hold Time Filter
                    hold_time = datetime.utcnow() - entry_time
                    if hold_time < timedelta(hours=4):
                        print(f"   >> [HOLD] SELL Rejected: Held for {hold_time.total_seconds()/3600:.1f}h. Min: 4h")
                        return

                    # 2. Fee Audit
                    profit_pct = (price - entry_price) / entry_price
                    if profit_pct < 0.002:
                        print(f"   >> [HOLD] SELL Rejected: Profit {profit_pct:.2%} < Fee Hurdle 0.20%")
                        return

                _broker.execute_trade("SELL", pos, price)
            else:
                print("   >> [HOLD] SELL Signal, but no BTC to sell.")

    except Exception as e:
        print(f"   >> ❌ ERROR: {e}")

if __name__ == "__main__":
    generate_signals()