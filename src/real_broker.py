# src/real_broker.py
# 🦅 ARGUS REAL BROKER - V8.0 (PERSISTENCE ENABLED)

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from pathlib import Path

load_dotenv()

HARDCODED_UUID = "5bce9ffb-611c-4dcb-9e18-75d3914825a1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "trade_state.json"

class RealBroker:
    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("❌ MISSING API KEYS in .env")

        self.api_secret = self.api_secret.replace('\\n', '\n')

        try:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            print(f"🔌 RealBroker: Connected. TARGETING UUID: {HARDCODED_UUID}")
        except Exception as e:
            print(f"❌ CONNECTION ERROR: {e}")
            raise e

    def save_trade_state(self, price):
        """Saves entry data so the Signal Generator can perform Audits later."""
        state = {
            "entry_timestamp": datetime.utcnow().isoformat(),
            "entry_price": float(price)
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        print(f"💾 Trade state saved: Entry at ${price}")

    def clear_trade_state(self):
        """Clears memory after a successful exit."""
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            print("🗑️ Trade state cleared.")

    def _get_value(self, obj):
        if obj is None: return 0.0
        if isinstance(obj, dict): return float(obj.get('value', 0))
        return float(getattr(obj, 'value', 0))

    def _get_accounts(self):
        try:
            return self.client.get_accounts(limit=250, portfolio_id=HARDCODED_UUID)
        except:
            return None

    @property
    def cash(self):
        try:
            response = self._get_accounts()
            for acc in getattr(response, 'accounts', []):
                if getattr(acc, 'currency', '') == 'USD':
                    return self._get_value(getattr(acc, 'available_balance', None))
        except: pass
        return 0.0

    @property
    def positions(self):
        try:
            response = self._get_accounts()
            for acc in getattr(response, 'accounts', []):
                if getattr(acc, 'currency', '') == 'BTC':
                    return self._get_value(getattr(acc, 'available_balance', None))
        except: pass
        return 0.0

    def execute_trade(self, action, qty, price=None):
        product_id = "BTC-USD"
        client_order_id = str(int(os.urandom(4).hex(), 16))

        try:
            if action.upper() == "BUY":
                usd_size = str(round(qty * price, 2))
                print(f"   🚀 SENDING BUY: ${usd_size}")
                resp = self.client.market_order_buy(client_order_id=client_order_id, product_id=product_id, quote_size=usd_size)
                
                if getattr(resp, 'success', False) or hasattr(resp, 'order_id'):
                    self.save_trade_state(price)
                    print(f"   ✅ BUY ORDER FILLED.")
                    return True

            elif action.upper() == "SELL":
                btc_size = f"{qty:.8f}"
                print(f"   🚀 SENDING SELL: {btc_size} BTC")
                resp = self.client.market_order_sell(client_order_id=client_order_id, product_id=product_id, base_size=btc_size)
                
                if getattr(resp, 'success', False) or hasattr(resp, 'order_id'):
                    self.clear_trade_state()
                    print(f"   ✅ SELL ORDER FILLED.")
                    return True

        except Exception as e:
            print(f"   ❌ ORDER ERROR: {e}")
        return False