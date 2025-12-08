import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class PaperBroker:
    def __init__(self, initial_cash=10000.0, state_file="paper_state.json"):
        self.cash = initial_cash
        self.positions = 0.0  # Amount of asset (e.g., BTC)
        self.trade_log = []   # History
        
        # Save file path
        self.state_path = Path(state_file)
        
        # Load existing state if available
        self.load_state()

    def get_portfolio_value(self, current_price):
        """Total Liquidation Value = Cash + (Crypto * Price)"""
        return self.cash + (self.positions * current_price)

    def execute_trade(self, action, amount, price, fee_pct=0.001):
        """
        Executes a trade and updates state immediately.
        action: "BUY" or "SELL"
        amount: Quantity of asset (e.g., 0.1 BTC)
        price: Current price
        fee_pct: 0.1% default
        """
        cost = amount * price
        fee = cost * fee_pct
        
        if action == "BUY":
            total_cost = cost + fee
            if total_cost > self.cash:
                logger.warning(f"⚠️ REJECTED: Insufficient Funds. Need ${total_cost:.2f}, Have ${self.cash:.2f}")
                return False
            
            self.cash -= total_cost
            self.positions += amount
            logger.info(f"PAPER FILL: BUY {amount} @ {price:.2f} (Fee: {fee:.2f})")
            
        elif action == "SELL":
            if self.positions < amount:
                logger.warning(f"⚠️ REJECTED: Sell size {amount} > Position {self.positions}")
                return False
                
            revenue = cost - fee
            self.cash += revenue
            self.positions -= amount
            logger.info(f"PAPER FILL: SELL {amount} @ {price:.2f} (Fee: {fee:.2f})")

        # Log the trade
        trade_record = {
            "ts": datetime.now(),  # Timestamp object (caused the crash before)
            "action": action,
            "qty": amount,
            "price": price,
            "fee": fee,
            "balance_after": self.cash
        }
        self.trade_log.append(trade_record)
        
        # Save immediately to disk
        self.save_state()
        return True

    def save_state(self):
        """
        Saves the current portfolio and trade history to a JSON file.
        Handles Timestamp serialization errors using default=str.
        """
        state = {
            "cash": self.cash,
            "positions": self.positions,
            "trade_log": self.trade_log
        }
        
        try:
            # FIX: default=str automatically converts datetime objects to strings
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=4, default=str) 
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")

    def load_state(self):
        """Loads wallet from disk so we don't reset to $10k on restart."""
        if not self.state_path.exists():
            return
            
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
                self.cash = data.get("cash", 10000.0)
                self.positions = data.get("positions", 0.0)
                self.trade_log = data.get("trade_log", [])
            logger.info(f"💰 Loaded State: Cash=${self.cash:.2f}, Pos={self.positions}")
        except Exception as e:
            logger.error(f"⚠️ Could not load state: {e}")