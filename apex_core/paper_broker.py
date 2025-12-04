import json
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from .interfaces import IBroker, TradeRecord

logger = logging.getLogger(__name__)

class PaperBroker(IBroker):
    """
    Simulates execution with fee/slippage modeling and JSON state persistence.
    """
    def __init__(self, state_file: Path, initial_capital: float = 10000.0, fee_rate: float = 0.001, slippage_bps: int = 5):
        self.state_file = state_file
        self.balance = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.positions: Dict[str, float] = {}
        self.trades: List[TradeRecord] = []
        
        # Load previous session if it exists
        self._load_state()

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol.upper(), 0.0)

    def get_balance(self) -> float:
        return self.balance

    def submit_order(self, symbol: str, side: str, qty: float, current_price: Optional[float] = None) -> Optional[TradeRecord]:
        if current_price is None:
            logger.error("PaperBroker requires 'current_price' to simulate execution.")
            return None

        symbol = symbol.upper()
        side = side.upper()

        # 1. Simulate Reality (Slippage)
        # Buying? You pay more. Selling? You get less.
        slippage_mult = 1 + (self.slippage_bps / 10000.0) if side == 'BUY' else 1 - (self.slippage_bps / 10000.0)
        fill_price = current_price * slippage_mult
        
        cost = fill_price * qty
        fee = cost * self.fee_rate

        # 2. Capital Governance
        if side == 'BUY':
            total_cost = cost + fee
            if total_cost > self.balance:
                logger.warning(f"REJECTED {symbol}: Cost {total_cost:.2f} > Balance {self.balance:.2f}")
                return None
            self.balance -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty

        elif side == 'SELL':
            if self.positions.get(symbol, 0.0) < qty:
                logger.warning(f"REJECTED {symbol}: Insufficient inventory.")
                return None
            revenue = cost - fee
            self.balance += revenue
            self.positions[symbol] -= qty

        # 3. Record Keeping
        trade = TradeRecord(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            qty=qty,
            price=fill_price,
            ts=datetime.now(timezone.utc),
            fee=fee
        )
        self.trades.append(trade)
        self._save_state()
        
        logger.info(f"PAPER FILL: {side} {qty} {symbol} @ {fill_price:.2f} (Fee: {fee:.2f})")
        return trade

    def _save_state(self):
        """Dump state to disk so we survive crashes."""
        data = {
            "balance": self.balance,
            "positions": self.positions,
            "trades": [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "price": t.price,
                    "ts": t.ts.isoformat(),
                    "fee": t.fee
                } for t in self.trades
            ]
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(data, indent=2))

    def _load_state(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.balance = data.get("balance", self.balance)
                self.positions = data.get("positions", {})
                # We could reload trades list here if needed for audit, 
                # but for execution logic, balance/positions is enough.
            except Exception as e:
                logger.error(f"Failed to load state: {e}")