from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeRecord:
    id: str
    symbol: str
    side: str
    qty: float
    price: float
    ts: datetime
    fee: float

class IBroker(ABC):
    """
    The Universal Interface.
    Both the Real Broker (Binance/IBKR) and the Paper Broker MUST follow these rules.
    """
    
    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Returns current quantity held (0.0 if flat)."""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Returns available cash/collateral."""
        pass

    @abstractmethod
    def submit_order(self, symbol: str, side: str, qty: float, current_price: Optional[float] = None) -> Optional[TradeRecord]:
        """
        Executes an order.
        current_price is optional for Real brokers (market order),
        but required for Paper brokers (to simulate fill).
        """
        pass