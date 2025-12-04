# src/adapters/price_adapter.py
"""
AlphaEngine Price Provider Adapter.

Adapts Yahoo Finance data fetching to signal-engine-core's PriceProvider interface.
This allows AlphaEngine to use signal-engine-core modules while maintaining
its Yahoo Finance data source.
"""

from signal_engine.interfaces import PriceProvider
from typing import Dict, List, Optional
import pandas as pd


class YahooFinanceAdapter(PriceProvider):
    """
    Adapter that implements PriceProvider using Yahoo Finance.

    This bridges AlphaEngine's yfinance dependency with signal-engine-core's
    generic interface, allowing core modules to work without knowing about
    Yahoo Finance specifically.
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize Yahoo Finance adapter.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_price(self, asset: str) -> Optional[float]:
        """
        Get latest price for an asset using Yahoo Finance.

        Args:
            asset: Asset ticker (e.g., "SPY", "QQQ")

        Returns:
            Latest closing price or None if unavailable
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(asset)
            data = ticker.history(period="1d")
            if data.empty:
                return None
            return float(data['Close'].iloc[-1])
        except Exception as e:
            print(f"Warning: Failed to fetch price for {asset}: {e}")
            return None

    def get_price_history(
        self,
        asset: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical prices for an asset using Yahoo Finance.

        Args:
            asset: Asset ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with columns ['ts', 'close'] or None if unavailable
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(asset)
            data = ticker.history(start=start_date, end=end_date)
            if data.empty:
                return None

            # Standardize column names to match signal-engine-core expectations
            data = data.reset_index()
            data.columns = [c.lower() for c in data.columns]
            data = data.rename(columns={'date': 'ts'})

            # Return minimal required columns
            return data[['ts', 'close']].copy()
        except Exception as e:
            print(f"Warning: Failed to fetch history for {asset}: {e}")
            return None

    def bulk_price_fetch(self, assets: List[str]) -> Dict[str, Optional[float]]:
        """
        Fetch current prices for multiple assets.

        Args:
            assets: List of asset tickers

        Returns:
            Dictionary mapping tickers to prices
        """
        results = {}
        for asset in assets:
            results[asset] = self.get_price(asset)
        return results


__all__ = ['YahooFinanceAdapter']
