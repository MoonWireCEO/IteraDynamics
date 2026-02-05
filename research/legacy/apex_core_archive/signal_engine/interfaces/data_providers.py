"""
Abstract base classes for data providers.

These interfaces must be implemented by product-specific code to provide
data to the core signal engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime


class PriceProvider(ABC):
    """
    Abstract interface for price data providers.

    Products must implement this to provide price data from their
    specific data sources (CoinGecko for crypto, Yahoo Finance for securities, etc.)
    """

    @abstractmethod
    def get_price(self, asset: str) -> Optional[float]:
        """
        Get current price for a single asset.

        Args:
            asset: Asset ticker/symbol (e.g., "BTC", "SPY")

        Returns:
            Current price in USD, or None if unavailable
        """
        pass

    @abstractmethod
    def bulk_price_fetch(self, assets: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple assets efficiently.

        Args:
            assets: List of asset tickers/symbols

        Returns:
            Dictionary mapping asset -> price in USD
        """
        pass

    @abstractmethod
    def get_historical_prices(
        self,
        asset: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h"
    ) -> List[Dict]:
        """
        Get historical price data for an asset.

        Args:
            asset: Asset ticker/symbol
            start_time: Start of time range
            end_time: End of time range
            interval: Time interval (e.g., "1h", "1d")

        Returns:
            List of price points with timestamp and price
            Example: [{"timestamp": 1234567890, "price": 100.0}, ...]
        """
        pass


class SentimentProvider(ABC):
    """
    Abstract interface for sentiment data providers.

    Products must implement this to provide sentiment scores from their
    specific sources (CryptoPanic for crypto, news APIs for securities, etc.)
    """

    @abstractmethod
    def fetch_sentiment(self, asset: str) -> float:
        """
        Get sentiment score for an asset.

        Args:
            asset: Asset ticker/symbol

        Returns:
            Sentiment score in range [-1.0, 1.0]
            -1.0 = most negative, 0.0 = neutral, 1.0 = most positive
        """
        pass

    @abstractmethod
    def fetch_bulk_sentiment(self, assets: List[str]) -> Dict[str, float]:
        """
        Get sentiment scores for multiple assets.

        Args:
            assets: List of asset tickers/symbols

        Returns:
            Dictionary mapping asset -> sentiment score [-1.0, 1.0]
        """
        pass


class MarketDataProvider(ABC):
    """
    Abstract interface for market data and metadata.

    Products implement this to provide market-specific information
    (trading hours, market status, etc.)
    """

    @abstractmethod
    def get_market_hours(self) -> bool:
        """
        Check if the market is currently open for trading.

        Returns:
            True if market is open, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_assets(self) -> List[str]:
        """
        Get list of all supported asset tickers.

        Returns:
            List of asset tickers/symbols supported by this provider
        """
        pass

    @abstractmethod
    def get_asset_metadata(self, asset: str) -> Optional[Dict]:
        """
        Get metadata for an asset.

        Args:
            asset: Asset ticker/symbol

        Returns:
            Dictionary with asset metadata (name, category, etc.)
            or None if asset not found
        """
        pass
