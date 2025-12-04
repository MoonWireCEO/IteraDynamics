# Integration Adapter Examples

This guide shows how to integrate signal-engine-core into AlphaEngine and MoonWire using the adapter pattern.

---

## Table of Contents

1. [Overview](#overview)
2. [AlphaEngine Integration](#alphaengine-integration)
3. [MoonWire Integration](#moonwire-integration)
4. [Creating Custom Adapters](#creating-custom-adapters)
5. [Testing Adapters](#testing-adapters)

---

## Overview

The adapter pattern allows products to integrate signal-engine-core while maintaining their product-specific implementations:

```
┌─────────────────────────────────────────┐
│         signal-engine-core              │
│  (Product-agnostic ML/Validation)       │
│                                          │
│  ┌──────────┐  ┌──────────┐            │
│  │ Inference│  │ Backtest │            │
│  └──────────┘  └──────────┘            │
│  ┌──────────┐  ┌──────────┐            │
│  │ Metrics  │  │  Tuner   │            │
│  └──────────┘  └──────────┘            │
└─────────────────────────────────────────┘
            ▲              ▲
            │              │
   ┌────────┴────┐    ┌───┴────────┐
   │ AlphaEngine │    │  MoonWire  │
   │  Adapters   │    │  Adapters  │
   └─────────────┘    └────────────┘
```

---

## AlphaEngine Integration

### Directory Structure

```
AlphaEngine/
├── alphaengine/
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── price_adapter.py       # Yahoo Finance → core interface
│   │   ├── model_loader.py        # Load models from AlphaEngine structure
│   │   └── data_adapter.py        # Data format conversions
│   ├── ml/
│   │   ├── inference_service.py   # Uses signal_engine.ml.InferenceEngine
│   │   ├── backtest_service.py    # Uses signal_engine.ml.run_backtest
│   │   └── metrics_service.py     # Uses signal_engine.ml.metrics
│   └── requirements.txt           # Add: signal-engine-core>=1.0.0
```

### 1. Price Provider Adapter

**File:** `alphaengine/adapters/price_adapter.py`

```python
"""
AlphaEngine Price Provider Adapter.

Adapts Yahoo Finance data fetching to signal-engine-core's PriceProvider interface.
"""

from signal_engine.interfaces import PriceProvider
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf


class YahooFinanceAdapter(PriceProvider):
    """
    Adapter that implements PriceProvider using Yahoo Finance.
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize Yahoo Finance adapter.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_price(self, asset: str) -> Optional[float]:
        """
        Get latest price for an asset.

        Args:
            asset: Asset ticker (e.g., "SPY", "QQQ")

        Returns:
            Latest closing price or None if unavailable
        """
        try:
            ticker = yf.Ticker(asset)
            data = ticker.history(period="1d")
            if data.empty:
                return None
            return float(data['Close'].iloc[-1])
        except Exception:
            return None

    def get_price_history(
        self,
        asset: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical prices for an asset.

        Args:
            asset: Asset ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with OHLCV data or None if unavailable
        """
        try:
            ticker = yf.Ticker(asset)
            data = ticker.history(start=start_date, end=end_date)
            if data.empty:
                return None

            # Standardize column names
            data = data.reset_index()
            data.columns = [c.lower() for c in data.columns]
            data = data.rename(columns={'date': 'ts'})

            return data
        except Exception:
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
```

### 2. Model Loader Adapter

**File:** `alphaengine/adapters/model_loader.py`

```python
"""
AlphaEngine Model Loader Adapter.

Handles loading models from AlphaEngine's specific directory structure.
"""

import joblib
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any


class AlphaEngineModelLoader:
    """
    Loads models from AlphaEngine's models/ directory structure.
    """

    def __init__(self, models_dir: Path = Path("models")):
        """
        Initialize model loader.

        Args:
            models_dir: Root models directory
        """
        self.models_dir = models_dir

    def load_latest_model(
        self,
        model_type: str = "current"
    ) -> Tuple[Any, List[str], Dict[str, Any]]:
        """
        Load the latest model with metadata.

        Args:
            model_type: Model type subdirectory (e.g., "current", "v1.2.3")

        Returns:
            Tuple of (model, feature_order, metadata)
        """
        model_dir = self.models_dir / model_type

        # Load model
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Extract feature order
        feature_order = metadata.get(
            "feature_order",
            metadata.get("features", [])
        )

        if not feature_order:
            # Try separate features file
            features_path = model_dir / "features.json"
            if features_path.exists():
                with open(features_path) as f:
                    features_data = json.load(f)
                    feature_order = features_data.get("feature_order", [])

        return model, feature_order, metadata

    def load_ensemble_models(
        self
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load all available models for ensemble.

        Returns:
            Dictionary mapping model names to their configs
        """
        models = {}

        for model_name in ["logistic", "rf", "gb"]:
            try:
                model_path = self.models_dir / f"trigger_likelihood_{model_name}.joblib"
                if not model_path.exists():
                    # Try alternative naming
                    model_path = self.models_dir / f"{model_name}.joblib"

                if model_path.exists():
                    model = joblib.load(model_path)

                    # Load metadata
                    meta_path = self.models_dir / f"trigger_likelihood_{model_name}.meta.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            metadata = json.load(f)
                        feature_order = metadata.get("feature_order", [])
                    else:
                        feature_order = []

                    models[model_name] = {
                        "model": model,
                        "feature_order": feature_order,
                        "metadata": metadata if 'metadata' in locals() else {},
                    }
            except Exception:
                continue

        return models
```

### 3. Inference Service

**File:** `alphaengine/ml/inference_service.py`

```python
"""
AlphaEngine Inference Service.

High-level service that uses signal-engine-core's inference capabilities
with AlphaEngine-specific adapters.
"""

from signal_engine.ml import InferenceEngine, EnsembleInferenceEngine
from alphaengine.adapters.model_loader import AlphaEngineModelLoader
from alphaengine.adapters.price_adapter import YahooFinanceAdapter
from typing import Dict, Any, Optional
from pathlib import Path


class AlphaEngineInferenceService:
    """
    AlphaEngine-specific inference service using core engine.
    """

    def __init__(self, models_dir: Path = Path("models")):
        """
        Initialize inference service.

        Args:
            models_dir: Directory containing models
        """
        self.model_loader = AlphaEngineModelLoader(models_dir)
        self.price_adapter = YahooFinanceAdapter()
        self.engine: Optional[InferenceEngine] = None
        self.ensemble: Optional[EnsembleInferenceEngine] = None

    def initialize_single_model(self, model_type: str = "current"):
        """
        Initialize single model inference.

        Args:
            model_type: Model type to load
        """
        model, feature_order, metadata = self.model_loader.load_latest_model(model_type)

        self.engine = InferenceEngine(
            model=model,
            feature_order=feature_order,
            metadata=metadata
        )

    def initialize_ensemble(self):
        """Initialize ensemble of models."""
        models_config = self.model_loader.load_ensemble_models()

        if not models_config:
            raise ValueError("No models found for ensemble")

        self.ensemble = EnsembleInferenceEngine(
            models=models_config,
            aggregation="mean"
        )

    def predict(
        self,
        features: Dict[str, Any],
        use_ensemble: bool = True,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction using appropriate engine.

        Args:
            features: Feature dictionary
            use_ensemble: Whether to use ensemble (if available)
            explain: Whether to include feature contributions

        Returns:
            Prediction result dictionary
        """
        if use_ensemble and self.ensemble is not None:
            result = self.ensemble.predict_proba(features)
        elif self.engine is not None:
            result = self.engine.predict_proba(features, explain=explain)
        else:
            raise RuntimeError("No inference engine initialized")

        return result

    def predict_for_asset(
        self,
        asset: str,
        features: Dict[str, Any],
        include_price: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction for a specific asset with additional context.

        Args:
            asset: Asset ticker
            features: Feature dictionary
            include_price: Whether to include current price

        Returns:
            Prediction with asset context
        """
        prediction = self.predict(features, explain=True)

        result = {
            "asset": asset,
            "prediction": prediction,
        }

        if include_price:
            current_price = self.price_adapter.get_price(asset)
            result["current_price"] = current_price

        return result
```

### 4. Usage Example

**File:** `alphaengine/examples/inference_example.py`

```python
"""
Example: Using signal-engine-core in AlphaEngine.
"""

from alphaengine.ml.inference_service import AlphaEngineInferenceService
from pathlib import Path


def main():
    # Initialize service
    service = AlphaEngineInferenceService(models_dir=Path("models"))

    # Option 1: Single model
    service.initialize_single_model()

    # Option 2: Ensemble (recommended)
    service.initialize_ensemble()

    # Make prediction
    features = {
        "burst_z": 2.5,
        "momentum_7d": 0.15,
        "volatility_14d": 0.08,
        "rsi_14": 65.0,
        # ... other features
    }

    # Predict for SPY
    result = service.predict_for_asset("SPY", features, include_price=True)

    print(f"Asset: {result['asset']}")
    print(f"Probability: {result['prediction']['probability']:.3f}")
    print(f"Current Price: ${result['current_price']:.2f}")

    if "contributions" in result['prediction']:
        print("\nTop Contributors:")
        for feature, contribution in list(result['prediction']['contributions'].items())[:5]:
            print(f"  {feature}: {contribution:.4f}")


if __name__ == "__main__":
    main()
```

---

## MoonWire Integration

### Directory Structure

```
moonwire-backend/
├── moonwire/
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── crypto_price_adapter.py  # CoinGecko → core interface
│   │   ├── model_loader.py          # Load models from MoonWire structure
│   │   └── data_adapter.py          # Crypto-specific data
│   ├── ml/
│   │   ├── inference_service.py     # Uses signal_engine.ml
│   │   ├── backtest_service.py      # Crypto backtesting
│   │   └── metrics_service.py       # Crypto metrics
│   └── requirements.txt             # Add: signal-engine-core>=1.0.0
```

### 1. Crypto Price Provider Adapter

**File:** `moonwire/adapters/crypto_price_adapter.py`

```python
"""
MoonWire Crypto Price Provider Adapter.

Adapts CoinGecko/crypto data sources to signal-engine-core's PriceProvider interface.
"""

from signal_engine.interfaces import PriceProvider
from typing import Dict, List, Optional
import pandas as pd
import requests
from datetime import datetime, timedelta


class CoinGeckoAdapter(PriceProvider):
    """
    Adapter that implements PriceProvider using CoinGecko API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko adapter.

        Args:
            api_key: Optional CoinGecko API key for higher rate limits
        """
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self._coin_id_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "AVAX": "avalanche-2",
            # Add more mappings as needed
        }

    def _get_coin_id(self, asset: str) -> str:
        """Convert ticker symbol to CoinGecko coin ID."""
        return self._coin_id_map.get(asset.upper(), asset.lower())

    def get_price(self, asset: str) -> Optional[float]:
        """
        Get latest price for a crypto asset.

        Args:
            asset: Asset ticker (e.g., "BTC", "ETH")

        Returns:
            Latest price in USD or None if unavailable
        """
        try:
            coin_id = self._get_coin_id(asset)
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return float(data.get(coin_id, {}).get("usd"))
        except Exception:
            return None

    def get_price_history(
        self,
        asset: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical prices for a crypto asset.

        Args:
            asset: Asset ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with price history or None if unavailable
        """
        try:
            coin_id = self._get_coin_id(asset)

            # Calculate days
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
            days = (end - start).days

            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "hourly" if days <= 90 else "daily"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            prices = data.get("prices", [])

            if not prices:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.drop("timestamp", axis=1)

            return df[["ts", "close"]]
        except Exception:
            return None

    def bulk_price_fetch(self, assets: List[str]) -> Dict[str, Optional[float]]:
        """
        Fetch current prices for multiple crypto assets.

        Args:
            assets: List of asset tickers

        Returns:
            Dictionary mapping tickers to prices
        """
        try:
            coin_ids = [self._get_coin_id(asset) for asset in assets]
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            results = {}
            for asset in assets:
                coin_id = self._get_coin_id(asset)
                price = data.get(coin_id, {}).get("usd")
                results[asset] = float(price) if price else None

            return results
        except Exception:
            return {asset: None for asset in assets}
```

### 2. Usage Example

**File:** `moonwire/examples/crypto_inference_example.py`

```python
"""
Example: Using signal-engine-core in MoonWire for crypto.
"""

from signal_engine.ml import InferenceEngine, detect_market_regime
from moonwire.adapters.crypto_price_adapter import CoinGeckoAdapter
from moonwire.adapters.model_loader import MoonWireModelLoader
from pathlib import Path
import pandas as pd


def main():
    # Initialize adapters
    price_adapter = CoinGeckoAdapter()
    model_loader = MoonWireModelLoader(models_dir=Path("models"))

    # Load model
    model, feature_order, metadata = model_loader.load_latest_model()

    # Create inference engine
    engine = InferenceEngine(
        model=model,
        feature_order=feature_order,
        metadata=metadata
    )

    # Get BTC price history for regime detection
    prices_df = price_adapter.get_price_history(
        "BTC",
        start_date="2025-01-01",
        end_date="2025-10-30"
    )

    if prices_df is not None:
        # Detect market regime
        regime = detect_market_regime(
            prices_df,
            symbol="BTC",
            volatility_threshold=0.08,  # Crypto has higher volatility
            trend_strength_threshold=0.15
        )

        latest_regime = regime.iloc[-1]
        print(f"Current BTC Regime: {latest_regime}")

    # Make prediction
    features = {
        "burst_z": 3.2,
        "social_sentiment": 0.72,
        "volume_spike": 2.1,
        "on_chain_activity": 1.5,
        # ... other crypto-specific features
    }

    result = engine.predict_proba(features, explain=True)

    print(f"\nBTC Signal Probability: {result['probability']:.3f}")

    if "contributions" in result:
        print("\nTop Contributors:")
        for feature, contribution in list(result['contributions'].items())[:5]:
            print(f"  {feature}: {contribution:.4f}")


if __name__ == "__main__":
    main()
```

---

## Creating Custom Adapters

### Template for New Products

```python
"""
Template for creating a custom price provider adapter.
"""

from signal_engine.interfaces import PriceProvider
from typing import Dict, List, Optional
import pandas as pd


class CustomPriceAdapter(PriceProvider):
    """
    Custom price provider adapter for your data source.
    """

    def __init__(self, config: dict):
        """
        Initialize with your configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        # Initialize your data source connection

    def get_price(self, asset: str) -> Optional[float]:
        """
        Get latest price for an asset.

        Implement this to fetch from your data source.
        """
        # Your implementation here
        pass

    def get_price_history(
        self,
        asset: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical prices.

        Implement this to fetch historical data.
        """
        # Your implementation here
        pass

    def bulk_price_fetch(self, assets: List[str]) -> Dict[str, Optional[float]]:
        """
        Batch fetch prices.

        Optional: Override for better performance.
        """
        return {asset: self.get_price(asset) for asset in assets}
```

---

## Testing Adapters

### Unit Test Template

```python
"""
Test template for adapters.
"""

import pytest
from your_product.adapters.price_adapter import YourPriceAdapter


def test_get_price():
    """Test single price fetch."""
    adapter = YourPriceAdapter()
    price = adapter.get_price("SPY")

    assert price is not None
    assert isinstance(price, float)
    assert price > 0


def test_get_price_history():
    """Test historical price fetch."""
    adapter = YourPriceAdapter()
    df = adapter.get_price_history("SPY", "2025-01-01", "2025-01-31")

    assert df is not None
    assert not df.empty
    assert "ts" in df.columns
    assert "close" in df.columns


def test_bulk_price_fetch():
    """Test batch price fetch."""
    adapter = YourPriceAdapter()
    assets = ["SPY", "QQQ", "DIA"]
    prices = adapter.bulk_price_fetch(assets)

    assert len(prices) == 3
    for asset in assets:
        assert asset in prices
        assert prices[asset] is None or isinstance(prices[asset], float)
```

---

## Summary

The adapter pattern provides:

✅ **Clean Separation** - Product code separate from core logic
✅ **Easy Testing** - Mock adapters for unit tests
✅ **Flexibility** - Switch data sources without changing core code
✅ **Reusability** - Same core logic across all products
✅ **Maintainability** - Updates to core benefit all products

**Next Steps:**
1. Implement adapters for your product
2. Write adapter tests
3. Integrate core modules
4. Measure performance improvements
