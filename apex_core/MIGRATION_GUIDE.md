# Migration Guide: Extracting Modules to signal-engine-core

## Overview

This guide shows how to systematically extract core modules from AlphaEngine/MoonWire into signal-engine-core.

## Phase 1: Core ML Modules (Priority 1)

### Files to Extract

From AlphaEngine `scripts/ml/` → signal-engine-core `signal_engine/ml/`:

| Source File | Destination | Notes |
|------------|-------------|-------|
| scripts/ml/train_predict.py | signal_engine/ml/train.py | Main training orchestration |
| scripts/ml/cv_eval.py | signal_engine/ml/cv_eval.py | Cross-validation |
| scripts/ml/tuner.py | signal_engine/ml/tuner.py | Hyperparameter tuning |
| scripts/ml/backtest.py | signal_engine/ml/backtest.py | Backtesting framework |
| scripts/ml/feature_builder.py | signal_engine/ml/feature_builder.py | Feature engineering |
| scripts/ml/data_loader.py | signal_engine/ml/data_loader.py | Data loading |
| scripts/ml/regime_detector.py | signal_engine/ml/regime_detector.py | Market regimes |
| scripts/ml/per_regime_trainer.py | signal_engine/ml/per_regime_trainer.py | Per-regime models |
| src/ml/infer.py | signal_engine/ml/infer.py | Inference engine |
| src/ml/metrics.py | signal_engine/ml/metrics.py | ML metrics |

### Extraction Steps

1. **Copy file to core**
   ```bash
   cp AlphaEngine/src/ml/metrics.py signal-engine-core/signal_engine/ml/metrics.py
   ```

2. **Remove product-specific code**
   - Remove hardcoded ticker references (SPY, QQQ, BTC, ETH)
   - Replace with parameters passed from product layer
   - Remove direct price fetcher imports
   - Use abstract interfaces instead

3. **Update imports**
   ```python
   # Before (product-specific)
   from src.price_fetcher import get_price

   # After (core with interface)
   from signal_engine.interfaces import PriceProvider

   def calculate_feature(asset: str, price_provider: PriceProvider):
       price = price_provider.get_price(asset)
   ```

4. **Add to core __init__.py**
   ```python
   # signal_engine/ml/__init__.py
   from signal_engine.ml.train import train_model
   from signal_engine.ml.infer import InferenceEngine
   from signal_engine.ml.metrics import calculate_metrics
   ```

## Phase 2: Governance Modules (Priority 1)

From `scripts/governance/` → `signal_engine/governance/`:

| Source File | Destination |
|------------|-------------|
| scripts/governance/auto_adjust_governance.py | signal_engine/governance/auto_adjust.py |
| scripts/governance/drift_response.py | signal_engine/governance/drift_response.py |
| scripts/governance/retrain_automation.py | signal_engine/governance/retrain_automation.py |
| scripts/governance/model_lineage.py | signal_engine/governance/model_lineage.py |
| scripts/governance/bluegreen_promotion.py | signal_engine/governance/bluegreen_promotion.py |

## Phase 3: Validation Modules (Priority 2)

From `scripts/summary_sections/` → `signal_engine/validation/`:

| Source File | Destination |
|------------|-------------|
| scripts/summary_sections/calibration*.py | signal_engine/validation/calibration.py |
| scripts/summary_sections/performance_validation.py | signal_engine/validation/performance.py |
| scripts/summary_sections/accuracy*.py | signal_engine/validation/accuracy.py |
| src/feedback_reliability.py | signal_engine/validation/reliability.py |

## Phase 4: Threshold Optimization (Priority 2)

From various locations → `signal_engine/threshold/`:

| Source File | Destination |
|------------|-------------|
| src/threshold_simulator.py | signal_engine/threshold/simulator.py |
| scripts/summary_sections/threshold*.py | signal_engine/threshold/ |

## Phase 5: Analytics (Priority 3)

From `src/analytics/` → `signal_engine/analytics/`:

Copy entire `src/analytics/` directory with minimal changes (already mostly product-agnostic).

## Example: Extracting metrics.py

### Before (Product-Specific)

```python
# AlphaEngine/src/ml/metrics.py
from src.price_fetcher import get_price_usd

def calculate_returns(asset: str):
    price = get_price_usd(asset)  # Hardcoded to Yahoo Finance
    # ... calculations
    return returns
```

### After (Core with Interface)

```python
# signal-engine-core/signal_engine/ml/metrics.py
from signal_engine.interfaces import PriceProvider
from typing import Optional

def calculate_returns(
    asset: str,
    price_provider: PriceProvider  # Injected dependency
) -> Optional[float]:
    """
    Calculate returns for an asset.

    Args:
        asset: Asset ticker/symbol
        price_provider: Implementation of PriceProvider interface

    Returns:
        Return value or None if price unavailable
    """
    price = price_provider.get_price(asset)
    if price is None:
        return None
    # ... calculations
    return returns
```

### Product Integration (AlphaEngine)

```python
# AlphaEngine/src/ml/metrics_integration.py
from signal_engine.ml.metrics import calculate_returns
from alphaengine.adapters import YahooFinanceAdapter

# Create product-specific adapter
price_provider = YahooFinanceAdapter()

# Use core function with product adapter
returns = calculate_returns("SPY", price_provider)
```

### Product Integration (MoonWire)

```python
# moonwire-backend/moonwire/ml/metrics_integration.py
from signal_engine.ml.metrics import calculate_returns
from moonwire.adapters import CoinGeckoAdapter

# Create product-specific adapter
price_provider = CoinGeckoAdapter()

# Use core function with product adapter
returns = calculate_returns("BTC", price_provider)
```

## Refactoring Patterns

### Pattern 1: Dependency Injection

**Before:**
```python
# Hardcoded dependency
from src.price_fetcher import bulk_price_fetch

def train_model(assets):
    prices = bulk_price_fetch(assets)
    # ...
```

**After:**
```python
# Injected dependency
from signal_engine.interfaces import PriceProvider

def train_model(assets: List[str], price_provider: PriceProvider):
    prices = price_provider.bulk_price_fetch(assets)
    # ...
```

### Pattern 2: Configuration Objects

**Before:**
```python
# Hardcoded tickers
ASSETS = ["SPY", "QQQ", "XLF"]
```

**After:**
```python
# Passed as parameter
def train_model(assets: List[str], ...):
    # No hardcoded tickers
```

### Pattern 3: Abstract Base Classes

**Before:**
```python
# Direct implementation
def get_sentiment(asset):
    return fetch_cryptopanic(asset)
```

**After:**
```python
# Interface-based
from abc import ABC, abstractmethod

class SentimentProvider(ABC):
    @abstractmethod
    def fetch_sentiment(self, asset: str) -> float:
        pass

# Products implement
class CryptoPanicAdapter(SentimentProvider):
    def fetch_sentiment(self, asset: str) -> float:
        return fetch_cryptopanic(asset)
```

## Testing Strategy

### Core Tests
Test core logic with mock providers:

```python
# tests/ml/test_metrics.py
from signal_engine.ml.metrics import calculate_returns
from signal_engine.interfaces import PriceProvider

class MockPriceProvider(PriceProvider):
    def get_price(self, asset: str) -> float:
        return 100.0  # Mock price

def test_calculate_returns():
    provider = MockPriceProvider()
    returns = calculate_returns("TEST", provider)
    assert returns is not None
```

### Integration Tests (in Products)
Test with real adapters:

```python
# AlphaEngine/tests/test_metrics_integration.py
from signal_engine.ml.metrics import calculate_returns
from alphaengine.adapters import YahooFinanceAdapter

def test_calculate_returns_yahoo():
    provider = YahooFinanceAdapter()
    returns = calculate_returns("SPY", provider)
    assert returns is not None
```

## Checklist for Each Module

- [ ] Copy file to core repo
- [ ] Remove hardcoded tickers
- [ ] Replace direct imports with interface injections
- [ ] Add type hints
- [ ] Update docstrings
- [ ] Add to module __init__.py
- [ ] Write unit tests with mocks
- [ ] Update core version if API changes
- [ ] Test integration in both products

## Version Management

When extracting modules, increment core version:
- **Major** (1.0.0 → 2.0.0): Breaking API changes
- **Minor** (1.0.0 → 1.1.0): New features, backwards-compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes

Products update their `requirements.txt`:
```txt
signal-engine-core>=1.1.0,<2.0.0  # Allow minor updates
```

## Next Steps

1. **Start with Priority 1 modules** (ML & Governance)
2. **Extract one module completely** as a proof of concept
3. **Test in both products** (moonwire-backend and alphaengine)
4. **Iterate** on remaining modules
5. **Publish to PyPI** (or private package registry)
6. **Update product dependencies** to use published package

---

**Ready to begin?** Start by extracting `src/ml/metrics.py` as shown in the example above.
