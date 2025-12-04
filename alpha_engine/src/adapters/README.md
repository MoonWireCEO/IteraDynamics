# AlphaEngine Adapters for signal-engine-core

This directory contains adapters that bridge AlphaEngine's specific implementations with signal-engine-core's generic interfaces.

## Overview

The adapter pattern allows AlphaEngine to use signal-engine-core's product-agnostic ML modules while maintaining AlphaEngine-specific data sources and model storage.

## Adapters

### YahooFinanceAdapter
**File:** `price_adapter.py`

Implements `PriceProvider` interface using Yahoo Finance (yfinance).

```python
from src.adapters import YahooFinanceAdapter

adapter = YahooFinanceAdapter()

# Get current price
price = adapter.get_price("SPY")

# Get historical prices
history = adapter.get_price_history("SPY", "2025-01-01", "2025-10-30")

# Bulk fetch
prices = adapter.bulk_price_fetch(["SPY", "QQQ", "DIA"])
```

### AlphaEngineModelLoader
**File:** `model_loader.py`

Loads models from AlphaEngine's `models/` directory structure.

```python
from src.adapters import AlphaEngineModelLoader

loader = AlphaEngineModelLoader()

# Load latest model
model, features, metadata = loader.load_latest_model("current")

# Load ensemble models
ensemble_configs = loader.load_ensemble_models()

# Get version
version = loader.get_model_version()
```

### AlphaEngineInferenceService
**File:** `inference_service.py`

High-level service that combines the adapters with signal-engine-core's InferenceEngine.

```python
from src.adapters import AlphaEngineInferenceService

service = AlphaEngineInferenceService()

# Initialize with ensemble
service.initialize_ensemble()

# Make prediction
features = {...}  # Your features
result = service.predict(features, explain=True)

# Predict for specific asset
result = service.predict_for_asset("SPY", features, include_price=True)
```

## Architecture

```
AlphaEngine
├── src/adapters/              # This directory
│   ├── price_adapter.py       # Yahoo Finance → PriceProvider
│   ├── model_loader.py        # AlphaEngine models → core format
│   ├── inference_service.py   # High-level service
│   └── __init__.py
│
└── signal-engine-core         # External package
    ├── signal_engine/
    │   ├── ml/                # Core ML modules
    │   │   ├── infer.py       # InferenceEngine, EnsembleInferenceEngine
    │   │   ├── metrics.py     # Performance metrics
    │   │   ├── backtest.py    # Backtesting
    │   │   └── ...
    │   ├── interfaces/        # Abstract interfaces
    │   │   └── data_providers.py  # PriceProvider, etc.
    │   └── ...
```

## Usage Examples

### Basic Usage

```python
from src.adapters import AlphaEngineInferenceService

# Initialize
service = AlphaEngineInferenceService()
service.initialize_ensemble()

# Get features from your feature builder
features = {
    "burst_z": 2.5,
    "momentum_7d": 0.15,
    # ... more features
}

# Predict
result = service.predict(features)
print(f"Probability: {result['probability']}")
```

### With Explanations

```python
result = service.predict(features, explain=True, top_n=5)

print(f"Probability: {result['probability']}")
print(f"Top contributors:")
for feature, contribution in result['contributions'].items():
    print(f"  {feature}: {contribution}")
```

### Asset-Specific Prediction

```python
result = service.predict_for_asset(
    asset="SPY",
    features=features,
    include_price=True
)

print(f"Asset: {result['asset']}")
print(f"Probability: {result['prediction']['probability']}")
print(f"Current Price: ${result['current_price']:.2f}")
```

### Using Core Modules Directly

You can also use signal-engine-core modules directly:

```python
from signal_engine.ml import run_backtest, detect_market_regime
from src.adapters import YahooFinanceAdapter
import pandas as pd

# Get price data
adapter = YahooFinanceAdapter()
prices = adapter.get_price_history("SPY", "2025-01-01")

# Detect regime
regime = detect_market_regime(prices, symbol="SPY")
print(f"Current regime: {regime.iloc[-1]}")

# Run backtest
predictions = pd.DataFrame({...})  # Your predictions
results = run_backtest(
    pred_df=predictions,
    prices_df=prices,
    conf_min=0.6,
    debounce_min=120,
    horizon_h=24
)
```

## Benefits

✅ **Separation of Concerns** - AlphaEngine-specific code separate from core logic
✅ **Reusability** - Same core modules work for MoonWire and future products
✅ **Testability** - Easy to mock adapters for unit testing
✅ **Maintainability** - Updates to core benefit all products
✅ **Flexibility** - Swap data sources without changing core code

## Testing

See `examples/use_signal_engine_core.py` for a complete working example.

## Next Steps

1. **Replace existing inference** - Update `src/ml/infer.py` to use the service
2. **Add metrics** - Use `signal_engine.ml.compute_accuracy_by_version`
3. **Add backtesting** - Use `signal_engine.ml.run_backtest`
4. **Add CV** - Use `signal_engine.ml.walk_forward_cv`
5. **Add tuning** - Use `signal_engine.ml.grid_search_thresholds`

## Documentation

- **signal-engine-core docs**: `../signal-engine-core/README_MIGRATION.md`
- **Usage examples**: `../signal-engine-core/USAGE_EXAMPLES.md`
- **Integration guide**: `../signal-engine-core/INTEGRATION_ADAPTERS.md`
