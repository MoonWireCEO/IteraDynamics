# Signal Engine Core - Usage Examples

This document demonstrates how to use the extracted modules from signal-engine-core in your products (AlphaEngine, MoonWire, or any other signal generation system).

## Installation

```bash
pip install -r requirements.txt
```

Or if published:
```bash
pip install signal-engine-core
```

## Example 1: Model Inference

### Using InferenceEngine

```python
import joblib
import json
from signal_engine.ml import InferenceEngine

# Load your trained model and metadata
model = joblib.load("models/my_model.joblib")
with open("models/metadata.json") as f:
    metadata = json.load(f)
    feature_order = metadata["feature_order"]

# Create inference engine
engine = InferenceEngine(
    model=model,
    feature_order=feature_order,
    metadata=metadata
)

# Make predictions with explanation
features = {
    "burst_z": 2.5,
    "momentum_7d": 0.15,
    "volatility_14d": 0.08,
    # ... other features
}

result = engine.predict_proba(features, explain=True, top_n=5)
print(f"Probability: {result['probability']:.3f}")
print(f"Top contributors: {result['contributions']}")

# Make binary prediction
prediction = engine.predict(features, threshold=0.6)
print(f"Prediction: {prediction['prediction']} (prob: {prediction['probability']:.3f})")
```

### Using EnsembleInferenceEngine

```python
from signal_engine.ml import EnsembleInferenceEngine
import joblib

# Load multiple models
models_config = {
    "logistic": {
        "model": joblib.load("models/logistic.joblib"),
        "feature_order": ["feature1", "feature2", "feature3"],
        "metadata": {"type": "logistic"}
    },
    "random_forest": {
        "model": joblib.load("models/rf.joblib"),
        "feature_order": ["feature1", "feature2", "feature3"],
        "metadata": {"type": "rf"}
    },
    "gradient_boosting": {
        "model": joblib.load("models/gb.joblib"),
        "feature_order": ["feature1", "feature2", "feature3"],
        "metadata": {"type": "gb"}
    }
}

# Create ensemble
ensemble = EnsembleInferenceEngine(
    models=models_config,
    aggregation="mean"  # or "median", "max", "min"
)

# Get ensemble prediction
result = ensemble.predict_proba(features)
print(f"Ensemble probability: {result['probability']:.3f}")
print(f"Individual votes: {result['votes']}")
print(f"Range: {result['low']:.3f} - {result['high']:.3f}")
```

## Example 2: Computing Model Metrics

```python
from signal_engine.ml import compute_accuracy_by_version
from pathlib import Path

# Compute per-version accuracy from trigger and label logs
results = compute_accuracy_by_version(
    trigger_log_path=Path("logs/triggers.jsonl"),
    label_log_path=Path("logs/labels.jsonl"),
    window_hours=72,  # Look back 3 days
    match_window_minutes=5,  # Match triggers within ±5 minutes of labels
    dedup_one_label_per_trigger=True
)

# Access overall metrics (micro-aggregated)
micro = results["_micro"]
print(f"Overall Precision: {micro['precision']:.3f}")
print(f"Overall Recall: {micro['recall']:.3f}")
print(f"Overall F1: {micro['f1']:.3f}")
print(f"TP: {micro['tp']}, FP: {micro['fp']}, FN: {micro['fn']}")

# Access per-version metrics
for version, metrics in results.items():
    if not version.startswith("_"):
        print(f"\nVersion {version}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  N: {metrics['n']}")

# Access macro-averaged metrics
macro = results["_macro"]
print(f"\nMacro-averaged Precision: {macro['precision']:.3f}")
print(f"Macro-averaged Recall: {macro['recall']:.3f}")
print(f"Versions analyzed: {macro['versions']}")
```

## Example 3: Backtesting a Strategy

```python
import pandas as pd
from signal_engine.ml import run_backtest, Trade

# Prepare predictions DataFrame
predictions = pd.DataFrame({
    'ts': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'p_long': [0.45, 0.62, 0.58, ...]  # Model predictions
})

# Prepare prices DataFrame
prices = pd.DataFrame({
    'ts': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'close': [100.5, 101.2, 100.8, ...]  # Historical prices
})

# Run backtest
results = run_backtest(
    pred_df=predictions,
    prices_df=prices,
    conf_min=0.6,  # Only enter when confidence >= 60%
    debounce_min=120,  # Wait 2 hours between trades
    horizon_h=24,  # Hold for 24 hours
    fees_bps=1.0,  # 0.01% fees
    slippage_bps=2.0,  # 0.02% slippage
    symbol="SPY"
)

# Access metrics
metrics = results["metrics"]
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Signals Per Day: {metrics['signals_per_day']:.2f}")
print(f"Total Trades: {metrics['n_trades']}")

# Access individual trades
trades = results["trades"]
for trade in trades[:5]:  # Show first 5
    print(f"\nTrade: {trade.symbol} {trade.side}")
    print(f"  Entry: {trade.ts_entry} @ ${trade.entry_px:.2f}")
    print(f"  Exit: {trade.ts_exit} @ ${trade.exit_px:.2f}")
    print(f"  PnL: {trade.pnl_pct:.2%}")

# Access equity curve
equity = results["equity"]
equity_df = pd.DataFrame(equity)
print(f"\nFinal Equity: {equity_df['equity'].iloc[-1]:.2f}")
```

## Example 4: Regime Detection

```python
import pandas as pd
from signal_engine.ml import detect_market_regime, get_regime_stats

# Prepare price data
prices = pd.DataFrame({
    'ts': pd.date_range('2024-01-01', periods=365),
    'close': [...]  # Daily closing prices
}).set_index('ts')

# Detect regime
regime = detect_market_regime(
    prices_df=prices,
    symbol="BTC",
    volatility_threshold=0.06,  # 6% relative volatility
    trend_strength_threshold=0.08,  # 8% trend strength
    lookback_volatility=14,  # 14-day volatility window
    lookback_trend=30  # 30-day trend window
)

print(regime.value_counts())

# Get regime statistics
stats = get_regime_stats(prices_df=prices, symbol="BTC")
print(f"\nRegime Statistics for {stats['symbol']}:")
print(f"  Total periods: {stats['total_periods']}")
print(f"  Trending: {stats['trending_periods']} ({stats['trending_pct']:.1f}%)")
print(f"  Choppy: {stats['choppy_periods']} ({stats['choppy_pct']:.1f}%)")

# Add regime as a feature to your feature DataFrame
from signal_engine.ml import add_regime_feature

features = pd.DataFrame({...})  # Your features
features_with_regime = add_regime_feature(
    df=features,
    prices_df=prices,
    symbol="BTC"
)
print(features_with_regime['regime_trending'].describe())

# Filter data to only trending periods
from signal_engine.ml import filter_by_regime

trending_only = filter_by_regime(
    df=features,
    prices_df=prices,
    keep_regime="trending"
)
print(f"Filtered to {len(trending_only)} trending periods from {len(features)} total")
```

## Example 5: Volatility-Adjusted Thresholds

```python
from signal_engine.ml import compute_volatility_adjusted_threshold

# Define regime-specific multipliers
multipliers = {
    "calm": 0.9,      # Lower threshold in calm markets (more sensitive)
    "normal": 1.0,    # Normal threshold
    "turbulent": 1.1  # Higher threshold in turbulent markets (less sensitive)
}

# Get adjusted threshold for current regime
result = compute_volatility_adjusted_threshold(
    base_threshold=0.5,
    regime="turbulent",
    regime_multipliers=multipliers
)

print(f"Base threshold: {result['base_threshold']}")
print(f"Regime: {result['volatility_regime']}")
print(f"Multiplier: {result['regime_multiplier']}")
print(f"Adjusted threshold: {result['threshold_after_volatility']}")

# Use in decision logic
prediction_prob = 0.53
threshold = result['threshold_after_volatility']
decision = prediction_prob >= threshold
print(f"\nProbability: {prediction_prob:.3f}")
print(f"Threshold: {threshold:.3f}")
print(f"Decision: {'TRIGGER' if decision else 'NO TRIGGER'}")
```

## Integration Examples

### AlphaEngine Integration

```python
# alphaengine/ml/metrics_integration.py
from signal_engine.ml import compute_accuracy_by_version
from alphaengine.config import TRIGGER_LOG_PATH, LABEL_LOG_PATH

def get_model_accuracy():
    """AlphaEngine-specific wrapper for metrics computation."""
    return compute_accuracy_by_version(
        trigger_log_path=TRIGGER_LOG_PATH,
        label_log_path=LABEL_LOG_PATH,
        window_hours=72
    )
```

### MoonWire Integration

```python
# moonwire/ml/inference.py
from signal_engine.ml import InferenceEngine
import joblib

class MoonWireInferenceService:
    """MoonWire-specific inference service using core engine."""

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.feature_order = self._load_feature_order()
        self.engine = InferenceEngine(
            model=self.model,
            feature_order=self.feature_order
        )

    def predict_crypto_signal(self, asset: str, features: dict):
        """Crypto-specific prediction wrapper."""
        result = self.engine.predict_proba(features, explain=True)
        # Add MoonWire-specific processing
        return {
            "asset": asset,
            "probability": result["probability"],
            "signal": "BUY" if result["probability"] >= 0.6 else "HOLD",
            "contributors": result.get("contributions", {})
        }
```

## Testing

```python
# tests/ml/test_inference.py
import pytest
from signal_engine.ml import InferenceEngine
from sklearn.linear_model import LogisticRegression
import numpy as np

def test_inference_engine():
    # Create mock model
    model = LogisticRegression()
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)

    # Create engine
    engine = InferenceEngine(
        model=model,
        feature_order=["feature1", "feature2", "feature3"]
    )

    # Test prediction
    features = {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
    result = engine.predict_proba(features)

    assert "probability" in result
    assert 0 <= result["probability"] <= 1

def test_backtest():
    import pandas as pd
    from signal_engine.ml import run_backtest

    pred_df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=100, freq='H'),
        'p_long': np.random.rand(100)
    })
    prices_df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=100, freq='H'),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5)
    })

    results = run_backtest(
        pred_df=pred_df,
        prices_df=prices_df,
        conf_min=0.6,
        debounce_min=60,
        horizon_h=6
    )

    assert "metrics" in results
    assert "trades" in results
    assert "equity" in results
```

## Advanced: Custom Adapters

```python
# Create custom adapter for your data source
from signal_engine.ml import InferenceEngine
from abc import ABC, abstractmethod

class ModelLoader(ABC):
    """Abstract base class for loading models."""

    @abstractmethod
    def load_model(self, version: str):
        pass

    @abstractmethod
    def get_feature_order(self, version: str):
        pass

class AlphaEngineModelLoader(ModelLoader):
    """AlphaEngine-specific model loader."""

    def load_model(self, version: str):
        import joblib
        from alphaengine.paths import MODELS_DIR
        return joblib.load(MODELS_DIR / version / "model.joblib")

    def get_feature_order(self, version: str):
        import json
        from alphaengine.paths import MODELS_DIR
        with open(MODELS_DIR / version / "metadata.json") as f:
            return json.load(f)["feature_order"]

# Use adapter
loader = AlphaEngineModelLoader()
model = loader.load_model("v1.2.3")
features = loader.get_feature_order("v1.2.3")
engine = InferenceEngine(model=model, feature_order=features)
```

## Summary

The extracted modules provide clean, product-agnostic interfaces that can be used across different signal generation systems. Key benefits:

- **No hardcoded dependencies** - All paths, data sources, and configurations are injected
- **Type-safe interfaces** - Clear function signatures with type hints
- **Comprehensive documentation** - Detailed docstrings for all public APIs
- **Testable** - Easy to mock and test with standard testing frameworks
- **Reusable** - Same code works for stocks, crypto, or any other asset class
