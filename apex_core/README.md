# Signal Engine Core

**Shared ML/Governance/Validation Engine for Multi-Asset Signal Generation Platforms**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Signal Engine Core is the shared "brain" that powers multiple signal generation platforms:
- **MoonWire** (cryptocurrency trading signals)
- **AlphaEngine** (securities/equities trading signals)

By extracting the core ML, governance, and validation logic into this shared package, we ensure:
- ✅ ML improvements benefit all products simultaneously
- ✅ Single source of truth for signal generation algorithms
- ✅ Consistent governance and drift detection across platforms
- ✅ Independent versioning and deployment

## Architecture

```
moonwire-backend ──┐
                   ├──> signal-engine-core (this package)
alphaengine ───────┘
```

Products implement abstract interfaces and provide product-specific data sources (price feeds, sentiment sources, etc.), while the core handles all ML training, inference, governance, and validation.

## Features

### 🧠 Machine Learning
- Model training with multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression)
- Real-time inference engine
- Feature engineering and data preprocessing
- Hyperparameter tuning with cross-validation
- Regime-aware modeling
- Probability calibration (Platt scaling, isotonic regression)

### ⚖️ Governance
- Automatic drift detection
- Auto-retraining when drift exceeds thresholds
- Model lineage tracking
- Blue-green deployment for model promotion
- Governance alerts and notifications

### ✅ Validation
- Signal calibration metrics (ECE, MCE)
- Performance validation frameworks
- Accuracy tracking per origin/version
- Rolling accuracy snapshots
- Per-origin calibration

### 🎯 Threshold Optimization
- Dynamic threshold tuning
- Backtest-driven optimization
- Per-origin threshold policies
- Volatility-aware thresholds

### 📊 Analytics
- Lead-lag analysis between signals
- Burst detection
- Cross-origin correlations
- Volatility regime detection
- Nowcast attention mechanisms

### 📈 Performance Simulation
- Paper trading simulator
- Shadow signal backfilling
- Historical replay capabilities
- P&L tracking with fees/slippage

## Installation

### From PyPI (when published)
```bash
pip install signal-engine-core
```

### From Source
```bash
git clone https://github.com/yourusername/signal-engine-core.git
cd signal-engine-core
pip install -e .
```

### Development
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Implement Data Provider Interfaces

Products must implement the abstract interfaces to provide data to the core:

```python
from signal_engine.interfaces import PriceProvider, SentimentProvider

class YourPriceProvider(PriceProvider):
    def get_price(self, asset: str) -> float:
        # Your implementation (CoinGecko, Yahoo Finance, etc.)
        return fetch_price_from_api(asset)

    def bulk_price_fetch(self, assets: List[str]) -> Dict[str, float]:
        # Bulk fetch implementation
        return {asset: fetch_price_from_api(asset) for asset in assets}

    def get_historical_prices(self, asset, start_time, end_time, interval):
        # Historical data implementation
        return fetch_historical_data(asset, start_time, end_time)

class YourSentimentProvider(SentimentProvider):
    def fetch_sentiment(self, asset: str) -> float:
        # Your sentiment source (CryptoPanic, NewsAPI, etc.)
        return calculate_sentiment(asset)

    def fetch_bulk_sentiment(self, assets: List[str]) -> Dict[str, float]:
        return {asset: calculate_sentiment(asset) for asset in assets}
```

### 2. Use Core ML Engine

```python
from signal_engine.ml import train, infer
from signal_engine.core import SignalGenerator

# Initialize with your data providers
signal_gen = SignalGenerator(
    price_provider=YourPriceProvider(),
    sentiment_provider=YourSentimentProvider()
)

# Generate signals
signal = signal_gen.generate_signal(asset="SPY")
```

### 3. Run Governance

```python
from signal_engine.governance import DriftDetector, AutoRetrainer

# Detect drift
drift_detector = DriftDetector()
if drift_detector.check_drift(model_id="model_v1"):
    # Auto-retrain
    retrainer = AutoRetrainer()
    new_model = retrainer.retrain()
```

## Module Structure

```
signal_engine/
├── ml/                  # ML training, inference, metrics
├── governance/          # Drift detection, auto-retraining
├── validation/          # Calibration, performance validation
├── threshold/           # Threshold optimization
├── analytics/           # Lead-lag, correlations, regimes
├── performance/         # Paper trading, backtesting
├── core/                # Signal generation, composition
└── interfaces/          # Abstract interfaces for products
```

## Integration Examples

### MoonWire (Crypto)
```python
from signal_engine import PriceProvider
import requests

class CoinGeckoAdapter(PriceProvider):
    def get_price(self, asset: str) -> float:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        # ... CoinGecko implementation
```

### AlphaEngine (Securities)
```python
from signal_engine import PriceProvider
import yfinance as yf

class YahooFinanceAdapter(PriceProvider):
    def get_price(self, asset: str) -> float:
        ticker = yf.Ticker(asset)
        return ticker.info['regularMarketPrice']
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black signal_engine/ tests/
```

### Type Checking
```bash
mypy signal_engine/
```

## Versioning

This package follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

Products can pin to specific versions for stability:
```python
# requirements.txt
signal-engine-core==1.0.0  # Pin to exact version
signal-engine-core>=1.0.0,<2.0.0  # Allow minor/patch updates
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details

## Support

- Issues: https://github.com/yourusername/signal-engine-core/issues
- Documentation: https://docs.signalengine.io
- Email: dev@signalengine.io
