# Next Steps: From Foundation to Full Implementation

## Current Status ✅

You now have:
- ✅ **signal-engine-core** - Fully structured Python package
- ✅ **Abstract interfaces** - PriceProvider, SentimentProvider, MarketDataProvider
- ✅ **Package setup** - setup.py, requirements.txt, proper structure
- ✅ **Documentation** - README, MIGRATION_GUIDE, GITHUB_SETUP
- ✅ **CI/CD workflow** - Automated testing and linting
- ✅ **Git repository** - Clean commit history, ready for GitHub

**Location:** `C:\Users\North East Collision\Desktop\signal-engine-core`

---

## Step 1: Push to GitHub (Do This First!)

Follow instructions in `GITHUB_SETUP.md`:

```bash
# 1. Create repo on GitHub
#    Go to: https://github.com/new
#    Name: signal-engine-core
#    Visibility: Private
#    Don't initialize with anything

# 2. Add remote and push
cd /c/Users/"North East Collision"/Desktop/signal-engine-core
git remote add origin https://github.com/YOUR_USERNAME/signal-engine-core.git
git branch -M main
git push -u origin main
```

---

## Step 2: Extract First Module (Proof of Concept)

Start with `src/ml/metrics.py` - it's relatively simple and has clear interfaces.

### 2.1 Copy and Refactor

```bash
# From AlphaEngine directory
cd /c/Users/"North East Collision"/Desktop/AlphaEngine

# Copy to core
cp src/ml/metrics.py ../signal-engine-core/signal_engine/ml/metrics.py
```

### 2.2 Refactor for Interfaces

Edit `signal-engine-core/signal_engine/ml/metrics.py`:

**Remove:** Direct imports to product-specific code
```python
# Remove these
from src.price_fetcher import get_price_usd
```

**Add:** Interface-based dependencies
```python
# Add these
from signal_engine.interfaces import PriceProvider
from typing import Optional

def calculate_metric(
    asset: str,
    price_provider: PriceProvider  # Injected dependency
) -> Optional[float]:
    price = price_provider.get_price(asset)
    if price is None:
        return None
    # ... rest of logic
```

### 2.3 Update Core __init__.py

```python
# signal_engine/ml/__init__.py
from signal_engine.ml.metrics import calculate_metric

__all__ = ["calculate_metric"]
```

### 2.4 Create Unit Tests

```python
# tests/ml/test_metrics.py
import pytest
from signal_engine.ml.metrics import calculate_metric
from signal_engine.interfaces import PriceProvider

class MockPriceProvider(PriceProvider):
    def get_price(self, asset: str) -> float:
        return 100.0

    def bulk_price_fetch(self, assets):
        return {a: 100.0 for a in assets}

    def get_historical_prices(self, asset, start, end, interval):
        return []

def test_calculate_metric():
    provider = MockPriceProvider()
    result = calculate_metric("TEST", provider)
    assert result is not None
```

### 2.5 Commit to Core

```bash
cd ../signal-engine-core
git add .
git commit -m "Add ML metrics module with interface-based dependencies"
git push origin main
```

---

## Step 3: Create Adapters in Products

### 3.1 AlphaEngine Adapter

```python
# AlphaEngine/alphaengine/adapters/__init__.py
from alphaengine.adapters.yahoo_finance_adapter import YahooFinanceAdapter
```

```python
# AlphaEngine/alphaengine/adapters/yahoo_finance_adapter.py
from signal_engine.interfaces import PriceProvider
from typing import Dict, List, Optional
from datetime import datetime
import requests

class YahooFinanceAdapter(PriceProvider):
    """Yahoo Finance implementation of PriceProvider for securities"""

    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"

    def get_price(self, asset: str) -> Optional[float]:
        try:
            url = f"{self.base_url}/{asset}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            result = data.get("chart", {}).get("result", [])
            if result and len(result) > 0:
                meta = result[0].get("meta", {})
                return meta.get("regularMarketPrice")
            return None
        except Exception as e:
            print(f"Error fetching price for {asset}: {e}")
            return None

    def bulk_price_fetch(self, assets: List[str]) -> Dict[str, float]:
        # Implement bulk fetch
        return {asset: self.get_price(asset) for asset in assets}

    def get_historical_prices(self, asset, start_time, end_time, interval):
        # Implement historical data
        return []
```

### 3.2 MoonWire Adapter

```python
# moonwire-backend/moonwire/adapters/coingecko_adapter.py
from signal_engine.interfaces import PriceProvider
from typing import Dict, List, Optional
from datetime import datetime
import requests

class CoinGeckoAdapter(PriceProvider):
    """CoinGecko implementation of PriceProvider for cryptocurrencies"""

    COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "DOGE": "dogecoin",
        "SOL": "solana",
        "ADA": "cardano"
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_price(self, asset: str) -> Optional[float]:
        asset = asset.upper()
        cg_id = self.COINGECKO_IDS.get(asset)
        if not cg_id:
            return None

        try:
            url = f"{self.base_url}/simple/price?ids={cg_id}&vs_currencies=usd"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data[cg_id]["usd"]
        except Exception as e:
            print(f"Error fetching price for {asset}: {e}")
            return None

    def bulk_price_fetch(self, assets: List[str]) -> Dict[str, float]:
        # Implement bulk fetch
        return {asset: self.get_price(asset) for asset in assets}

    def get_historical_prices(self, asset, start_time, end_time, interval):
        # Implement historical data
        return []
```

---

## Step 4: Install Core in Products

### 4.1 Local Development Install

**AlphaEngine:**
```bash
cd /c/Users/"North East Collision"/Desktop/AlphaEngine

# Add to requirements.txt
echo "signal-engine-core @ file:///C:/Users/North%20East%20Collision/Desktop/signal-engine-core" >> requirements.txt

# Install
pip install -e .
```

**MoonWire:**
```bash
cd /c/Users/"North East Collision"/Desktop/moonwire-backend

# Add to requirements.txt
echo "signal-engine-core @ file:///C:/Users/North%20East%20Collision/Desktop/signal-engine-core" >> requirements.txt

# Install
pip install -e .
```

### 4.2 Use Core in Products

**AlphaEngine:**
```python
# AlphaEngine/src/ml/metrics_wrapper.py
from signal_engine.ml.metrics import calculate_metric
from alphaengine.adapters import YahooFinanceAdapter

# Initialize adapter once
price_provider = YahooFinanceAdapter()

# Use core function
def get_metric_for_asset(asset: str):
    return calculate_metric(asset, price_provider)
```

**MoonWire:**
```python
# moonwire-backend/moonwire/ml/metrics_wrapper.py
from signal_engine.ml.metrics import calculate_metric
from moonwire.adapters import CoinGeckoAdapter

# Initialize adapter once
price_provider = CoinGeckoAdapter()

# Use core function
def get_metric_for_asset(asset: str):
    return calculate_metric(asset, price_provider)
```

---

## Step 5: Continue Extracting Modules

Follow this priority order:

**Priority 1 (Critical):**
1. ✅ `src/ml/metrics.py` → `signal_engine/ml/metrics.py`
2. `src/ml/infer.py` → `signal_engine/ml/infer.py`
3. `scripts/ml/train_predict.py` → `signal_engine/ml/train.py`
4. `scripts/governance/*` → `signal_engine/governance/`

**Priority 2 (Important):**
5. `scripts/ml/feature_builder.py` → `signal_engine/ml/feature_builder.py`
6. `scripts/ml/cv_eval.py` → `signal_engine/ml/cv_eval.py`
7. `src/threshold_simulator.py` → `signal_engine/threshold/simulator.py`
8. `scripts/summary_sections/calibration*.py` → `signal_engine/validation/`

**Priority 3 (Nice to Have):**
9. `src/analytics/*` → `signal_engine/analytics/`
10. `scripts/perf/*` → `signal_engine/performance/`

---

## Step 6: Publish to Package Registry

### Option A: PyPI (Public)

```bash
cd /c/Users/"North East Collision"/Desktop/signal-engine-core

# Install build tools
pip install build twine

# Build package
python -m build

# Upload to PyPI
twine upload dist/*

# Then in products:
# requirements.txt:
# signal-engine-core==1.0.0
```

### Option B: Private GitHub Package Registry

```yaml
# .github/workflows/publish.yml
name: Publish Package

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Publish to GitHub Packages
      run: |
        python -m build
        pip install twine
        twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```

---

## Verification Checklist

Before considering extraction complete:

- [ ] Core package installs without errors
- [ ] All tests pass in core repo
- [ ] Adapters created for both products
- [ ] Products can import and use core functions
- [ ] No circular dependencies
- [ ] Documentation updated
- [ ] Version numbers aligned
- [ ] CI/CD pipeline green

---

## Timeline Estimate

- **Step 1** (GitHub setup): 10 minutes
- **Step 2** (First module): 2-4 hours
- **Step 3** (Adapters): 2-3 hours
- **Step 4** (Integration): 1-2 hours
- **Step 5** (Full extraction): 2-3 weeks (iterative)
- **Step 6** (Publishing): 1-2 hours

**Total for MVP (Steps 1-4):** 1-2 days

---

## Success Criteria

You'll know the architecture is working when:

1. ✅ You make a change to core ML logic
2. ✅ Both MoonWire and AlphaEngine get the improvement automatically
3. ✅ Products only differ in their adapters and configuration
4. ✅ CI/CD catches breaking changes before they reach products
5. ✅ New products can be added easily by implementing adapters

---

## Support

If you need help:
- Review `MIGRATION_GUIDE.md` for detailed examples
- Check `README.md` for architecture overview
- See `ARCHITECTURE_PLAN.md` in AlphaEngine for full design

**You're ready to build!** Start with Step 1 (GitHub) and Step 2 (first module extraction).
