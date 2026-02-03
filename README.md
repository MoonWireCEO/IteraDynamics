# Itera Dynamics

![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-monorepo-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

> **Quantitative Trading Research & Execution Platform**

---

## Overview

**Itera Dynamics** is a quantitative trading platform built around a modular, asset-agnostic architecture. The system separates signal generation from execution, allowing the same core intelligence to power multiple market deployments.

### Current Focus: BTC Trading via Argus

The platform currently operates **Argus**, an hourly BTC trading system running against Coinbase. Future expansion to securities (stocks, ETFs) is architected but dormant.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       ITERA DYNAMICS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌─────────────────────────────────────────┐                  │
│    │       APEX CORTEX (apex_core/)          │                  │
│    │       The Brain - Signal Logic          │                  │
│    │  • ML inference & backtesting           │                  │
│    │  • Regime detection                     │                  │
│    │  • Governance & drift monitoring        │                  │
│    └──────────────┬──────────────────────────┘                  │
│                   │                                             │
│         ┌─────────┴─────────┐                                   │
│         ▼                   ▼                                   │
│    ┌─────────┐        ┌─────────────┐                           │
│    │  ARGUS  │        │ AlphaEngine │                           │
│    │  (BTC)  │        │ (Securities)│                           │
│    │ ACTIVE  │        │   DORMANT   │                           │
│    └─────────┘        └─────────────┘                           │
│                                                                 │
│    ┌─────────────────────────────────────────┐                  │
│    │            RESEARCH LAB                 │                  │
│    │    Strategy development & backtesting   │                  │
│    └─────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **Apex Cortex** | `apex_core/` | Asset-agnostic signal engine, ML, governance | Active |
| **Argus** | `runtime/argus/` | Live BTC trading scheduler & execution | Active |
| **Research** | `research/` | Strategy R&D, backtesting, experiments | Active |
| **AlphaEngine** | `alpha_engine/` | Securities platform (Yahoo Finance) | Dormant |
| **Dashboard** | `dashboard.py` | Streamlit mission control | Active |

---

## Project Structure

```
IteraDynamics_Mono/
│
├── apex_core/                    # 🧠 THE BRAIN - Asset-agnostic signal logic
│   ├── __init__.py               # Clean public API
│   ├── signal_engine/            # Organized ML/Analytics/Governance
│   │   ├── ml/                   # Backtesting, metrics, tuning
│   │   ├── analytics/            # Origin analysis, burst detection
│   │   ├── validation/           # Calibration, reliability
│   │   ├── governance/           # Model lifecycle, drift detection
│   │   └── threshold/            # Threshold optimization
│   ├── infer.py                  # ML inference functions
│   ├── regime_detector.py        # Market regime detection
│   └── ...                       # Additional modules
│
├── runtime/                      # 🦅 LIVE EXECUTION
│   └── argus/                    # BTC trading service
│       ├── run_live.py           # Hourly scheduler (main entry point)
│       ├── apex_core/            # Runtime-specific signal generators
│       └── models/               # Trained ML models
│
├── research/                     # 🔬 STRATEGY R&D
│   ├── strategies/               # Strategy implementations
│   │   ├── regime_trend.py       # Regime-based trend following
│   │   ├── guardian.py           # Capital preservation strategy
│   │   ├── sentinel.py           # Momentum trend following
│   │   └── rtr1.py               # RTR-1 strategy
│   ├── engine/                   # Backtesting engine
│   │   └── backtest_core.py      # Core backtest functionality
│   ├── experiments/              # One-off experiments
│   └── backtests/                # Results & artifacts
│
├── scripts/                      # 🛠️ UTILITIES
│   ├── training/                 # Model training scripts
│   ├── data/                     # Data download & preparation
│   ├── analysis/                 # Trade analysis tools
│   └── debug/                    # Debug & inspection utilities
│
├── alpha_engine/                 # 📈 SECURITIES (Dormant)
│   └── ...                       # Yahoo Finance integration (future)
│
├── data/                         # 📊 Data files
├── output/                       # 📁 Results & artifacts (gitignored)
│
├── dashboard.py                  # Mission Control (Streamlit)
├── pyproject.toml                # Build configuration
├── requirements.txt              # Dependencies
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/IteraDynamics/IteraDynamics.git
cd IteraDynamics_Mono
pip install -e .
```

### 2. Run Research Backtests

```bash
# Run the Regime Trend strategy backtest
cd research
python run_regime_trend.py
```

### 3. Run Live Trading (Argus)

Configure your Coinbase API credentials in `.env`:

```env
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret
COINBASE_PORTFOLIO_UUID=your_portfolio_uuid
```

Start the scheduler:

```bash
cd runtime/argus
python run_live.py
```

### 4. Launch Dashboard

```bash
python -m streamlit run dashboard.py
```

---

## Using the Library

### Apex Cortex Public API

```python
# Regime detection
from apex_core import MarketRegimeDetector
detector = MarketRegimeDetector()
regime = detector.detect_regime(price_df)

# ML inference
from apex_core import infer_score, infer_score_ensemble
result = infer_score(features)

# Signal engine tools
from apex_core.signal_engine import ml
from apex_core.signal_engine.ml import Trade, run_backtest
```

### Research Strategies

```python
from research.strategies.regime_trend import RegimeTrendParams, build_regime_signals
from research.engine.backtest_core import BacktestConfig, run_backtest_long_only

# Configure strategy
params = RegimeTrendParams(
    regime_sma=200,
    confirm_sma=50,
    entry_buffer_pct=3.0,
)

# Generate signals and backtest
signals = build_regime_signals(df, params)
result = run_backtest_long_only(df, signals, BacktestConfig())
```

---

## Research Strategies

| Strategy | Description | Status |
|----------|-------------|--------|
| **Regime Trend** | SMA-based trend following with regime filter | Tested |
| **Guardian** | Capital preservation + active trading | Tested |
| **Sentinel** | Momentum trend following with protection | Tested |
| **RTR-1** | Research trend strategy variant | In development |

See `research/STRATEGY_SUMMARY.md` for detailed performance analysis.

---

## Key Features

### Signal Generation
- **Regime Detection**: Volatility + trend-based market state classification
- **ML Ensemble**: Logistic regression, Random Forest, Gradient Boosting
- **Feature Engineering**: RSI, Bollinger Bands, volume z-scores, custom features

### Risk Management
- **Drift Detection**: Monitors feature importance decay and Sharpe degradation
- **Execution Gates**: Pre-trade checks for liquidity, spread, account health
- **Kill Switch**: Automated shadow mode if drawdown exceeds threshold

### Research Tools
- **Walk-Forward Validation**: Temporal splitting with embargo gaps
- **Monte Carlo Analysis**: Statistical significance testing
- **Trade Analysis**: R-multiples, equity curves, drawdown analysis

---

## Development

### Running Tests

```bash
# Test apex_core imports
python -c "from apex_core import MarketRegimeDetector; print('OK')"

# Test research imports
python -c "from research.strategies.regime_trend import RegimeTrendParams; print('OK')"
```

### Adding New Strategies

1. Create strategy in `research/strategies/your_strategy.py`
2. Create runner in `research/run_your_strategy.py`
3. Backtest and iterate
4. Graduate to `apex_core/` when production-ready

---

## License

MIT License - See `LICENSE` for details.

> **Disclaimer**: This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results.

---

## Acknowledgments

Built with Python, pandas, scikit-learn, and Streamlit.
