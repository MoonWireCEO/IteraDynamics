"""
Signal Engine Core

Shared ML/Governance/Validation engine for multi-asset signal generation platforms.

Modules:
- ml: Machine learning (inference, metrics, backtest, tuning, regime detection, cross-validation)
- validation: Model validation and feedback reliability
- threshold: Threshold optimization and simulation
- analytics: Data processing and origin utilities
- interfaces: Abstract interfaces for data providers
- governance: Model governance and lifecycle management
- performance: Performance monitoring
- core: Core utilities
"""

__version__ = "1.0.0"
__author__ = "Signal Engine Team"

from signal_engine.interfaces import (
    PriceProvider,
    SentimentProvider,
    MarketDataProvider,
)

# Import submodules for easier access
from signal_engine import ml
from signal_engine import validation
from signal_engine import threshold
from signal_engine import analytics
from signal_engine import interfaces

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Interfaces
    "PriceProvider",
    "SentimentProvider",
    "MarketDataProvider",
    # Submodules
    "ml",
    "validation",
    "threshold",
    "analytics",
    "interfaces",
]
