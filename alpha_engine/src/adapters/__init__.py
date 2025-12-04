# AlphaEngine Adapters for signal-engine-core
"""
Adapters that implement signal-engine-core interfaces for AlphaEngine.

These adapters bridge AlphaEngine's specific implementations (Yahoo Finance,
model directory structure) with signal-engine-core's generic interfaces.

Usage:
    from src.adapters import AlphaEngineInferenceService

    service = AlphaEngineInferenceService()
    service.initialize_ensemble()
    result = service.predict(features)
"""

from src.adapters.price_adapter import YahooFinanceAdapter
from src.adapters.model_loader import AlphaEngineModelLoader
from src.adapters.inference_service import AlphaEngineInferenceService

__all__ = [
    'YahooFinanceAdapter',
    'AlphaEngineModelLoader',
    'AlphaEngineInferenceService',
]
