# src/adapters/inference_service.py
"""
AlphaEngine Inference Service.

High-level service that uses signal-engine-core's inference capabilities
with AlphaEngine-specific adapters. This provides a clean interface for
AlphaEngine to use core ML functionality.
"""

from signal_engine.ml import InferenceEngine, EnsembleInferenceEngine
from src.adapters.model_loader import AlphaEngineModelLoader
from src.adapters.price_adapter import YahooFinanceAdapter
from typing import Dict, Any, Optional
from pathlib import Path


class AlphaEngineInferenceService:
    """
    AlphaEngine-specific inference service using signal-engine-core.

    This service:
    - Loads models from AlphaEngine's directory structure
    - Uses signal-engine-core's InferenceEngine
    - Provides AlphaEngine-specific conveniences
    """

    def __init__(self, models_dir: Path = Path("models")):
        """
        Initialize inference service.

        Args:
            models_dir: Directory containing models (default: "models")
        """
        self.model_loader = AlphaEngineModelLoader(models_dir)
        self.price_adapter = YahooFinanceAdapter()
        self.engine: Optional[InferenceEngine] = None
        self.ensemble: Optional[EnsembleInferenceEngine] = None
        self._mode = None

    def initialize_single_model(self, model_type: str = "current"):
        """
        Initialize single model inference.

        Args:
            model_type: Model type to load (default: "current")

        Raises:
            FileNotFoundError: If model not found
            ValueError: If feature order missing
        """
        model, feature_order, metadata = self.model_loader.load_latest_model(model_type)

        self.engine = InferenceEngine(
            model=model,
            feature_order=feature_order,
            metadata=metadata
        )
        self._mode = "single"
        print(f"✅ Initialized single model: {model_type}")

    def initialize_ensemble(self):
        """
        Initialize ensemble of models.

        Raises:
            ValueError: If no models found
        """
        models_config = self.model_loader.load_ensemble_models()

        if not models_config:
            raise ValueError("No models found for ensemble")

        self.ensemble = EnsembleInferenceEngine(
            models=models_config,
            aggregation="mean"
        )
        self._mode = "ensemble"
        print(f"✅ Initialized ensemble with {len(models_config)} models: {list(models_config.keys())}")

    def predict(
        self,
        features: Dict[str, Any],
        use_ensemble: bool = True,
        explain: bool = False,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Make prediction using appropriate engine.

        Args:
            features: Feature dictionary
            use_ensemble: Whether to use ensemble if available (default: True)
            explain: Whether to include feature contributions (default: False)
            top_n: Number of top contributors to return if explain=True

        Returns:
            Prediction result dictionary with 'probability' key

        Raises:
            RuntimeError: If no engine initialized
        """
        if use_ensemble and self.ensemble is not None:
            result = self.ensemble.predict_proba(features)
        elif self.engine is not None:
            result = self.engine.predict_proba(features, explain=explain, top_n=top_n)
        else:
            raise RuntimeError(
                "No inference engine initialized. "
                "Call initialize_single_model() or initialize_ensemble() first."
            )

        return result

    def predict_for_asset(
        self,
        asset: str,
        features: Dict[str, Any],
        include_price: bool = True,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction for a specific asset with additional context.

        Args:
            asset: Asset ticker (e.g., "SPY")
            features: Feature dictionary
            include_price: Whether to include current price (default: True)
            explain: Whether to include feature contributions

        Returns:
            Dictionary with:
            - asset: Asset ticker
            - prediction: Prediction result from engine
            - current_price: Current price (if include_price=True)
        """
        prediction = self.predict(features, explain=explain)

        result = {
            "asset": asset,
            "prediction": prediction,
        }

        if include_price:
            current_price = self.price_adapter.get_price(asset)
            result["current_price"] = current_price

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model(s).

        Returns:
            Dictionary with model information
        """
        info = {
            "mode": self._mode,
            "version": self.model_loader.get_model_version(),
        }

        if self.engine and self.engine.metadata:
            info["metadata"] = self.engine.metadata

        if self.ensemble:
            info["models"] = list(self.ensemble.engines.keys())

        return info


__all__ = ['AlphaEngineInferenceService']
