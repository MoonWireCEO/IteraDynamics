# src/adapters/model_loader.py
"""
AlphaEngine Model Loader Adapter.

Handles loading models from AlphaEngine's specific directory structure
and prepares them for use with signal-engine-core.
"""

import joblib
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional


class AlphaEngineModelLoader:
    """
    Loads models from AlphaEngine's models/ directory structure.

    This adapter understands AlphaEngine's specific file layout and
    provides models in a format compatible with signal-engine-core.
    """

    def __init__(self, models_dir: Path = Path("models")):
        """
        Initialize model loader.

        Args:
            models_dir: Root models directory (default: "models")
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

        Raises:
            FileNotFoundError: If model not found
            ValueError: If feature order is missing
        """
        model_dir = self.models_dir / model_type

        # Load model
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            # Try alternative naming
            model_path = self.models_dir / f"{model_type}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        # Load metadata
        metadata = {}
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # Extract feature order from multiple possible sources
        feature_order = self._extract_feature_order(model_dir, metadata)

        if not feature_order:
            raise ValueError(f"Could not determine feature order for model at {model_dir}")

        return model, feature_order, metadata

    def _extract_feature_order(
        self,
        model_dir: Path,
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Extract feature order from various possible sources.

        Args:
            model_dir: Directory containing model files
            metadata: Loaded metadata dictionary

        Returns:
            List of feature names in order
        """
        # Try metadata first
        feature_order = metadata.get("feature_order") or metadata.get("features")

        # Try separate features file
        if not feature_order:
            features_path = model_dir / "features.json"
            if features_path.exists():
                with open(features_path, 'r', encoding='utf-8') as f:
                    features_data = json.load(f)
                    feature_order = (
                        features_data.get("feature_order") or
                        features_data.get("features")
                    )

        # Try meta.json file (alternative naming)
        if not feature_order:
            meta_path = model_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    feature_order = meta.get("feature_order") or meta.get("features")

        return feature_order or []

    def load_ensemble_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all available models for ensemble inference.

        Returns:
            Dictionary mapping model names to their configs:
            {
                'model_name': {
                    'model': trained_model,
                    'feature_order': list_of_features,
                    'metadata': metadata_dict
                }
            }
        """
        models = {}

        # Try loading standard ensemble models
        model_names = ["logistic", "rf", "gb"]

        for model_name in model_names:
            try:
                # Try different naming patterns
                model_path = self.models_dir / f"trigger_likelihood_{model_name}.joblib"
                meta_path = self.models_dir / f"trigger_likelihood_{model_name}.meta.json"

                if not model_path.exists():
                    model_path = self.models_dir / f"{model_name}.joblib"
                    meta_path = self.models_dir / f"{model_name}.meta.json"

                if model_path.exists():
                    model = joblib.load(model_path)

                    # Load metadata
                    metadata = {}
                    if meta_path.exists():
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                    feature_order = metadata.get("feature_order", [])

                    models[model_name] = {
                        "model": model,
                        "feature_order": feature_order,
                        "metadata": metadata,
                    }
            except Exception as e:
                print(f"Warning: Could not load {model_name} model: {e}")
                continue

        return models

    def get_model_version(self) -> str:
        """
        Get the current model version.

        Returns:
            Version string or "unknown"
        """
        version_file = self.models_dir / "training_version.txt"
        if version_file.exists():
            return version_file.read_text(encoding='utf-8').strip()
        return "unknown"


__all__ = ['AlphaEngineModelLoader']
