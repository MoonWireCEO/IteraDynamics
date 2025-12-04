# signal_engine/ml/infer.py
"""
Signal Engine Inference Module.

Core inference functionality for ML model scoring and prediction.
This module provides product-agnostic inference capabilities that can be
used across different signal generation systems.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np


# ---------- Core inference utilities ----------

def vectorize_features(
    features: Dict[str, Any],
    feature_order: List[str]
) -> np.ndarray:
    """
    Convert a feature dictionary to a numpy array based on a specified feature order.

    Args:
        features: Dictionary mapping feature names to values
        feature_order: List of feature names in the order expected by the model

    Returns:
        numpy array of shape (1, n_features) with feature values
    """
    return np.array(
        [[float(features.get(k, 0.0) or 0.0) for k in feature_order]],
        dtype=float
    )


def compute_feature_contributions(
    model: Any,
    feature_vector: np.ndarray,
    feature_order: List[str],
    top_n: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute feature contributions for a linear model.

    Args:
        model: Trained model with coef_ attribute (e.g., LogisticRegression)
        feature_vector: numpy array of shape (1, n_features)
        feature_order: List of feature names corresponding to vector positions
        top_n: If specified, return only the top N contributors by absolute value

    Returns:
        Dictionary mapping feature names to their contributions
    """
    try:
        coef = model.coef_.ravel()
        contributions = {
            feature_order[i]: float(coef[i] * feature_vector[0, i])
            for i in range(len(feature_order))
        }
        if top_n is not None:
            sorted_items = sorted(
                contributions.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True
            )
            return dict(sorted_items[:top_n])
        return contributions
    except Exception:
        return {}


class InferenceEngine:
    """
    Core inference engine for ML model scoring.

    This class provides a product-agnostic interface for running inference
    with trained models. It handles feature vectorization, prediction, and
    optional explainability.
    """

    def __init__(
        self,
        model: Any,
        feature_order: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained model with predict_proba or predict method
            feature_order: List of feature names in order expected by model
            metadata: Optional metadata about the model (version, training date, etc.)
        """
        self.model = model
        self.feature_order = feature_order
        self.metadata = metadata or {}

    def predict_proba(
        self,
        features: Dict[str, Any],
        explain: bool = False,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Predict probability and optionally explain contributions.

        Args:
            features: Dictionary of feature values
            explain: Whether to include feature contributions in output
            top_n: Number of top contributors to include if explain=True

        Returns:
            Dictionary containing:
            - probability: float, predicted probability
            - contributions: dict (if explain=True), feature contributions
        """
        try:
            x = vectorize_features(features, self.feature_order)

            # Get probability
            if hasattr(self.model, 'predict_proba'):
                proba = float(self.model.predict_proba(x)[0, 1])
            elif hasattr(self.model, 'decision_function'):
                # For models with decision_function (SVM, etc.)
                z = float(self.model.decision_function(x)[0])
                proba = 1.0 / (1.0 + np.exp(-z))
            else:
                raise ValueError("Model must have predict_proba or decision_function")

            result: Dict[str, Any] = {"probability": proba}

            # Add contributions if requested
            if explain:
                contributions = compute_feature_contributions(
                    self.model, x, self.feature_order, top_n=top_n
                )
                result["contributions"] = contributions

            return result

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    def predict(
        self,
        features: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Make binary prediction based on threshold.

        Args:
            features: Dictionary of feature values
            threshold: Decision threshold (default: 0.5)

        Returns:
            Dictionary containing:
            - prediction: int (0 or 1)
            - probability: float
        """
        result = self.predict_proba(features, explain=False)
        proba = result["probability"]
        prediction = 1 if proba >= threshold else 0

        return {
            "prediction": prediction,
            "probability": proba
        }


class EnsembleInferenceEngine:
    """
    Ensemble inference engine that combines predictions from multiple models.

    This class allows voting/averaging across different model types
    to produce more robust predictions.
    """

    def __init__(
        self,
        models: Dict[str, Dict[str, Any]],
        aggregation: str = "mean"
    ):
        """
        Initialize ensemble inference engine.

        Args:
            models: Dictionary mapping model names to dicts containing:
                - 'model': trained model
                - 'feature_order': list of feature names
                - 'metadata': optional metadata dict
            aggregation: How to combine predictions ('mean', 'median', 'max', 'min')
        """
        self.engines: Dict[str, InferenceEngine] = {}
        for name, config in models.items():
            self.engines[name] = InferenceEngine(
                model=config['model'],
                feature_order=config['feature_order'],
                metadata=config.get('metadata')
            )
        self.aggregation = aggregation

    def predict_proba(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict using ensemble of models.

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary containing:
            - probability: float, aggregated probability
            - votes: dict, individual model probabilities
            - models: list, names of models that contributed
            - low: float, minimum probability across models
            - high: float, maximum probability across models
        """
        votes: Dict[str, float] = {}

        # Get predictions from each model
        for name, engine in self.engines.items():
            try:
                result = engine.predict_proba(features, explain=False)
                votes[name] = result["probability"]
            except Exception:
                # Skip models that fail
                continue

        if not votes:
            raise RuntimeError("No models in ensemble produced valid predictions")

        # Aggregate predictions
        probs = list(votes.values())
        if self.aggregation == "mean":
            aggregated = float(np.mean(probs))
        elif self.aggregation == "median":
            aggregated = float(np.median(probs))
        elif self.aggregation == "max":
            aggregated = float(np.max(probs))
        elif self.aggregation == "min":
            aggregated = float(np.min(probs))
        else:
            aggregated = float(np.mean(probs))  # default to mean

        return {
            "probability": aggregated,
            "votes": votes,
            "models": list(votes.keys()),
            "low": float(min(probs)),
            "high": float(max(probs)),
        }

    def predict(
        self,
        features: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Make binary prediction using ensemble.

        Args:
            features: Dictionary of feature values
            threshold: Decision threshold (default: 0.5)

        Returns:
            Dictionary containing prediction and probability info
        """
        result = self.predict_proba(features)
        proba = result["probability"]
        prediction = 1 if proba >= threshold else 0

        return {
            **result,
            "prediction": prediction,
        }


def compute_volatility_adjusted_threshold(
    base_threshold: float,
    regime: str,
    regime_multipliers: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Adjust decision threshold based on market regime.

    This function allows dynamic threshold adjustment based on market conditions,
    which can help reduce false positives during turbulent periods and increase
    sensitivity during calm periods.

    Args:
        base_threshold: Base decision threshold (e.g., 0.5)
        regime: Market regime label (e.g., "calm", "normal", "turbulent")
        regime_multipliers: Optional custom multipliers for each regime.
            Defaults to {"calm": 0.9, "normal": 1.0, "turbulent": 1.1}

    Returns:
        Dictionary containing:
        - base_threshold: original threshold
        - volatility_regime: regime label
        - regime_multiplier: multiplier applied
        - threshold_after_volatility: adjusted threshold
    """
    if regime_multipliers is None:
        regime_multipliers = {
            "calm": 0.9,
            "normal": 1.0,
            "turbulent": 1.1,
        }

    regime_normalized = str(regime).strip().lower()
    multiplier = regime_multipliers.get(regime_normalized, 1.0)
    adjusted = base_threshold * multiplier

    return {
        "base_threshold": base_threshold,
        "volatility_regime": regime,
        "regime_multiplier": multiplier,
        "threshold_after_volatility": adjusted,
    }


__all__ = [
    'InferenceEngine',
    'EnsembleInferenceEngine',
    'vectorize_features',
    'compute_feature_contributions',
    'compute_volatility_adjusted_threshold',
]
