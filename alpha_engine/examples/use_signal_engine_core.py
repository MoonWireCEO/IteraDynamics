#!/usr/bin/env python
"""
Example: Using signal-engine-core in AlphaEngine

This example demonstrates how to use the new adapters and signal-engine-core
modules for inference in AlphaEngine.
"""

from src.adapters import AlphaEngineInferenceService
from pathlib import Path
import sys


def main():
    """
    Demonstrates three ways to use signal-engine-core in AlphaEngine:
    1. Single model inference
    2. Ensemble inference
    3. Asset-specific prediction with price data
    """

    print("=" * 60)
    print("AlphaEngine + signal-engine-core Integration Example")
    print("=" * 60)

    # Initialize the inference service
    service = AlphaEngineInferenceService(models_dir=Path("models"))

    # Option 1: Try ensemble first (recommended)
    print("\n🔍 Attempting to load ensemble models...")
    try:
        service.initialize_ensemble()
        mode = "ensemble"
    except (ValueError, FileNotFoundError) as e:
        print(f"⚠️  Ensemble not available: {e}")
        print("🔍 Falling back to single model...")
        try:
            service.initialize_single_model("current")
            mode = "single"
        except (FileNotFoundError, ValueError) as e2:
            print(f"❌ No models available: {e2}")
            print("\nTo use this example:")
            print("1. Train a model using AlphaEngine's training pipeline")
            print("2. Ensure model files exist in models/ directory")
            print("3. Run this script again")
            return 1

    # Show model info
    print("\n📊 Model Information:")
    model_info = service.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")

    # Example features (you would get these from your feature builder)
    print("\n🎯 Making predictions...")
    example_features = {
        "burst_z": 2.5,
        "momentum_7d": 0.15,
        "volatility_14d": 0.08,
        "rsi_14": 65.0,
        "volume_spike": 1.2,
        "social_sentiment": 0.6,
        # Add more features as needed
    }

    # Make prediction
    try:
        result = service.predict(
            features=example_features,
            use_ensemble=(mode == "ensemble"),
            explain=True
        )

        print(f"\n✅ Prediction successful!")
        print(f"   Probability: {result['probability']:.4f}")

        if "votes" in result:
            print(f"\n   Individual Model Votes:")
            for model_name, prob in result["votes"].items():
                print(f"      {model_name}: {prob:.4f}")
            print(f"   Range: {result['low']:.4f} - {result['high']:.4f}")

        if "contributions" in result:
            print(f"\n   Top Feature Contributors:")
            for feature, contribution in list(result["contributions"].items())[:5]:
                print(f"      {feature}: {contribution:+.4f}")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return 1

    # Example with asset-specific prediction
    print(f"\n" + "=" * 60)
    print("Asset-Specific Prediction Example")
    print("=" * 60)

    try:
        asset_result = service.predict_for_asset(
            asset="SPY",
            features=example_features,
            include_price=True,
            explain=False
        )

        print(f"\n✅ SPY Prediction:")
        print(f"   Probability: {asset_result['prediction']['probability']:.4f}")
        if asset_result.get('current_price'):
            print(f"   Current Price: ${asset_result['current_price']:.2f}")

    except Exception as e:
        print(f"⚠️  Asset prediction failed (may need internet): {e}")

    print(f"\n" + "=" * 60)
    print("✅ Integration example complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
