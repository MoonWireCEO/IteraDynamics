#!/usr/bin/env bash
# Approach 3: Simplified LR-only Model
# Use only Logistic Regression with top 5 features to reduce overfitting

set -e

echo "=================================================="
echo "ML Improvement Approach 3: Simplified LR-only Model"
echo "=================================================="
echo ""
echo "Strategy: Reduce model complexity"
echo "- Use only Logistic Regression (drop RF + GB)"
echo "- Select top 5 most important features"
echo "- Strong regularization (C=0.1)"
echo "- Train on 365 days for stability"
echo ""
echo "Expected impact: Reduce overfitting, improve consistency"
echo "Target: 52%+ WR, 1.15+ PF with low fold variance"
echo ""

# Set environment variables for this approach
export AE_ML_LOOKBACK_DAYS=365
export AE_ML_MODEL=logreg
export AE_FEATURE_SELECTION=top5
export AE_ML_SYMBOLS=SPY,QQQ

# Disable regime features for clean comparison
export AE_REGIME_ENABLED=0
export AE_REGIME_FILTER_ENABLED=0

echo "Configuration:"
echo "- Training data: 365 days"
echo "- Model type: logreg ONLY (no ensemble)"
echo "- Features: TOP 5 (r_6h, vol_6h, social_score, atr_14h, sma_gap)"
echo "- Regularization: Strong (C=0.1)"
echo "- Regime filtering: DISABLED"
echo ""

# Run training
python -m scripts.ml.train_predict

echo ""
echo "✓ Training complete!"
echo "Check artifacts/ml_* and models/ml_model_manifest.json for results"
echo ""
echo "Key metric: Check fold variance (should be <7% for consistency)"
