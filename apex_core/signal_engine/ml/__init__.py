# ml module
"""
Signal Engine ML module.

Core machine learning functionality for signal generation systems including:
- Model training and inference
- Performance metrics
- Cross-validation
- Hyperparameter tuning
- Backtesting
"""

from signal_engine.ml.metrics import (
    compute_accuracy_by_version,
    rolling_precision_recall_snapshot,
)
from signal_engine.ml.infer import (
    InferenceEngine,
    EnsembleInferenceEngine,
    vectorize_features,
    compute_feature_contributions,
    compute_volatility_adjusted_threshold,
)
from signal_engine.ml.backtest import (
    Trade,
    run_backtest,
)
from signal_engine.ml.regime_detector import (
    RegimeType,
    detect_market_regime,
    add_regime_feature,
    filter_by_regime,
    get_regime_stats,
)
from signal_engine.ml.cv_eval import (
    FoldStats,
    compute_fold_stats,
    compute_future_return,
    walk_forward_cv,
    time_series_split,
)
from signal_engine.ml.tuner import (
    extract_backtest_metrics,
    aggregate_metrics,
    objective_score,
    grid_search_thresholds,
)

__all__ = [
    # Metrics
    'compute_accuracy_by_version',
    'rolling_precision_recall_snapshot',
    # Inference
    'InferenceEngine',
    'EnsembleInferenceEngine',
    'vectorize_features',
    'compute_feature_contributions',
    'compute_volatility_adjusted_threshold',
    # Backtesting
    'Trade',
    'run_backtest',
    # Regime Detection
    'RegimeType',
    'detect_market_regime',
    'add_regime_feature',
    'filter_by_regime',
    'get_regime_stats',
    # Cross-Validation
    'FoldStats',
    'compute_fold_stats',
    'compute_future_return',
    'walk_forward_cv',
    'time_series_split',
    # Hyperparameter Tuning
    'extract_backtest_metrics',
    'aggregate_metrics',
    'objective_score',
    'grid_search_thresholds',
]
