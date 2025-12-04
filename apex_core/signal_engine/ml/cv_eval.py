# signal_engine/ml/cv_eval.py
"""
Cross-Validation Module for ML Models.

Provides walk-forward cross-validation for time-series data with
trading strategy evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class FoldStats:
    """
    Statistics for a single cross-validation fold.

    Attributes:
        trades: Number of trades executed in this fold
        win_rate: Percentage of winning trades (None if no trades)
        profit_factor: Ratio of gross profits to gross losses (None if no losses)
    """
    trades: int
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None


def compute_fold_stats(signed_returns: np.ndarray) -> FoldStats:
    """
    Compute trading statistics from signed returns.

    Args:
        signed_returns: Array of realized returns where direction is already applied
                       (positive = profit, negative = loss)

    Returns:
        FoldStats with trades count, win rate, and profit factor
    """
    sr = np.array([x for x in signed_returns if np.isfinite(x)], dtype=float)

    if sr.size == 0:
        return FoldStats(trades=0, win_rate=None, profit_factor=None)

    wins = sr[sr > 0.0]
    losses = sr[sr < 0.0]

    win_rate = float((sr > 0.0).mean()) if sr.size else None

    profit_factor = None
    if losses.size > 0:
        profit_factor = float(wins.sum() / abs(losses.sum())) if wins.size > 0 else 0.0
    elif wins.size > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return FoldStats(trades=int(sr.size), win_rate=win_rate, profit_factor=profit_factor)


def compute_future_return(close: pd.Series, horizon: int) -> pd.Series:
    """
    Compute forward return over the next N periods.

    Args:
        close: Series of closing prices
        horizon: Number of periods to look ahead

    Returns:
        Series of forward returns: (close[t+horizon] / close[t] - 1)
    """
    return (close.shift(-horizon) / close - 1.0).astype(float)


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    future_returns: np.ndarray,
    train_fn: Callable[[np.ndarray, np.ndarray], Any],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
    n_splits: int = 5,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Perform walk-forward cross-validation for time-series data.

    This function splits the data into n_splits sequential chunks, trains on
    earlier data and tests on later data (maintaining temporal order).

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        future_returns: Actual future returns for each sample (n_samples,)
        train_fn: Function that takes (X_train, y_train) and returns trained model
        predict_fn: Function that takes (model, X_test) and returns probabilities
        n_splits: Number of walk-forward splits (default: 5)
        train_size: Number of samples in each training set (default: auto)
        test_size: Number of samples in each test set (default: auto)
        confidence_threshold: Minimum confidence to take a trade (default: 0.5)

    Returns:
        Dictionary containing:
        - per_fold: List of per-fold results
        - aggregate: Aggregated statistics across all folds
    """
    n_samples = len(X)

    # Calculate split sizes if not provided
    if train_size is None:
        train_size = n_samples // (n_splits + 1)
    if test_size is None:
        test_size = max(n_samples // (n_splits * 2), 1)

    per_fold: List[Dict[str, Any]] = []
    fold_id = 0

    # Walk-forward splits
    for i in range(n_splits):
        # Calculate indices for this fold
        train_start = i * test_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size

        # Ensure we don't go past the end
        if test_end > n_samples:
            break
        if train_end - train_start < 20 or test_end - test_start < 5:
            continue

        fold_id += 1

        # Get train/test splits
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        returns_test = future_returns[test_start:test_end]

        # Train and predict
        model = train_fn(X_train, y_train)
        probabilities = predict_fn(model, X_test)

        # Trading logic:
        # - Take trade if confidence >= threshold
        # - Direction: long if p >= 0.5, short otherwise
        # - Confidence: max(p, 1-p)
        direction_long = (probabilities >= 0.5)
        confidence = np.where(direction_long, probabilities, 1.0 - probabilities)
        take_trade = (confidence >= confidence_threshold)

        # Realized signed returns for trades taken
        signed_returns = np.where(direction_long, returns_test, -returns_test)
        signed_returns_taken = signed_returns[take_trade]

        # Compute fold statistics
        stats = compute_fold_stats(signed_returns_taken)

        per_fold.append({
            "fold": fold_id,
            "train_size": int(train_end - train_start),
            "test_size": int(test_end - test_start),
            "trades": stats.trades,
            "win_rate": stats.win_rate,
            "profit_factor": stats.profit_factor,
        })

    # Aggregate across folds
    trades = [f["trades"] for f in per_fold]
    win_rates = [f["win_rate"] for f in per_fold if f["win_rate"] is not None]
    profit_factors = [f["profit_factor"] for f in per_fold if f["profit_factor"] is not None]

    aggregate = {
        "n_folds": len(per_fold),
        "total_trades": int(sum(trades)),
        "win_rate_mean": float(np.mean(win_rates)) if win_rates else None,
        "win_rate_std": float(np.std(win_rates, ddof=1)) if len(win_rates) > 1 else None,
        "profit_factor_mean": float(np.mean(profit_factors)) if profit_factors else None,
        "profit_factor_std": float(np.std(profit_factors, ddof=1)) if len(profit_factors) > 1 else None,
    }

    return {
        "per_fold": per_fold,
        "aggregate": aggregate,
    }


def time_series_split(
    data: pd.DataFrame,
    n_splits: int = 5,
    train_days: Optional[int] = None,
    test_days: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward time-series splits.

    Args:
        data: DataFrame with datetime index or 'ts' column
        n_splits: Number of splits to generate
        train_days: Days in training set (default: auto-calculated)
        test_days: Days in test set (default: auto-calculated)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    n = len(data)

    # Auto-calculate split sizes if not provided
    if train_days is None:
        train_days = n // (n_splits + 1)
    if test_days is None:
        test_days = max(n // (n_splits * 2), 1)

    splits = []
    for i in range(n_splits):
        train_start = i * test_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days

        if test_end > n:
            break

        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)

        splits.append((train_indices, test_indices))

    return splits


__all__ = [
    'FoldStats',
    'compute_fold_stats',
    'compute_future_return',
    'walk_forward_cv',
    'time_series_split',
]
