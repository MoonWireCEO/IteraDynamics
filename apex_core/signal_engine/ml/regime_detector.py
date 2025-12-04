# signal_engine/ml/regime_detector.py
"""
Market regime detection for ML model performance optimization.

Detects whether market conditions are favorable for model predictions:
- trending: Strong directional move with moderate volatility (model works well)
- choppy: Low volatility or whipsaw action (model fails)

This module is product-agnostic and works with any price DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Dict, Any

RegimeType = Literal["trending", "choppy"]


def detect_market_regime(
    prices_df: pd.DataFrame,
    symbol: str = "",
    volatility_threshold: float = 0.06,
    trend_strength_threshold: float = 0.08,
    lookback_volatility: int = 14,
    lookback_trend: int = 30,
) -> pd.Series:
    """
    Detect if market is trending (model works) vs choppy (model fails).

    Args:
        prices_df: DataFrame with 'close' column and datetime index
        symbol: Symbol name for logging (not used in calculation)
        volatility_threshold: Max relative volatility for trending regime (default: 0.06 = 6%)
        trend_strength_threshold: Min absolute price change over lookback period (default: 0.08 = 8%)
        lookback_volatility: Days/periods to calculate rolling volatility (default: 14)
        lookback_trend: Days/periods to measure trend strength (default: 30)

    Returns:
        Series with same index as prices_df, values are 'trending' or 'choppy'

    Logic:
        - volatility = price.rolling(14).std() / price.mean()
        - trend_strength = abs(price.pct_change(30))
        - If volatility < 0.06 AND trend_strength > 0.08: 'trending'
        - Else: 'choppy'

    Rationale:
        Models work best when there's clear directional movement without excessive noise.
        Low volatility + strong trend = predictable moves = good model performance.
        High volatility OR no clear trend = choppy = bad model performance.
    """
    if "close" not in prices_df.columns:
        raise ValueError("prices_df must have 'close' column")

    close = prices_df["close"].astype(float)

    # Calculate rolling volatility as fraction of mean price
    rolling_std = close.rolling(
        lookback_volatility,
        min_periods=max(3, lookback_volatility // 2)
    ).std()
    mean_price = close.rolling(
        lookback_volatility,
        min_periods=max(3, lookback_volatility // 2)
    ).mean()
    volatility = (rolling_std / mean_price.replace(0, np.nan)).fillna(0)

    # Calculate trend strength as absolute price change percentage
    trend_strength = close.pct_change(lookback_trend).abs().fillna(0)

    # Regime logic: trending if low volatility AND strong trend
    is_trending = (volatility < volatility_threshold) & (trend_strength > trend_strength_threshold)

    regime = pd.Series("choppy", index=prices_df.index, dtype=str)
    regime[is_trending] = "trending"

    return regime


def add_regime_feature(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    symbol: str = "",
    **regime_kwargs
) -> pd.DataFrame:
    """
    Add regime as a binary feature column to feature DataFrame.

    Args:
        df: Feature DataFrame to augment
        prices_df: Price data with 'close' column
        symbol: Symbol name for logging
        **regime_kwargs: Additional arguments passed to detect_market_regime()

    Returns:
        Copy of df with added 'regime_trending' column (1=trending, 0=choppy)
    """
    df = df.copy()

    regime = detect_market_regime(prices_df, symbol=symbol, **regime_kwargs)

    # Align regime with feature df index
    regime_aligned = regime.reindex(df.index, fill_value="choppy")

    # Add as binary feature
    df["regime_trending"] = (regime_aligned == "trending").astype(float)

    return df


def filter_by_regime(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    symbol: str = "",
    keep_regime: RegimeType = "trending",
    **regime_kwargs
) -> pd.DataFrame:
    """
    Filter feature DataFrame to only include rows matching specified regime.

    Args:
        df: Feature DataFrame to filter
        prices_df: Price data with 'close' column
        symbol: Symbol name for logging
        keep_regime: Which regime to keep ('trending' or 'choppy', default: 'trending')
        **regime_kwargs: Additional arguments passed to detect_market_regime()

    Returns:
        Filtered DataFrame containing only rows in specified regime
    """
    regime = detect_market_regime(prices_df, symbol=symbol, **regime_kwargs)

    # Align regime with feature df index
    regime_aligned = regime.reindex(df.index, fill_value="choppy")

    # Filter
    mask = regime_aligned == keep_regime
    return df[mask].copy()


def get_regime_stats(
    prices_df: pd.DataFrame,
    symbol: str = "",
    **regime_kwargs
) -> Dict[str, Any]:
    """
    Get statistics about regime distribution in price data.

    Args:
        prices_df: Price data with 'close' column
        symbol: Symbol name for logging
        **regime_kwargs: Additional arguments passed to detect_market_regime()

    Returns:
        Dictionary with keys:
        - symbol: str
        - total_periods: int
        - trending_periods: int
        - choppy_periods: int
        - trending_pct: float (0-100)
        - choppy_pct: float (0-100)
    """
    regime = detect_market_regime(prices_df, symbol=symbol, **regime_kwargs)

    total = len(regime)
    trending = (regime == "trending").sum()
    choppy = (regime == "choppy").sum()

    return {
        "symbol": symbol,
        "total_periods": int(total),
        "trending_periods": int(trending),
        "choppy_periods": int(choppy),
        "trending_pct": round(100 * trending / total, 2) if total > 0 else 0.0,
        "choppy_pct": round(100 * choppy / total, 2) if total > 0 else 0.0,
    }


__all__ = [
    'RegimeType',
    'detect_market_regime',
    'add_regime_feature',
    'filter_by_regime',
    'get_regime_stats',
]
