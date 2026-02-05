# signal_engine/ml/tuner.py
"""
Hyperparameter Tuning Module.

Provides grid search functionality for optimizing trading strategy parameters
such as confidence thresholds, debounce periods, and holding horizons.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd


def extract_backtest_metrics(backtest_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize backtest results into a standard metrics dictionary.

    Handles varying output formats from different backtest implementations.

    Args:
        backtest_result: Dictionary from run_backtest() or similar function

    Returns:
        Normalized metrics dict with keys:
        - n_trades, win_rate, profit_factor, max_drawdown, signals_per_day
    """
    metrics = backtest_result.get("metrics", {})
    trades = backtest_result.get("trades", metrics.get("trades", []))

    # Extract n_trades
    if isinstance(metrics.get("n_trades"), int):
        n_trades = metrics["n_trades"]
    elif isinstance(trades, (list, tuple)):
        n_trades = len(trades)
    else:
        n_trades = 0

    # Extract or calculate wins/losses
    wins = metrics.get("wins", 0)
    losses = metrics.get("losses", 0)

    if wins + losses == 0 and isinstance(trades, (list, tuple)) and len(trades) > 0:
        for trade in trades:
            if isinstance(trade, dict):
                pnl = trade.get("pnl", trade.get("pnl_pct", 0.0))
            else:
                pnl = getattr(trade, "pnl", 0.0)

            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

    # Calculate win rate
    win_rate = metrics.get("win_rate", 0.0)
    if win_rate == 0.0 and (wins + losses) > 0:
        win_rate = wins / (wins + losses)

    return {
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "signals_per_day": float(metrics.get("signals_per_day", 0.0)),
    }


def aggregate_metrics(
    metrics_by_asset: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple assets.

    Weighting strategy:
    - win_rate: weighted by n_trades
    - profit_factor: weighted by n_trades
    - max_drawdown: worst (minimum) across assets
    - signals_per_day: sum across assets
    - n_trades: sum across assets

    Args:
        metrics_by_asset: Dict mapping asset names to their metrics

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_by_asset:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "signals_per_day": 0.0,
            "n_trades": 0,
        }

    total_trades = sum(m.get("n_trades", 0) for m in metrics_by_asset.values())

    # Weighted averages
    if total_trades > 0:
        win_rate = sum(
            m.get("win_rate", 0.0) * m.get("n_trades", 0)
            for m in metrics_by_asset.values()
        ) / total_trades

        profit_factor = sum(
            m.get("profit_factor", 0.0) * m.get("n_trades", 0)
            for m in metrics_by_asset.values()
        ) / total_trades
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Sum and min
    signals_per_day = sum(m.get("signals_per_day", 0.0) for m in metrics_by_asset.values())
    max_drawdown = min(m.get("max_drawdown", 0.0) for m in metrics_by_asset.values())

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "signals_per_day": float(signals_per_day),
        "n_trades": int(total_trades),
    }


def objective_score(
    metrics: Dict[str, Any],
    min_win_rate: float = 0.60,
    min_signals_per_day: float = 5.0,
    max_signals_per_day: float = 10.0,
) -> Tuple[int, float, float]:
    """
    Compute objective score for ranking parameter combinations.

    Returns a tuple that can be sorted (ascending = better):
    1. Feasibility flag (0 = feasible, 1 = infeasible)
    2. Negative profit factor (higher PF is better)
    3. Max drawdown (less negative is better)

    Args:
        metrics: Metrics dictionary to score
        min_win_rate: Minimum required win rate
        min_signals_per_day: Minimum signals per day
        max_signals_per_day: Maximum signals per day

    Returns:
        Tuple of (feasibility, -profit_factor, max_drawdown) for sorting
    """
    win_rate = metrics.get("win_rate", 0.0)
    signals_per_day = metrics.get("signals_per_day", 0.0)
    profit_factor = metrics.get("profit_factor", 0.0)
    max_drawdown = metrics.get("max_drawdown", 0.0)

    # Check feasibility constraints
    is_feasible = (
        win_rate >= min_win_rate and
        min_signals_per_day <= signals_per_day <= max_signals_per_day
    )

    feasibility_flag = 0 if is_feasible else 1

    return (feasibility_flag, -profit_factor, max_drawdown)


def grid_search_thresholds(
    predictions_by_asset: Dict[str, pd.DataFrame],
    prices_by_asset: Dict[str, pd.DataFrame],
    backtest_fn: Callable,
    confidence_grid: Optional[Iterable[float]] = None,
    debounce_grid: Optional[Iterable[int]] = None,
    horizon_grid: Optional[Iterable[int]] = None,
    fees_bps: float = 1.0,
    slippage_bps: float = 2.0,
    objective_fn: Optional[Callable[[Dict[str, Any]], Tuple]] = None,
) -> Dict[str, Any]:
    """
    Perform grid search over trading strategy parameters.

    Args:
        predictions_by_asset: Dict mapping asset names to prediction DataFrames
                             (must have 'ts' and 'p_long' columns)
        prices_by_asset: Dict mapping asset names to price DataFrames
                        (must have 'ts' and 'close' columns)
        backtest_fn: Function with signature (pred_df, prices_df, conf_min,
                     debounce_min, horizon_h, fees_bps, slippage_bps) -> dict
        confidence_grid: Confidence threshold values to try (default: [0.52-0.65])
        debounce_grid: Debounce minutes to try (default: [10-60])
        horizon_grid: Holding period hours to try (default: [1-3])
        fees_bps: Trading fees in basis points
        slippage_bps: Slippage in basis points
        objective_fn: Custom objective function for ranking (default: objective_score)

    Returns:
        Dictionary with:
        - params: Best parameter combination
        - metrics: Aggregated metrics for best params
        - per_asset: Per-asset metrics for best params
        - all_results: List of all tried combinations with their scores
    """
    # Default grids
    if confidence_grid is None:
        confidence_grid = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    if debounce_grid is None:
        debounce_grid = [10, 15, 20, 30, 45, 60]
    if horizon_grid is None:
        horizon_grid = [1, 2, 3]

    if objective_fn is None:
        objective_fn = objective_score

    # Normalize prediction DataFrames
    assets = sorted(predictions_by_asset.keys())
    predictions_normalized: Dict[str, pd.DataFrame] = {}

    for asset in assets:
        df = predictions_by_asset[asset].copy()

        # Ensure 'ts' column exists
        if "ts" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "ts"})
            else:
                df["ts"] = np.arange(len(df))

        # Ensure 'p_long' column exists
        if "p_long" not in df.columns:
            # Try common aliases
            for col in df.columns:
                if col.lower() in ("prob", "proba", "p", "plong", "probability"):
                    df = df.rename(columns={col: "p_long"})
                    break

        # Sort by time
        df = df.sort_values("ts").reset_index(drop=True)
        predictions_normalized[asset] = df[["ts", "p_long"]].copy()

    # Grid search
    all_results: List[Dict[str, Any]] = []

    for conf_min in confidence_grid:
        for debounce_min in debounce_grid:
            for horizon_h in horizon_grid:
                per_asset_metrics: Dict[str, Dict[str, Any]] = {}

                # Run backtest for each asset
                for asset in assets:
                    pred_df = predictions_normalized.get(asset)
                    price_df = prices_by_asset.get(asset)

                    if pred_df is None or price_df is None or pred_df.empty or price_df.empty:
                        per_asset_metrics[asset] = {
                            "n_trades": 0,
                            "win_rate": 0.0,
                            "profit_factor": 0.0,
                            "max_drawdown": 0.0,
                            "signals_per_day": 0.0,
                        }
                        continue

                    # Run backtest
                    backtest_result = backtest_fn(
                        pred_df=pred_df,
                        prices_df=price_df,
                        conf_min=float(conf_min),
                        debounce_min=int(debounce_min),
                        horizon_h=int(horizon_h),
                        fees_bps=float(fees_bps),
                        slippage_bps=float(slippage_bps),
                    )

                    per_asset_metrics[asset] = extract_backtest_metrics(backtest_result)

                # Aggregate across assets
                aggregated = aggregate_metrics(per_asset_metrics)

                # Store result
                params = {
                    "conf_min": float(conf_min),
                    "debounce_min": int(debounce_min),
                    "horizon_h": int(horizon_h),
                }

                all_results.append({
                    "params": params,
                    "metrics": aggregated,
                    "per_asset": per_asset_metrics,
                    "score": objective_fn(aggregated),
                })

    # Select best result
    if not all_results:
        return {
            "params": {"conf_min": 0.55, "debounce_min": 15, "horizon_h": 1},
            "metrics": {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "signals_per_day": 0.0,
                "n_trades": 0,
            },
            "per_asset": {},
            "all_results": [],
        }

    # Sort by objective score (ascending = better)
    all_results.sort(key=lambda x: x["score"])
    best = all_results[0]

    return {
        "params": best["params"],
        "metrics": best["metrics"],
        "per_asset": best["per_asset"],
        "all_results": all_results,
    }


__all__ = [
    'extract_backtest_metrics',
    'aggregate_metrics',
    'objective_score',
    'grid_search_thresholds',
]
