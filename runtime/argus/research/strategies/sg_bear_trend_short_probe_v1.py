"""
Bear Trend Short Probe v1 (Research-Only)
========================================

Purpose
-------
Provide a simple, conservative short-side probe that activates only in clearly
hostile regimes. Designed to be paired with long sleeves like VB, not replace them.

Logic (high level)
------------------
- Only consider shorts when macro / regime is bearish
- Require downside confirmation (price below recent support)
- Use faster invalidation than long sleeve (cover on strength)

Notes
-----
- Research-only: intended for backtest_runner_directional.py
- No persistence; uses only closed-bar data
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from research.regime.regime_engine import classify_regime


LOOKBACK = 48
CONF = 0.6


def _resolve_close(df: pd.DataFrame) -> pd.Series:
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    if "close" in df.columns:
        return pd.to_numeric(df["close"], errors="coerce")
    return pd.Series(dtype=float)


def generate_intent(
    df: pd.DataFrame,
    ctx: Any = None,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    close = _resolve_close(df)
    close = close.dropna()
    if len(close) < LOOKBACK + 5:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "desired_exposure_frac": 0.0,
            "horizon_hours": LOOKBACK,
            "reason": "insufficient_history",
            "meta": {},
        }

    regime = classify_regime(df, closed_only=closed_only)
    regime_label = str(regime.label)

    last = float(close.iloc[-1])
    recent_low = float(close.iloc[-LOOKBACK:].min())
    recent_high = float(close.iloc[-LOOKBACK:].max())

    # Bear regime gate
    is_bear = regime_label in ("TREND_DOWN", "VOL_EXPANSION", "PANIC")

    # Entry: breakdown in bear regime
    if is_bear and last <= recent_low:
        return {
            "action": "ENTER_SHORT",
            "confidence": CONF,
            "desired_exposure_frac": -0.5,  # conservative sizing
            "horizon_hours": LOOKBACK,
            "reason": f"bear_breakdown ({regime_label})",
            "meta": {
                "regime": regime_label,
                "recent_low": recent_low,
            },
        }

    # Exit: strength against position
    if last >= recent_high:
        return {
            "action": "EXIT_SHORT",
            "confidence": CONF,
            "desired_exposure_frac": 0.0,
            "horizon_hours": LOOKBACK,
            "reason": "short_invalidated_by_strength",
            "meta": {
                "recent_high": recent_high,
            },
        }

    return {
        "action": "HOLD",
        "confidence": 0.2,
        "desired_exposure_frac": 0.0,
        "horizon_hours": LOOKBACK,
        "reason": "no_signal",
        "meta": {
            "regime": regime_label,
        },
    }
