"""
sg_mean_reversion_a.py

Sleeve 2 Candidate A: RSI Mean Reversion
Itera Dynamics - Research Phase

Mandate:
- Monetize CHOP and VOL_COMPRESSION regimes
- RSI-based oversold/overbought mean reversion
- Return density engine (frequent small wins)

Design:
- Entry: RSI(14) < 30 in CHOP/VOL_COMPRESSION
- Exit: RSI(14) > 60 OR regime flip
- Horizon: 12 hours
- Exposure: 0.20-0.30 (conservative)

Layer 2 contract compliance:
- Calls classify_regime() from Layer 1
- Returns StrategyIntent dict
- Deterministic, closed_only=True
- No side effects
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from research.regime import classify_regime


# Environment variables
ENV_RSI_LEN = "SG_MR_A_RSI_LEN"                    # default 14
ENV_RSI_OVERSOLD = "SG_MR_A_RSI_OVERSOLD"          # default 30
ENV_RSI_EXIT = "SG_MR_A_RSI_EXIT"                  # default 60
ENV_HORIZON_HOURS = "SG_MR_A_HORIZON_HOURS"        # default 12
ENV_EXPOSURE_FRAC = "SG_MR_A_EXPOSURE_FRAC"        # default 0.25


def _get_env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)).strip())
        return v if v > 0 else default
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, str(default)).strip())
        if "FRAC" in name.upper() or "EXPO" in name.upper():
            return float(min(1.0, max(0.0, v)))
        return float(v)
    except Exception:
        return default


def _safe_copy_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=False)


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic OHLCV normalization (same as Core/Vol Probe)"""
    if df is None or not hasattr(df, "columns"):
        return df

    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    def pick(*names: str):
        for n in names:
            if n in lower_map:
                return lower_map[n]
        return None

    src_close = pick("close", "c", "price", "last", "mid")
    if src_close is None:
        return df

    out = df.copy(deep=False)
    out["close"] = out[src_close].astype(float)

    src_open = pick("open", "o")
    src_high = pick("high", "h")
    src_low = pick("low", "l")
    src_vol = pick("volume", "vol", "v")

    if src_open is not None:
        out["open"] = out[src_open].astype(float)
    if src_high is not None:
        out["high"] = out[src_high].astype(float)
    if src_low is not None:
        out["low"] = out[src_low].astype(float)
    if src_vol is not None:
        out["volume"] = out[src_vol].astype(float)

    return out


def _rsi(close: pd.Series, length: int) -> pd.Series:
    """RSI calculation (Wilder's smoothing)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def _action_dict(
    *,
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    horizon_hours: int,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "action": str(action),
        "confidence": float(confidence),
        "desired_exposure_frac": float(desired_exposure_frac),
        "horizon_hours": int(horizon_hours),
        "reason": str(reason),
        "meta": meta,
    }


def generate_intent(df: pd.DataFrame, ctx: Dict[str, Any], *, closed_only: bool = True) -> Dict[str, Any]:
    """Layer 2 strategy entrypoint"""
    
    df0 = _safe_copy_df(df)
    df0 = _normalize_ohlcv_columns(df0)

    # Parameters
    rsi_len = _get_env_int(ENV_RSI_LEN, 14)
    rsi_oversold = _get_env_float(ENV_RSI_OVERSOLD, 30.0)
    rsi_exit = _get_env_float(ENV_RSI_EXIT, 60.0)
    horizon_hours = _get_env_int(ENV_HORIZON_HOURS, 12)
    exposure_frac = _get_env_float(ENV_EXPOSURE_FRAC, 0.25)

    # Layer 1: Regime classification
    regime_state = None
    dropped_last_row = False
    regime_meta = None
    regime_err = None
    
    try:
        reg = classify_regime(df0, closed_only=closed_only)
        regime_state = getattr(reg, "label", None)
        regime_meta = getattr(reg, "meta", None)
        dropped_last_row = bool((regime_meta or {}).get("dropped_last_row", False))
        if isinstance(regime_state, str):
            regime_state = regime_state.strip()
        else:
            regime_state = None
    except Exception as e:
        regime_err = f"{type(e).__name__}: {e}"

    # Mirror Layer 1 timeline
    df_used = df0.iloc[:-1] if dropped_last_row else df0

    meta: Dict[str, Any] = {
        "strategy": "sg_mean_reversion_a",
        "closed_only": bool(closed_only),
        "regime_state": regime_state,
        "dropped_last_row": bool(dropped_last_row),
        "regime_meta": regime_meta,
        "regime_error": regime_err,
        "params": {
            "rsi_len": rsi_len,
            "rsi_oversold": rsi_oversold,
            "rsi_exit": rsi_exit,
            "horizon_hours": horizon_hours,
            "exposure_frac": exposure_frac,
        },
    }

    # Fail-closed guards
    if regime_state is None:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="missing_regime_state_fail_closed",
            meta=meta,
        )

    if df_used is None or "close" not in df_used.columns:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="missing_close_column",
            meta=meta,
        )

    need = rsi_len + 3
    if len(df_used) < need:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"insufficient_history(len={len(df_used)}, need={need})",
            meta=meta,
        )

    # Calculate RSI
    close = df_used["close"].astype(float)
    rsi = _rsi(close, rsi_len)

    last_i = df_used.index[-1]
    rsi_val = float(rsi.loc[last_i]) if np.isfinite(rsi.loc[last_i]) else np.nan
    px = float(close.loc[last_i]) if np.isfinite(close.loc[last_i]) else np.nan

    signal = {"price": px, "rsi": rsi_val}
    meta = {**meta, "signal": signal}

    # NaN guard
    if not np.isfinite(rsi_val):
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="nan_guard(rsi_invalid)",
            meta=meta,
        )

    # Regime gates
    if regime_state == "PANIC":
        return _action_dict(
            action="EXIT_LONG",
            confidence=1.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_panic_force_exit",
            meta=meta,
        )

    # Only trade in CHOP or VOL_COMPRESSION
    if regime_state not in ["CHOP", "VOL_COMPRESSION"]:
        # Defer to Core in trending regimes
        return _action_dict(
            action="HOLD",
            confidence=0.20,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"regime_not_target({regime_state})",
            meta=meta,
        )

    # Mean reversion logic
    # Entry: RSI oversold
    if rsi_val < rsi_oversold:
        # Confidence based on how oversold
        if rsi_val < 20:
            conf = 0.75  # Extremely oversold
        elif rsi_val < 25:
            conf = 0.65  # Very oversold
        else:
            conf = 0.55  # Moderately oversold
        
        return _action_dict(
            action="ENTER_LONG",
            confidence=conf,
            desired_exposure_frac=exposure_frac,
            horizon_hours=horizon_hours,
            reason=f"rsi_oversold(rsi={rsi_val:.1f})",
            meta=meta,
        )

    # Exit: RSI returned to normal/overbought
    if rsi_val > rsi_exit:
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.70,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"rsi_exit_target(rsi={rsi_val:.1f})",
            meta=meta,
        )

    # Hold: Waiting for signal
    return _action_dict(
        action="HOLD",
        confidence=0.30,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason=f"waiting_for_signal(rsi={rsi_val:.1f})",
        meta=meta,
    )
