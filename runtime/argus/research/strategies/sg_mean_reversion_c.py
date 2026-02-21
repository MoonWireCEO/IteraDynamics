"""
sg_mean_reversion_c.py

Sleeve 2 Candidate C: Hybrid Mean Reversion (RSI + Bollinger)
Itera Dynamics - Research Phase

Mandate:
- Monetize CHOP and VOL_COMPRESSION regimes
- Confluence approach: RSI + Bollinger Bands
- Higher conviction entries, potentially better win rate

Design:
- Entry: RSI < 35 AND Price < BB_lower
- Exit: RSI > 55 OR Price > BB_mid
- Horizon: 24 hours (longer hold for confluence)
- Exposure: 0.30-0.40 (higher confidence = higher size)

Layer 2 contract compliance:
- Calls classify_regime() from Layer 1
- Returns StrategyIntent dict
- Deterministic, closed_only=True
- No side effects
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from research.regime import classify_regime


# Environment variables
ENV_RSI_LEN = "SG_MR_C_RSI_LEN"                    # default 14
ENV_RSI_OVERSOLD = "SG_MR_C_RSI_OVERSOLD"          # default 35
ENV_RSI_EXIT = "SG_MR_C_RSI_EXIT"                  # default 55
ENV_BB_LEN = "SG_MR_C_BB_LEN"                      # default 20
ENV_BB_STD = "SG_MR_C_BB_STD"                      # default 2.0
ENV_HORIZON_HOURS = "SG_MR_C_HORIZON_HOURS"        # default 24
ENV_BASE_EXPOSURE = "SG_MR_C_BASE_EXPOSURE"        # default 0.35


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
    """Deterministic OHLCV normalization"""
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


def _bollinger(close: pd.Series, length: int, n_std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands calculation"""
    mid = close.rolling(window=length, min_periods=length).mean()
    sd = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + (n_std * sd)
    lower = mid - (n_std * sd)
    return mid, upper, lower


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
    rsi_oversold = _get_env_float(ENV_RSI_OVERSOLD, 35.0)
    rsi_exit = _get_env_float(ENV_RSI_EXIT, 55.0)
    bb_len = _get_env_int(ENV_BB_LEN, 20)
    bb_std = _get_env_float(ENV_BB_STD, 2.0)
    horizon_hours = _get_env_int(ENV_HORIZON_HOURS, 24)
    base_exposure = _get_env_float(ENV_BASE_EXPOSURE, 0.35)

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
        "strategy": "sg_mean_reversion_c",
        "closed_only": bool(closed_only),
        "regime_state": regime_state,
        "dropped_last_row": bool(dropped_last_row),
        "regime_meta": regime_meta,
        "regime_error": regime_err,
        "params": {
            "rsi_len": rsi_len,
            "rsi_oversold": rsi_oversold,
            "rsi_exit": rsi_exit,
            "bb_len": bb_len,
            "bb_std": bb_std,
            "horizon_hours": horizon_hours,
            "base_exposure": base_exposure,
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

    need = max(rsi_len, bb_len) + 3
    if len(df_used) < need:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"insufficient_history(len={len(df_used)}, need={need})",
            meta=meta,
        )

    # Calculate indicators
    close = df_used["close"].astype(float)
    rsi = _rsi(close, rsi_len)
    bb_mid, bb_upper, bb_lower = _bollinger(close, bb_len, bb_std)

    last_i = df_used.index[-1]
    px = float(close.loc[last_i]) if np.isfinite(close.loc[last_i]) else np.nan
    rsi_val = float(rsi.loc[last_i]) if np.isfinite(rsi.loc[last_i]) else np.nan
    mid = float(bb_mid.loc[last_i]) if np.isfinite(bb_mid.loc[last_i]) else np.nan
    lower = float(bb_lower.loc[last_i]) if np.isfinite(bb_lower.loc[last_i]) else np.nan

    signal = {
        "price": px,
        "rsi": rsi_val,
        "bb_mid": mid,
        "bb_lower": lower,
    }
    meta = {**meta, "signal": signal}

    # NaN guard
    if not (np.isfinite(px) and np.isfinite(rsi_val) and np.isfinite(mid) and np.isfinite(lower)):
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="nan_guard(indicators_invalid)",
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
        return _action_dict(
            action="HOLD",
            confidence=0.20,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"regime_not_target({regime_state})",
            meta=meta,
        )

    # Confluence entry logic
    # Both conditions must be true for high-conviction entry
    rsi_oversold_signal = rsi_val < rsi_oversold
    bb_oversold_signal = px < lower

    if rsi_oversold_signal and bb_oversold_signal:
        # Strongest confluence: both extremely oversold
        if rsi_val < 25 and px < (lower * 0.99):
            conf = 0.85
            exposure = min(base_exposure * 1.15, 0.50)
        # Strong confluence
        elif rsi_val < 30:
            conf = 0.75
            exposure = base_exposure
        # Moderate confluence
        else:
            conf = 0.65
            exposure = base_exposure * 0.85
        
        return _action_dict(
            action="ENTER_LONG",
            confidence=conf,
            desired_exposure_frac=exposure,
            horizon_hours=horizon_hours,
            reason=f"confluence_entry(rsi={rsi_val:.1f},below_bb)",
            meta={**meta, "exposure_scaled": exposure},
        )

    # Exit logic: Either condition triggers exit
    # RSI returned to normal
    if rsi_val > rsi_exit:
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.75,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"rsi_exit(rsi={rsi_val:.1f})",
            meta=meta,
        )

    # Price returned to midline
    if px >= mid:
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.70,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="bb_mid_reached",
            meta=meta,
        )

    # Waiting for entry signal or holding position
    if rsi_oversold_signal or bb_oversold_signal:
        # Partial signal (only one condition)
        return _action_dict(
            action="HOLD",
            confidence=0.35,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"partial_signal(rsi_os={rsi_oversold_signal},bb_os={bb_oversold_signal})",
            meta=meta,
        )

    # No signal
    return _action_dict(
        action="HOLD",
        confidence=0.25,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason="waiting_for_confluence",
        meta=meta,
    )
