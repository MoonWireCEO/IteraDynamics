"""
sg_mean_reversion_b.py

Sleeve 2 Candidate B: Bollinger Mean Reversion
Itera Dynamics - Research Phase

Mandate:
- Monetize CHOP and VOL_COMPRESSION regimes
- Bollinger Band mean reversion
- Volatility-scaled exposure

Design:
- Entry: Price < BB_lower (2.0 std) in CHOP/VOL_COMPRESSION
- Exit: Price > BB_mid OR regime flip
- Horizon: 18 hours
- Exposure: 0.25-0.35 (volatility scaled)

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
ENV_BB_LEN = "SG_MR_B_BB_LEN"                      # default 20
ENV_BB_STD = "SG_MR_B_BB_STD"                      # default 2.0
ENV_HORIZON_HOURS = "SG_MR_B_HORIZON_HOURS"        # default 18
ENV_BASE_EXPOSURE = "SG_MR_B_BASE_EXPOSURE"        # default 0.30
ENV_ATR_LEN = "SG_MR_B_ATR_LEN"                    # default 14
ENV_TARGET_ATR_PCT = "SG_MR_B_TARGET_ATR_PCT"      # default 0.010
ENV_SCALE_CAP = "SG_MR_B_SCALE_CAP"                # default 1.20


def _get_env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)).strip())
        return v if v > 0 else default
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, str(default)).strip())
        if "FRAC" in name.upper() or "EXPO" in name.upper() or "CAP" in name.upper():
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
    src_open = pick("open", "o")
    src_high = pick("high", "h")
    src_low = pick("low", "l")
    src_vol = pick("volume", "vol", "v")

    if src_close is None:
        return df

    out = df.copy(deep=False)
    out["close"] = out[src_close].astype(float)
    
    if src_open is not None:
        out["open"] = out[src_open].astype(float)
    if src_high is not None:
        out["high"] = out[src_high].astype(float)
    if src_low is not None:
        out["low"] = out[src_low].astype(float)
    if src_vol is not None:
        out["volume"] = out[src_vol].astype(float)

    return out


def _bollinger(close: pd.Series, length: int, n_std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands calculation"""
    mid = close.rolling(window=length, min_periods=length).mean()
    sd = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + (n_std * sd)
    lower = mid - (n_std * sd)
    return mid, upper, lower


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    """ATR calculation"""
    high = df.get("high")
    low = df.get("low")
    close = df.get("close")
    
    if high is None or low is None or close is None:
        return pd.Series(np.nan, index=df.index)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=length, min_periods=length).mean()


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
    bb_len = _get_env_int(ENV_BB_LEN, 20)
    bb_std = _get_env_float(ENV_BB_STD, 2.0)
    horizon_hours = _get_env_int(ENV_HORIZON_HOURS, 18)
    base_exposure = _get_env_float(ENV_BASE_EXPOSURE, 0.30)
    atr_len = _get_env_int(ENV_ATR_LEN, 14)
    target_atr_pct = _get_env_float(ENV_TARGET_ATR_PCT, 0.010)
    scale_cap = _get_env_float(ENV_SCALE_CAP, 1.20)

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
        "strategy": "sg_mean_reversion_b",
        "closed_only": bool(closed_only),
        "regime_state": regime_state,
        "dropped_last_row": bool(dropped_last_row),
        "regime_meta": regime_meta,
        "regime_error": regime_err,
        "params": {
            "bb_len": bb_len,
            "bb_std": bb_std,
            "horizon_hours": horizon_hours,
            "base_exposure": base_exposure,
            "atr_len": atr_len,
            "target_atr_pct": target_atr_pct,
            "scale_cap": scale_cap,
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

    need = max(bb_len, atr_len) + 3
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
    bb_mid, bb_upper, bb_lower = _bollinger(close, bb_len, bb_std)
    atr = _atr(df_used, atr_len)

    last_i = df_used.index[-1]
    px = float(close.loc[last_i]) if np.isfinite(close.loc[last_i]) else np.nan
    mid = float(bb_mid.loc[last_i]) if np.isfinite(bb_mid.loc[last_i]) else np.nan
    upper = float(bb_upper.loc[last_i]) if np.isfinite(bb_upper.loc[last_i]) else np.nan
    lower = float(bb_lower.loc[last_i]) if np.isfinite(bb_lower.loc[last_i]) else np.nan
    atr_val = float(atr.loc[last_i]) if np.isfinite(atr.loc[last_i]) else np.nan

    # Volatility scaling
    atr_pct = (atr_val / px) if (np.isfinite(atr_val) and np.isfinite(px) and px > 0) else np.nan
    if np.isfinite(atr_pct) and atr_pct > 0:
        vol_mult = min(scale_cap, target_atr_pct / max(0.003, atr_pct))
    else:
        vol_mult = 1.0

    signal = {
        "price": px,
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "atr": atr_val,
        "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else None,
        "vol_mult": float(vol_mult),
    }
    meta = {**meta, "signal": signal}

    # NaN guard
    if not (np.isfinite(px) and np.isfinite(mid) and np.isfinite(lower) and np.isfinite(upper)):
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

    # Mean reversion logic
    # Entry: Price below lower band
    if px < lower:
        # How far below? Deeper = higher confidence
        deviation = (lower - px) / mid if mid > 0 else 0
        
        if deviation > 0.04:  # >4% below
            conf = 0.75
        elif deviation > 0.02:  # >2% below
            conf = 0.65
        else:
            conf = 0.55
        
        # Volatility-scaled exposure
        exposure = min(base_exposure * vol_mult, 0.45)
        
        return _action_dict(
            action="ENTER_LONG",
            confidence=conf,
            desired_exposure_frac=exposure,
            horizon_hours=horizon_hours,
            reason=f"bb_mean_reversion(dev={deviation:.3f})",
            meta={**meta, "deviation": deviation, "exposure_scaled": exposure},
        )

    # Exit: Price returned to or above midline
    if px >= mid:
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.70,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="bb_target_reached(px_above_mid)",
            meta=meta,
        )

    # Hold: Between lower and mid (let it ride)
    return _action_dict(
        action="HOLD",
        confidence=0.40,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason="bb_between_lower_and_mid",
        meta=meta,
    )
