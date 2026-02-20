# File: runtime/argus/research/strategies/sg_compression_shot_v1.py
"""
sg_compression_shot_v1 — Volatility Compression "Shot" (Layer 2)

Purpose:
- Structural + research-friendly probe for compression->breakout behavior
- Only participates during VOL_COMPRESSION regime
- Deterministic, closed_only=True default
- No side effects, no broker/wallet/state/files

Contract:
generate_intent(df, ctx, *, closed_only=True) -> dict with keys:
  action, confidence, desired_exposure_frac, horizon_hours, reason, meta

Regime:
- Calls Layer 1 directly via research.regime.regime_engine.classify_regime(df, closed_only=closed_only)
- Mirrors Layer 1 dropped_last_row via RegimeState.meta["dropped_last_row"]
- Avoids double-dropping: strategy does NOT drop last row itself.

Column normalization:
- Supports flight recorder schema (Timestamp/Open/High/Low/Close/Volume) and lowercase variants.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from research.regime.regime_engine import classify_regime


STRATEGY_NAME = "sg_compression_shot_v1"


# -----------------------------
# Helpers (deterministic, pure)
# -----------------------------
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_bool01(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return bool(default)
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a view/copy of df with standardized OHLCV names:
      open, high, low, close, volume, timestamp (if present)
    Supports capitalized Flight Recorder schema and common lowercase schema.
    Does NOT mutate input df in-place.
    """
    if df is None or len(df) == 0:
        return df

    col_map = {}
    lower = {c.lower(): c for c in df.columns}

    # common variants
    for want in ["open", "high", "low", "close", "volume", "timestamp"]:
        if want in lower:
            col_map[lower[want]] = want

    # sometimes "date" instead of timestamp
    if "timestamp" not in col_map.values():
        if "date" in lower:
            col_map[lower["date"]] = "timestamp"
        elif "time" in lower:
            col_map[lower["time"]] = "timestamp"

    out = df.rename(columns=col_map).copy()
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _rolling_std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(window=n, min_periods=n).std(ddof=0)


def _bollinger(close: pd.Series, n: int, k: float) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      mid, upper, lower, width  (width = (upper-lower)/mid)
    """
    mid = close.rolling(window=n, min_periods=n).mean()
    sd = _rolling_std(close, n)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return mid, upper, lower, width


def _ema(series: pd.Series, n: int) -> pd.Series:
    # pandas ewm is deterministic given fixed inputs
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def _macro_filter(close: pd.Series, ema_len: int) -> Dict[str, Any]:
    """
    Macro bull definition:
      macro_bull = close > EMA(ema_len) AND EMA slope > 0 (simple 1-step slope)
    Returns dict with macro_ema, macro_slope, macro_bull (last available point).
    """
    ema = _ema(close, ema_len)
    ema_last = _safe_float(ema.iloc[-1]) if len(ema) > 0 else None
    ema_prev = _safe_float(ema.iloc[-2]) if len(ema) > 1 else None
    slope = None
    if ema_last is not None and ema_prev is not None:
        slope = ema_last - ema_prev
    macro_bull = False
    c_last = _safe_float(close.iloc[-1]) if len(close) > 0 else None
    if c_last is not None and ema_last is not None and slope is not None:
        macro_bull = (c_last > ema_last) and (slope > 0.0)
    return {
        "macro_ema": ema_last,
        "macro_slope": slope,
        "macro_bull": bool(macro_bull),
    }


def _result(
    *,
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    horizon_hours: int,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    # Strict contract keys only
    return {
        "action": action,
        "confidence": float(confidence),
        "desired_exposure_frac": float(desired_exposure_frac),
        "horizon_hours": int(horizon_hours),
        "reason": str(reason),
        "meta": meta,
    }


# -----------------------------
# Strategy
# -----------------------------
def generate_intent(df: pd.DataFrame, ctx: Dict[str, Any], *, closed_only: bool = True) -> Dict[str, Any]:
    """
    Strategy behavior:
      - Active only when regime == VOL_COMPRESSION
      - Entry requires:
          width_ratio = bb_width / bb_width_prev >= 1.20
          AND breakout: close > bb_upper
      - Exposure cap depends on EMA2000 macro filter:
          if macro_bull: cap=0.20 else cap=0.05
      - Exit on PANIC, TREND_DOWN, VOL_EXPANSION, or any non-compression regime:
          return EXIT_LONG (desired_exposure_frac=0)
    """
    # Params (ENV knobs)
    bb_len = _env_int("SG_COMP_BB_LEN", 20)
    bb_std = _env_float("SG_COMP_BB_STD", 2.0)
    width_expand_factor = _env_float("SG_COMP_WIDTH_EXPAND_FACTOR", 1.20)

    horizon_hours = _env_int("SG_COMP_HORIZON_H", 18)

    macro_ema_len = _env_int("SG_COMP_MACRO_EMA_LEN", 2000)
    cap_bull = _env_float("SG_COMP_CAP_BULL", 0.20)
    cap_bear = _env_float("SG_COMP_CAP_BEAR", 0.05)
    enable_macro = _env_bool01("SG_COMP_ENABLE_MACRO_FILTER", True)

    df0 = _std_cols(df)

    # Basic column sanity
    if df0 is None or len(df0) == 0:
        return _result(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="empty_df",
            meta={
                "strategy": STRATEGY_NAME,
                "closed_only": bool(closed_only),
                "dropped_last_row": False,
                "regime_state": None,
                "params": {
                    "bb_len": bb_len,
                    "bb_std": bb_std,
                    "width_expand_factor": width_expand_factor,
                    "horizon_hours": horizon_hours,
                    "macro_ema_len": macro_ema_len,
                    "cap_bull": cap_bull,
                    "cap_bear": cap_bear,
                    "enable_macro": bool(enable_macro),
                },
            },
        )

    if "close" not in df0.columns:
        return _result(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="missing_close_column",
            meta={
                "strategy": STRATEGY_NAME,
                "closed_only": bool(closed_only),
                "dropped_last_row": False,
                "regime_state": None,
                "params": {
                    "bb_len": bb_len,
                    "bb_std": bb_std,
                    "width_expand_factor": width_expand_factor,
                    "horizon_hours": horizon_hours,
                    "macro_ema_len": macro_ema_len,
                    "cap_bull": cap_bull,
                    "cap_bear": cap_bear,
                    "enable_macro": bool(enable_macro),
                },
            },
        )

    # Ensure numeric close
    close = pd.to_numeric(df0["close"], errors="coerce")

    # Layer 1 regime classification (no ctx, mirror dropped_last_row from Layer 1)
    regime_error = None
    try:
        reg = classify_regime(df0, closed_only=closed_only)
        regime_state = getattr(reg, "label", None)
        dropped_last_row = bool(getattr(reg, "meta", {}).get("dropped_last_row", False))
        regime_meta = getattr(reg, "meta", {}) or {}
    except Exception as e:
        regime_state = None
        dropped_last_row = False
        regime_meta = {}
        regime_error = f"{type(e).__name__}: {e}"

    # We rely on Layer 1 to mirror timeline; DO NOT drop locally.
    # For indicators, we still use the same df0/close as provided.
    # If closed_only=True, Layer 1 may have used a trimmed view internally;
    # our 'dropped_last_row' in meta reflects Layer 1's behavior.

    # Need sufficient history for BB + macro
    need = max(bb_len + 1, macro_ema_len + 1) if enable_macro else (bb_len + 1)
    if len(close) < need:
        return _result(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"insufficient_history(len={len(close)}, need={need})",
            meta={
                "strategy": STRATEGY_NAME,
                "regime_state": regime_state,
                "closed_only": bool(closed_only),
                "dropped_last_row": bool(dropped_last_row),
                "regime_meta": regime_meta,
                "regime_error": regime_error,
                "params": {
                    "bb_len": bb_len,
                    "bb_std": bb_std,
                    "width_expand_factor": width_expand_factor,
                    "horizon_hours": horizon_hours,
                    "macro_ema_len": macro_ema_len,
                    "cap_bull": cap_bull,
                    "cap_bear": cap_bear,
                    "enable_macro": bool(enable_macro),
                },
                "signal": {},
                "macro": {"macro_bull": None, "macro_ema": None, "macro_slope": None, "macro_ema_len": macro_ema_len},
            },
        )

    # Compute BB
    bb_mid, bb_upper, bb_lower, bb_width = _bollinger(close, bb_len, bb_std)
    bb_mid_last = _safe_float(bb_mid.iloc[-1])
    bb_upper_last = _safe_float(bb_upper.iloc[-1])
    bb_width_last = _safe_float(bb_width.iloc[-1])
    bb_width_prev = _safe_float(bb_width.iloc[-2]) if len(bb_width) > 1 else None

    price_last = _safe_float(close.iloc[-1])

    width_ratio = None
    if bb_width_last is not None and bb_width_prev is not None and bb_width_prev > 0:
        width_ratio = bb_width_last / bb_width_prev

    # Macro filter
    macro = {"macro_bull": None, "macro_ema": None, "macro_slope": None, "macro_ema_len": macro_ema_len}
    macro_bull = True  # if macro filter disabled, treat as bull for cap selection
    if enable_macro:
        macro_calc = _macro_filter(close, macro_ema_len)
        macro.update(macro_calc)
        macro_bull = bool(macro_calc.get("macro_bull", False))

    # Exit on risk/off regimes (hard exit)
    if regime_state in ("PANIC", "TREND_DOWN", "VOL_EXPANSION") or (regime_state is not None and regime_state != "VOL_COMPRESSION"):
        r = (
            "regime_panic_exit"
            if regime_state == "PANIC"
            else "regime_trend_down_exit"
            if regime_state == "TREND_DOWN"
            else "regime_vol_expansion_exit"
            if regime_state == "VOL_EXPANSION"
            else f"regime_not_compression({regime_state})"
        )
        return _result(
            action="EXIT_LONG",
            confidence=0.9,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=r,
            meta={
                "strategy": STRATEGY_NAME,
                "regime_state": regime_state,
                "closed_only": bool(closed_only),
                "dropped_last_row": bool(dropped_last_row),
                "regime_meta": regime_meta,
                "regime_error": regime_error,
                "params": {
                    "bb_len": bb_len,
                    "bb_std": bb_std,
                    "width_expand_factor": width_expand_factor,
                    "horizon_hours": horizon_hours,
                    "macro_ema_len": macro_ema_len,
                    "cap_bull": cap_bull,
                    "cap_bear": cap_bear,
                    "enable_macro": bool(enable_macro),
                },
                "macro": macro,
                "signal": {
                    "price": price_last,
                    "bb_mid": bb_mid_last,
                    "bb_upper": bb_upper_last,
                    "bb_width": bb_width_last,
                    "bb_width_prev": bb_width_prev,
                    "width_ratio": width_ratio,
                },
            },
        )

    # If we couldn't classify regime, fail closed (no entry)
    if regime_state is None:
        return _result(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_unavailable",
            meta={
                "strategy": STRATEGY_NAME,
                "regime_state": None,
                "closed_only": bool(closed_only),
                "dropped_last_row": bool(dropped_last_row),
                "regime_meta": regime_meta,
                "regime_error": regime_error,
                "params": {
                    "bb_len": bb_len,
                    "bb_std": bb_std,
                    "width_expand_factor": width_expand_factor,
                    "horizon_hours": horizon_hours,
                    "macro_ema_len": macro_ema_len,
                    "cap_bull": cap_bull,
                    "cap_bear": cap_bear,
                    "enable_macro": bool(enable_macro),
                },
                "macro": macro,
                "signal": {
                    "price": price_last,
                    "bb_mid": bb_mid_last,
                    "bb_upper": bb_upper_last,
                    "bb_width": bb_width_last,
                    "bb_width_prev": bb_width_prev,
                    "width_ratio": width_ratio,
                },
            },
        )

    # Active only in VOL_COMPRESSION (at this point)
    # Entry conditions: width expansion AND breakout above upper band
    width_expanding = (width_ratio is not None) and (width_ratio >= width_expand_factor)
    breakout = (price_last is not None) and (bb_upper_last is not None) and (price_last > bb_upper_last)

    if width_expanding and breakout:
        cap = cap_bull if macro_bull else cap_bear
        # Confidence buckets: stronger if ratio is high
        if width_ratio is None:
            conf = 0.55
            bucket = "unknown"
        elif width_ratio >= (width_expand_factor * 1.5):
            conf = 0.75
            bucket = "strong_expand"
        elif width_ratio >= (width_expand_factor * 1.25):
            conf = 0.65
            bucket = "moderate_expand"
        else:
            conf = 0.60
            bucket = "weak_expand"

        return _result(
            action="ENTER_LONG",
            confidence=conf,
            desired_exposure_frac=float(max(0.0, min(1.0, cap))),
            horizon_hours=horizon_hours,
            reason=f"compression_breakout(bucket={bucket})",
            meta={
                "strategy": STRATEGY_NAME,
                "regime_state": regime_state,
                "closed_only": bool(closed_only),
                "dropped_last_row": bool(dropped_last_row),
                "regime_meta": regime_meta,
                "regime_error": regime_error,
                "params": {
                    "bb_len": bb_len,
                    "bb_std": bb_std,
                    "width_expand_factor": width_expand_factor,
                    "horizon_hours": horizon_hours,
                    "macro_ema_len": macro_ema_len,
                    "cap_bull": cap_bull,
                    "cap_bear": cap_bear,
                    "enable_macro": bool(enable_macro),
                },
                "macro": macro,
                "signal": {
                    "price": price_last,
                    "bb_mid": bb_mid_last,
                    "bb_upper": bb_upper_last,
                    "bb_width": bb_width_last,
                    "bb_width_prev": bb_width_prev,
                    "width_ratio": width_ratio,
                },
            },
        )

    # Otherwise hold (no new exposure)
    # Keep desired_exposure_frac at 0.0 (Prime/horizon will manage existing positions elsewhere).
    return _result(
        action="HOLD",
        confidence=0.2 if (breakout or width_expanding) else 0.1,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason=f"no_entry(width_expanding={width_expanding}, breakout={breakout})",
        meta={
            "strategy": STRATEGY_NAME,
            "regime_state": regime_state,
            "closed_only": bool(closed_only),
            "dropped_last_row": bool(dropped_last_row),
            "regime_meta": regime_meta,
            "regime_error": regime_error,
            "params": {
                "bb_len": bb_len,
                "bb_std": bb_std,
                "width_expand_factor": width_expand_factor,
                "horizon_hours": horizon_hours,
                "macro_ema_len": macro_ema_len,
                "cap_bull": cap_bull,
                "cap_bear": cap_bear,
                "enable_macro": bool(enable_macro),
            },
            "macro": macro,
            "signal": {
                "price": price_last,
                "bb_mid": bb_mid_last,
                "bb_upper": bb_upper_last,
                "bb_width": bb_width_last,
                "bb_width_prev": bb_width_prev,
                "width_ratio": width_ratio,
            },
        },
    )