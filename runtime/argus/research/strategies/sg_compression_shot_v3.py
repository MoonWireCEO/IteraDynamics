# File: runtime/argus/research/strategies/sg_compression_shot_v3.py
"""
sg_compression_shot_v3 — Volatility Compression "Shot" with Compression Memory (Layer 2)

What changed vs v2:
- Fixes regime-scarcity by expanding entry eligibility with a *compression memory* window.
- Entry eligibility is true if:
    (regime_state == "VOL_COMPRESSION") OR (compression_recent == True)
  where compression_recent is derived ONLY from df using BB width history:
    compressed_state = (bb_width <= width_compress_max)
    compression_recent = any(compressed_state in last mem_bars), optionally with min_hits.

Core constraints preserved:
- Deterministic, stateless, no side effects
- closed_only=True default
- Calls Layer 1 classify_regime(df, closed_only=closed_only)
- Mirrors Layer 1 dropped_last_row to avoid forming candle leakage (no double-dropping)
- Strict StrategyIntent contract keys only:
    action, confidence, desired_exposure_frac, horizon_hours, reason, meta
- Hard exits identical to v2:
    EXIT_LONG only on PANIC, TREND_DOWN, VOL_EXPANSION
- If in_position inferred from ctx (best-effort), HOLD through regime churn (Prime horizon manages hold length)
- Normalizes OHLCV columns to support Flight Recorder schema (capitalized) and lowercase variants.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from research.regime.regime_engine import classify_regime

STRATEGY_NAME = "sg_compression_shot_v3"


# -----------------------------
# ENV helpers (deterministic)
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


# -----------------------------
# Data helpers
# -----------------------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with standardized OHLCV names:
      open, high, low, close, volume, timestamp (if present)
    Supports capitalized Flight Recorder schema and common lowercase schema.
    Does NOT mutate input df in-place.
    """
    if df is None or len(df) == 0:
        return df

    lower = {c.lower(): c for c in df.columns}
    col_map: Dict[str, str] = {}

    for want in ["open", "high", "low", "close", "volume", "timestamp"]:
        if want in lower:
            col_map[lower[want]] = want

    if "timestamp" not in col_map.values():
        if "date" in lower:
            col_map[lower["date"]] = "timestamp"
        elif "time" in lower:
            col_map[lower["time"]] = "timestamp"

    return df.rename(columns=col_map).copy()


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
      mid, upper, lower, width
    width = (upper - lower) / mid
    """
    mid = close.rolling(window=n, min_periods=n).mean()
    sd = _rolling_std(close, n)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return mid, upper, lower, width


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def _macro_filter(close: pd.Series, ema_len: int) -> Dict[str, Any]:
    """
    Macro bull definition:
      macro_bull = close > EMA(ema_len) AND EMA slope > 0 (1-step slope)
    """
    ema = _ema(close, ema_len)
    ema_last = _safe_float(ema.iloc[-1]) if len(ema) > 0 else None
    ema_prev = _safe_float(ema.iloc[-2]) if len(ema) > 1 else None
    slope = None
    if ema_last is not None and ema_prev is not None:
        slope = ema_last - ema_prev

    c_last = _safe_float(close.iloc[-1]) if len(close) > 0 else None
    macro_bull = False
    if c_last is not None and ema_last is not None and slope is not None:
        macro_bull = (c_last > ema_last) and (slope > 0.0)

    return {"macro_ema": ema_last, "macro_slope": slope, "macro_bull": bool(macro_bull)}


def _infer_in_position(ctx: Dict[str, Any]) -> Optional[bool]:
    """
    Best-effort inference from ctx. We do NOT require this to be perfect.
    Returns True/False if confidently inferred, otherwise None.
    """
    if not isinstance(ctx, dict):
        return None

    # Common patterns / likely keys (non-breaking checks)
    for k in ("in_position", "in_pos", "position_open", "has_position"):
        v = ctx.get(k, None)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(v)

    # Nested structures (sometimes ctx carries execution state)
    prime = ctx.get("prime", None)
    if isinstance(prime, dict):
        v = prime.get("in_position", None)
        if isinstance(v, bool):
            return v

    # If we can’t confidently infer, return None
    return None


def _result(
    *,
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    horizon_hours: int,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
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
    # v2 params + new v3 memory params
    bb_len = _env_int("SG_COMP_BB_LEN", 20)
    bb_std = _env_float("SG_COMP_BB_STD", 2.0)
    width_expand_factor = _env_float("SG_COMP_WIDTH_EXPAND_FACTOR", 1.20)
    horizon_hours = _env_int("SG_COMP_HORIZON_H", 18)

    macro_ema_len = _env_int("SG_COMP_MACRO_EMA_LEN", 2000)
    cap_bull = _env_float("SG_COMP_CAP_BULL", 0.20)
    cap_bear = _env_float("SG_COMP_CAP_BEAR", 0.05)
    enable_macro = _env_bool01("SG_COMP_ENABLE_MACRO_FILTER", True)

    # NEW v3 knobs
    mem_bars = _env_int("SG_COMP_MEMORY_BARS", 72)
    width_compress_max = _env_float("SG_COMP_WIDTH_COMPRESS_MAX", 0.06)
    mem_min_hits = _env_int("SG_COMP_MEMORY_MIN_HITS", 1)

    df0 = _std_cols(df)

    params = {
        "bb_len": bb_len,
        "bb_std": bb_std,
        "width_expand_factor": width_expand_factor,
        "horizon_hours": horizon_hours,
        "macro_ema_len": macro_ema_len,
        "cap_bull": cap_bull,
        "cap_bear": cap_bear,
        "enable_macro": bool(enable_macro),
        "mem_bars": mem_bars,
        "width_compress_max": width_compress_max,
        "mem_min_hits": mem_min_hits,
    }

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
                "params": params,
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
                "params": params,
            },
        )

    # Classify regime (latest-bar) + mirror dropped_last_row
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

    # Mirror Layer 1’s timeline to avoid forming candle leakage.
    # If Layer 1 dropped last row (forming candle), we compute indicators on df_eval[:-1].
    df_eval = df0.iloc[:-1] if (dropped_last_row and len(df0) > 1) else df0
    close = pd.to_numeric(df_eval["close"], errors="coerce")

    # Need sufficient history: BB uses (bb_len), width ratio uses prev bar (+1),
    # macro uses (macro_ema_len + 1) when enabled, memory uses mem_bars on width series.
    need_bb = bb_len + 1
    need_macro = (macro_ema_len + 1) if enable_macro else 0
    need_mem = (bb_len + max(2, mem_bars))  # width needs bb_len, then memory window on width
    need = max(need_bb, need_macro, need_mem)

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
                "params": params,
                "macro": {"macro_bull": None, "macro_ema": None, "macro_slope": None, "macro_ema_len": macro_ema_len},
                "signal": {},
                "compression_recent": None,
                "compression_hits": None,
                "mem_bars": mem_bars,
                "width_compress_max": width_compress_max,
            },
        )

    # Compute indicators
    bb_mid, bb_upper, bb_lower, bb_width = _bollinger(close, bb_len, bb_std)

    price_last = _safe_float(close.iloc[-1])
    bb_mid_last = _safe_float(bb_mid.iloc[-1])
    bb_upper_last = _safe_float(bb_upper.iloc[-1])
    bb_width_last = _safe_float(bb_width.iloc[-1])
    bb_width_prev = _safe_float(bb_width.iloc[-2]) if len(bb_width) > 1 else None

    width_ratio = None
    if bb_width_last is not None and bb_width_prev is not None and bb_width_prev > 0:
        width_ratio = bb_width_last / bb_width_prev

    # Macro filter (same behavior as v2)
    macro = {"macro_bull": None, "macro_ema": None, "macro_slope": None, "macro_ema_len": macro_ema_len}
    macro_bull = True  # if macro disabled, treat as bull for cap selection
    if enable_macro:
        m = _macro_filter(close, macro_ema_len)
        macro.update(m)
        macro_bull = bool(m.get("macro_bull", False))

    # v3 compression memory (derived ONLY from df indicators)
    # compressed_state := bb_width <= width_compress_max
    # compression_recent := any(compressed_state in last mem_bars), with optional min_hits
    # Note: use the width series, and restrict to last mem_bars values that are non-null.
    w = bb_width.copy()
    compressed = (w <= float(width_compress_max))
    compressed = compressed.fillna(False).astype(bool)

    tail = compressed.iloc[-int(mem_bars) :] if int(mem_bars) > 0 else compressed.iloc[-0:]
    compression_hits = int(tail.sum()) if len(tail) > 0 else 0
    compression_recent = bool(compression_hits >= int(mem_min_hits))

    # Entry conditions (unchanged)
    width_expanding = (width_ratio is not None) and (width_ratio >= float(width_expand_factor))
    breakout = (price_last is not None) and (bb_upper_last is not None) and (price_last > bb_upper_last)

    # Hard exits (IDENTICAL gating as v2) — only these produce EXIT_LONG
    if regime_state in ("PANIC", "TREND_DOWN", "VOL_EXPANSION"):
        r = (
            "regime_panic_exit"
            if regime_state == "PANIC"
            else "regime_trend_down_exit"
            if regime_state == "TREND_DOWN"
            else "regime_vol_expansion_exit"
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
                "params": params,
                "macro": macro,
                "signal": {
                    "price": price_last,
                    "bb_mid": bb_mid_last,
                    "bb_upper": bb_upper_last,
                    "bb_width": bb_width_last,
                    "bb_width_prev": bb_width_prev,
                    "width_ratio": width_ratio,
                },
                "compression_recent": compression_recent,
                "compression_hits": compression_hits,
                "mem_bars": int(mem_bars),
                "width_compress_max": float(width_compress_max),
            },
        )

    # If in_position: HOLD through regime churn (no exits except hard exits above)
    in_pos = _infer_in_position(ctx)
    if in_pos is True:
        return _result(
            action="HOLD",
            confidence=0.2,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="in_position_hold_through_churn",
            meta={
                "strategy": STRATEGY_NAME,
                "regime_state": regime_state,
                "closed_only": bool(closed_only),
                "dropped_last_row": bool(dropped_last_row),
                "regime_meta": regime_meta,
                "regime_error": regime_error,
                "params": params,
                "macro": macro,
                "signal": {
                    "price": price_last,
                    "bb_mid": bb_mid_last,
                    "bb_upper": bb_upper_last,
                    "bb_width": bb_width_last,
                    "bb_width_prev": bb_width_prev,
                    "width_ratio": width_ratio,
                },
                "compression_recent": compression_recent,
                "compression_hits": compression_hits,
                "mem_bars": int(mem_bars),
                "width_compress_max": float(width_compress_max),
                "ctx_in_position": True,
            },
        )

    # If regime is unavailable, fail closed (no entry)
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
                "params": params,
                "macro": macro,
                "signal": {
                    "price": price_last,
                    "bb_mid": bb_mid_last,
                    "bb_upper": bb_upper_last,
                    "bb_width": bb_width_last,
                    "bb_width_prev": bb_width_prev,
                    "width_ratio": width_ratio,
                },
                "compression_recent": compression_recent,
                "compression_hits": compression_hits,
                "mem_bars": int(mem_bars),
                "width_compress_max": float(width_compress_max),
                "ctx_in_position": in_pos,
            },
        )

    # v3 eligibility expansion:
    eligible = (regime_state == "VOL_COMPRESSION") or bool(compression_recent)

    if eligible and width_expanding and breakout:
        cap = cap_bull if macro_bull else cap_bear

        # Confidence bucket using ratio
        if width_ratio is None:
            conf = 0.55
            bucket = "unknown"
        elif width_ratio >= (float(width_expand_factor) * 1.5):
            conf = 0.75
            bucket = "strong_expand"
        elif width_ratio >= (float(width_expand_factor) * 1.25):
            conf = 0.65
            bucket = "moderate_expand"
        else:
            conf = 0.60
            bucket = "weak_expand"

        why_eligible = "regime" if (regime_state == "VOL_COMPRESSION") else "memory"

        return _result(
            action="ENTER_LONG",
            confidence=conf,
            desired_exposure_frac=float(max(0.0, min(1.0, cap))),
            horizon_hours=horizon_hours,
            reason=f"compression_breakout_v3(bucket={bucket}, eligible={why_eligible})",
            meta={
                "strategy": STRATEGY_NAME,
                "regime_state": regime_state,
                "closed_only": bool(closed_only),
                "dropped_last_row": bool(dropped_last_row),
                "regime_meta": regime_meta,
                "regime_error": regime_error,
                "params": params,
                "macro": macro,
                "signal": {
                    "price": price_last,
                    "bb_mid": bb_mid_last,
                    "bb_upper": bb_upper_last,
                    "bb_width": bb_width_last,
                    "bb_width_prev": bb_width_prev,
                    "width_ratio": width_ratio,
                },
                "compression_recent": compression_recent,
                "compression_hits": compression_hits,
                "mem_bars": int(mem_bars),
                "width_compress_max": float(width_compress_max),
                "ctx_in_position": in_pos,
            },
        )

    # Otherwise HOLD (no entry)
    return _result(
        action="HOLD",
        confidence=0.2 if (breakout or width_expanding) else 0.1,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason=f"no_entry_v3(eligible={eligible}, width_expanding={width_expanding}, breakout={breakout})",
        meta={
            "strategy": STRATEGY_NAME,
            "regime_state": regime_state,
            "closed_only": bool(closed_only),
            "dropped_last_row": bool(dropped_last_row),
            "regime_meta": regime_meta,
            "regime_error": regime_error,
            "params": params,
            "macro": macro,
            "signal": {
                "price": price_last,
                "bb_mid": bb_mid_last,
                "bb_upper": bb_upper_last,
                "bb_width": bb_width_last,
                "bb_width_prev": bb_width_prev,
                "width_ratio": width_ratio,
            },
            "compression_recent": compression_recent,
            "compression_hits": compression_hits,
            "mem_bars": int(mem_bars),
            "width_compress_max": float(width_compress_max),
            "ctx_in_position": in_pos,
        },
    )