"""
Vol compression -> release prototype.

Role in portfolio research:
- Pre-breakout timing (quiet-to-expansion transition), not mature trend continuation.
- Intended to be behaviorally distinct from BTC VB entry timing.

Contract:
  generate_intent(df, ctx=None, closed_only=True) -> dict
Closed-bar deterministic, no lookahead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

ATR_PERIOD = 14
ATR_MA_PERIOD = 50
COMPRESSION_RATIO_MAX = 0.85
BREAKOUT_LOOKBACK = 12
TRAIL_STOP_LOOKBACK = 8
MAX_HOLD_BARS = 72
MAX_EXPOSURE = 0.985
BREAKOUT_BUFFER = 0.003
VOL_CONFIRM_MULT = 1.2
COOLDOWN_BARS = 12
STATE_KEY = "_sg_vol_compression_release_v1"


def _pick_cols(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    def _pick(*names: str) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
        return None

    o = _pick("Open", "open")
    h = _pick("High", "high")
    l = _pick("Low", "low")
    c = _pick("Close", "close")
    v = _pick("Volume", "volume")
    if any(x is None for x in (o, h, l, c, v)):
        raise ValueError("Missing OHLCV columns.")
    return o or "Open", h or "High", l or "Low", c or "Close", v or "Volume"


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return pd.Series(tr, index=close.index).rolling(period, min_periods=period).mean()


def _state_get(ctx: Any) -> Dict[str, Any]:
    if isinstance(ctx, dict):
        st = ctx.get(STATE_KEY)
        if isinstance(st, dict):
            return st
    return {}


def _state_set(ctx: Any, st: Dict[str, Any]) -> None:
    if isinstance(ctx, dict):
        ctx[STATE_KEY] = dict(st)


def generate_intent(df: pd.DataFrame, ctx: Any = None, *, closed_only: bool = True, **kwargs: Any) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {"action": "HOLD", "confidence": 0.0, "desired_exposure_frac": 0.0, "horizon_hours": 72, "reason": "no_data", "meta": {"strategy": "sg_vol_compression_release_v1"}}

    _, h_col, l_col, c_col, v_col = _pick_cols(df)
    high = pd.to_numeric(df[h_col], errors="coerce")
    low = pd.to_numeric(df[l_col], errors="coerce")
    close = pd.to_numeric(df[c_col], errors="coerce")
    volume = pd.to_numeric(df[v_col], errors="coerce").fillna(0.0)

    atr = _atr(high, low, close, ATR_PERIOD)
    atr_ma = atr.rolling(ATR_MA_PERIOD, min_periods=ATR_MA_PERIOD).mean()
    atr_ratio = atr / atr_ma.replace(0.0, np.nan)

    # Prior-bar breakout threshold for no-lookahead entry.
    breakout_high_prior = high.rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).max().shift(1)
    vol_ma = volume.rolling(20, min_periods=20).mean()

    i = len(df) - 1
    px = float(close.iloc[i]) if pd.notna(close.iloc[i]) else float("nan")
    atr_ratio_i = float(atr_ratio.iloc[i]) if pd.notna(atr_ratio.iloc[i]) else float("nan")
    bh_i = float(breakout_high_prior.iloc[i]) if pd.notna(breakout_high_prior.iloc[i]) else float("nan")
    vol_i = float(volume.iloc[i]) if pd.notna(volume.iloc[i]) else float("nan")
    vol_ma_i = float(vol_ma.iloc[i]) if pd.notna(vol_ma.iloc[i]) else float("nan")

    st = _state_get(ctx)
    in_pos = bool(st.get("in_pos", False))
    entry_i = st.get("entry_i")
    last_exit_i = st.get("last_exit_i")

    if not (np.isfinite(px) and np.isfinite(atr_ratio_i)):
        return {"action": "HOLD", "confidence": 0.0, "desired_exposure_frac": 0.0 if not in_pos else MAX_EXPOSURE, "horizon_hours": 72, "reason": "insufficient_history", "meta": {"strategy": "sg_vol_compression_release_v1", "in_pos": in_pos}}

    if in_pos:
        bars_held = (i - int(entry_i)) if isinstance(entry_i, int) else 0
        # Prior trailing low to avoid lookahead.
        tr_stop = low.rolling(TRAIL_STOP_LOOKBACK, min_periods=TRAIL_STOP_LOOKBACK).min().shift(1)
        tr_stop_i = float(tr_stop.iloc[i]) if pd.notna(tr_stop.iloc[i]) else float("nan")
        exit_hit = (
            (np.isfinite(tr_stop_i) and px < tr_stop_i)
            or bars_held >= MAX_HOLD_BARS
        )
        if exit_hit:
            _state_set(ctx, {"in_pos": False, "entry_i": None, "last_exit_i": i})
            return {
                "action": "EXIT",
                "confidence": 0.75,
                "desired_exposure_frac": 0.0,
                "horizon_hours": 0,
                "reason": "compression_release_exit",
                "meta": {"strategy": "sg_vol_compression_release_v1", "px": px, "atr_ratio": atr_ratio_i, "bars_held": bars_held},
            }
        return {
            "action": "BUY",
            "confidence": 0.5,
            "desired_exposure_frac": MAX_EXPOSURE,
            "horizon_hours": MAX_HOLD_BARS,
            "reason": "hold_long",
            "meta": {"strategy": "sg_vol_compression_release_v1", "px": px, "atr_ratio": atr_ratio_i},
        }

    atr_ratio_prev = float(atr_ratio.iloc[i - 1]) if i >= 1 and pd.notna(atr_ratio.iloc[i - 1]) else float("nan")
    compression_on = atr_ratio_i <= COMPRESSION_RATIO_MAX
    breakout_up = np.isfinite(bh_i) and px > (bh_i * (1.0 + BREAKOUT_BUFFER))
    vol_confirm = np.isfinite(vol_ma_i) and np.isfinite(vol_i) and vol_i >= (VOL_CONFIRM_MULT * vol_ma_i)
    expansion_confirm = (
        np.isfinite(atr_ratio_i)
        and np.isfinite(atr_ratio_prev)
        and atr_ratio_i > 1.0
        and atr_ratio_i > atr_ratio_prev
    )
    cooldown_ok = (not isinstance(last_exit_i, int)) or ((i - int(last_exit_i)) >= COOLDOWN_BARS)
    if compression_on and breakout_up and vol_confirm and expansion_confirm and cooldown_ok:
        _state_set(ctx, {"in_pos": True, "entry_i": i, "last_exit_i": last_exit_i})
        return {
            "action": "BUY",
            "confidence": 0.72,
            "desired_exposure_frac": MAX_EXPOSURE,
            "horizon_hours": MAX_HOLD_BARS,
            "reason": "compression_release_breakout",
            "meta": {"strategy": "sg_vol_compression_release_v1", "px": px, "atr_ratio": atr_ratio_i, "breakout_high_prior": bh_i},
        }

    return {
        "action": "HOLD",
        "confidence": 0.2,
        "desired_exposure_frac": 0.0,
        "horizon_hours": 0,
        "reason": "no_entry",
        "meta": {"strategy": "sg_vol_compression_release_v1", "px": px, "atr_ratio": atr_ratio_i, "breakout_high_prior": bh_i},
    }

