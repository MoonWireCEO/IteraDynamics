"""
Behavioral exhaustion-reversal prototype.

Role in portfolio research:
- Targets panic/washout rebounds instead of trend continuation.
- Designed to monetize short-horizon behavioral overshoot and snapback.

Contract:
  generate_intent(df, ctx=None, closed_only=True) -> dict
Deterministic, closed-bar only, no lookahead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

RET_LOOKBACK = 4
DROP_THRESHOLD = -0.04
RSI_PERIOD = 7
RSI_OVERSOLD = 30.0
VOL_SPIKE_WINDOW = 20
VOL_SPIKE_MULT = 1.6
MAX_HOLD_BARS = 48
TRAIL_LOOKBACK = 8
MAX_EXPOSURE = 0.985
STATE_KEY = "_sg_exhaustion_reversal_v1"


def _pick_cols(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    def _pick(*names: str) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
        return None

    h = _pick("High", "high")
    l = _pick("Low", "low")
    c = _pick("Close", "close")
    v = _pick("Volume", "volume")
    if any(x is None for x in (h, l, c, v)):
        raise ValueError("Missing HLCV columns.")
    return h or "High", l or "Low", c or "Close", v or "Volume"


def _rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_dn = dn.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


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
        return {"action": "HOLD", "confidence": 0.0, "desired_exposure_frac": 0.0, "horizon_hours": 36, "reason": "no_data", "meta": {"strategy": "sg_exhaustion_reversal_v1"}}

    h_col, l_col, c_col, v_col = _pick_cols(df)
    high = pd.to_numeric(df[h_col], errors="coerce")
    low = pd.to_numeric(df[l_col], errors="coerce")
    close = pd.to_numeric(df[c_col], errors="coerce")
    volume = pd.to_numeric(df[v_col], errors="coerce").fillna(0.0)

    rsi = _rsi(close, RSI_PERIOD)
    ret_n = close.pct_change(RET_LOOKBACK)
    bar_abs_ret = close.pct_change().abs()
    rv = bar_abs_ret.rolling(VOL_SPIKE_WINDOW, min_periods=VOL_SPIKE_WINDOW).mean()
    vol_spike = bar_abs_ret / rv.replace(0.0, np.nan)
    vol_mean = volume.rolling(VOL_SPIKE_WINDOW, min_periods=VOL_SPIKE_WINDOW).mean()

    i = len(df) - 1
    px = float(close.iloc[i]) if pd.notna(close.iloc[i]) else float("nan")
    prev_px = float(close.iloc[i - 1]) if i >= 1 and pd.notna(close.iloc[i - 1]) else float("nan")
    rsi_i = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else float("nan")
    retn_i = float(ret_n.iloc[i]) if pd.notna(ret_n.iloc[i]) else float("nan")
    spike_i = float(vol_spike.iloc[i]) if pd.notna(vol_spike.iloc[i]) else float("nan")
    vol_i = float(volume.iloc[i]) if pd.notna(volume.iloc[i]) else float("nan")
    vol_ma_i = float(vol_mean.iloc[i]) if pd.notna(vol_mean.iloc[i]) else float("nan")

    st = _state_get(ctx)
    in_pos = bool(st.get("in_pos", False))
    entry_i = st.get("entry_i")

    if not (np.isfinite(px) and np.isfinite(rsi_i)):
        return {"action": "HOLD", "confidence": 0.0, "desired_exposure_frac": 0.0 if not in_pos else MAX_EXPOSURE, "horizon_hours": 36, "reason": "insufficient_history", "meta": {"strategy": "sg_exhaustion_reversal_v1", "in_pos": in_pos}}

    if in_pos:
        bars_held = (i - int(entry_i)) if isinstance(entry_i, int) else 0
        trail = low.rolling(TRAIL_LOOKBACK, min_periods=TRAIL_LOOKBACK).min().shift(1)
        trail_i = float(trail.iloc[i]) if pd.notna(trail.iloc[i]) else float("nan")
        mean_revert_target = close.rolling(12, min_periods=12).mean().shift(1)
        target_i = float(mean_revert_target.iloc[i]) if pd.notna(mean_revert_target.iloc[i]) else float("nan")
        exit_hit = (
            (np.isfinite(target_i) and px >= target_i)
            or (np.isfinite(trail_i) and px < trail_i)
            or bars_held >= MAX_HOLD_BARS
        )
        if exit_hit:
            _state_set(ctx, {"in_pos": False, "entry_i": None})
            return {
                "action": "EXIT",
                "confidence": 0.75,
                "desired_exposure_frac": 0.0,
                "horizon_hours": 0,
                "reason": "exhaustion_reversal_exit",
                "meta": {"strategy": "sg_exhaustion_reversal_v1", "px": px, "target": target_i, "bars_held": bars_held},
            }
        return {
            "action": "BUY",
            "confidence": 0.5,
            "desired_exposure_frac": MAX_EXPOSURE,
            "horizon_hours": MAX_HOLD_BARS,
            "reason": "hold_long",
            "meta": {"strategy": "sg_exhaustion_reversal_v1", "px": px, "rsi": rsi_i},
        }

    impulse_down = np.isfinite(retn_i) and retn_i <= DROP_THRESHOLD
    oversold = rsi_i <= RSI_OVERSOLD
    vol_event = (np.isfinite(spike_i) and spike_i >= VOL_SPIKE_MULT) or (
        np.isfinite(vol_ma_i) and np.isfinite(vol_i) and vol_i >= (1.2 * vol_ma_i)
    )
    prev_mid = float((high.iloc[i - 1] + low.iloc[i - 1]) / 2.0) if i >= 1 and pd.notna(high.iloc[i - 1]) and pd.notna(low.iloc[i - 1]) else float("nan")
    reversal_confirm = (
        (np.isfinite(prev_px) and px > (prev_px * 1.002))
        or (np.isfinite(prev_mid) and px > prev_mid)
    )

    if impulse_down and oversold and reversal_confirm and vol_event:
        _state_set(ctx, {"in_pos": True, "entry_i": i})
        return {
            "action": "BUY",
            "confidence": 0.72,
            "desired_exposure_frac": MAX_EXPOSURE,
            "horizon_hours": MAX_HOLD_BARS,
            "reason": "panic_rebound_entry",
            "meta": {"strategy": "sg_exhaustion_reversal_v1", "px": px, "ret_n": retn_i, "rsi": rsi_i, "spike": spike_i},
        }

    return {
        "action": "HOLD",
        "confidence": 0.2,
        "desired_exposure_frac": 0.0,
        "horizon_hours": 0,
        "reason": "no_entry",
        "meta": {"strategy": "sg_exhaustion_reversal_v1", "px": px, "ret_n": retn_i, "rsi": rsi_i, "spike": spike_i},
    }

