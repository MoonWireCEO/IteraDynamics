"""
Concave mean-reversion prototype: VWAP stretch + RSI confirmation.

Role in portfolio research:
- Behaviorally distinct from BTC VB trend breakout.
- Attempts to monetize oversold dislocations and reversion to a fair-value anchor.

Contract:
  generate_intent(df, ctx=None, closed_only=True) -> dict
Deterministic, closed-bar only, no lookahead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

VWAP_WINDOW = 48
ENTRY_BAND = 0.025
RSI_PERIOD = 14
OVERSOLD_THRESHOLD = 30.0
STOP_LOSS_PCT = 0.03
MAX_HOLD_BARS = 18
EMA_GUARD_LEN = 200
MAX_EXPOSURE = 0.985
VOL_WINDOW = 24
VOL_MED_WINDOW = 500
VOL_HIGH_MULT = 1.25
VOL_LOW_MULT = 0.80
STATE_KEY = "_sg_mean_reversion_vwap_rsi_v1"


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


def _rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    pv = tp * volume
    pv_sum = pv.rolling(window=window, min_periods=window).sum()
    v_sum = volume.rolling(window=window, min_periods=window).sum()
    return pv_sum / v_sum.replace(0.0, np.nan)


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


def _vol_bucket(close: pd.Series) -> pd.Series:
    """
    Classify realized-vol regime per bar using only historical data:
      high_vol if rv >= 1.25 * rolling-median(rv)
      low_vol  if rv <= 0.80 * rolling-median(rv)
      else normal_vol
    """
    ret1 = close.pct_change()
    rv = ret1.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    rv_med = rv.rolling(VOL_MED_WINDOW, min_periods=100).median()
    out = pd.Series("unknown", index=close.index, dtype=object)
    high_mask = rv.notna() & rv_med.notna() & (rv >= (VOL_HIGH_MULT * rv_med))
    low_mask = rv.notna() & rv_med.notna() & (rv <= (VOL_LOW_MULT * rv_med))
    normal_mask = rv.notna() & ~high_mask & ~low_mask
    out.loc[high_mask] = "high_vol"
    out.loc[low_mask] = "low_vol"
    out.loc[normal_mask] = "normal_vol"
    return out


def generate_intent(df: pd.DataFrame, ctx: Any = None, *, closed_only: bool = True, **kwargs: Any) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {"action": "HOLD", "confidence": 0.0, "desired_exposure_frac": 0.0, "horizon_hours": 48, "reason": "no_data", "meta": {"strategy": "sg_mean_reversion_vwap_rsi_v1"}}

    o_col, h_col, l_col, c_col, v_col = _pick_cols(df)
    high = pd.to_numeric(df[h_col], errors="coerce")
    low = pd.to_numeric(df[l_col], errors="coerce")
    close = pd.to_numeric(df[c_col], errors="coerce")
    volume = pd.to_numeric(df[v_col], errors="coerce").fillna(0.0)

    vwap = _rolling_vwap(high, low, close, volume, VWAP_WINDOW)
    rsi = _rsi(close, RSI_PERIOD)
    ema_guard = close.ewm(span=EMA_GUARD_LEN, adjust=False, min_periods=EMA_GUARD_LEN).mean()
    vol_bucket = _vol_bucket(close)

    i = len(df) - 1
    px = float(close.iloc[i]) if pd.notna(close.iloc[i]) else float("nan")
    vwap_i = float(vwap.iloc[i]) if pd.notna(vwap.iloc[i]) else float("nan")
    rsi_i = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else float("nan")
    ema_i = float(ema_guard.iloc[i]) if pd.notna(ema_guard.iloc[i]) else float("nan")
    ema_prev = float(ema_guard.iloc[i - 1]) if i >= 1 and pd.notna(ema_guard.iloc[i - 1]) else float("nan")
    vol_bucket_i = str(vol_bucket.iloc[i]) if i >= 0 else "unknown"

    st = _state_get(ctx)
    in_pos = bool(st.get("in_pos", False))
    entry_px = st.get("entry_px")
    entry_i = st.get("entry_i")

    if not (np.isfinite(px) and np.isfinite(vwap_i) and np.isfinite(rsi_i)):
        return {"action": "HOLD", "confidence": 0.0, "desired_exposure_frac": 0.0 if not in_pos else MAX_EXPOSURE, "horizon_hours": 48, "reason": "insufficient_history", "meta": {"strategy": "sg_mean_reversion_vwap_rsi_v1", "in_pos": in_pos}}

    if in_pos:
        bars_held = (i - int(entry_i)) if isinstance(entry_i, int) else 0
        stop = float(entry_px) * (1.0 - STOP_LOSS_PCT) if isinstance(entry_px, (int, float)) else float("nan")
        exit_hit = (
            px >= vwap_i
            or (np.isfinite(stop) and px <= stop)
            or bars_held >= MAX_HOLD_BARS
        )
        if exit_hit:
            _state_set(ctx, {"in_pos": False, "entry_px": None, "entry_i": None})
            return {
                "action": "EXIT",
                "confidence": 0.75,
                "desired_exposure_frac": 0.0,
                "horizon_hours": 0,
                "reason": "mean_reversion_exit",
                "meta": {"strategy": "sg_mean_reversion_vwap_rsi_v1", "px": px, "vwap": vwap_i, "rsi": rsi_i, "bars_held": bars_held},
            }
        return {
            "action": "BUY",
            "confidence": 0.5,
            "desired_exposure_frac": MAX_EXPOSURE,
            "horizon_hours": MAX_HOLD_BARS,
            "reason": "hold_long",
            "meta": {"strategy": "sg_mean_reversion_vwap_rsi_v1", "px": px, "vwap": vwap_i, "rsi": rsi_i, "vol_bucket": vol_bucket_i},
        }

    ema_slope_ok = (not np.isfinite(ema_i)) or (not np.isfinite(ema_prev)) or (ema_i >= ema_prev)
    regime_ok = ((not np.isfinite(ema_i)) or (px >= ema_i)) and ema_slope_ok
    # v4: keep only low-vol mean-reversion entries; high-vol caused tail losses in diagnostics.
    vol_gate_ok = vol_bucket_i == "low_vol"
    stretch_ok = px <= vwap_i * (1.0 - ENTRY_BAND)
    rsi_ok = rsi_i <= OVERSOLD_THRESHOLD
    if regime_ok and vol_gate_ok and stretch_ok and rsi_ok:
        _state_set(ctx, {"in_pos": True, "entry_px": px, "entry_i": i})
        return {
            "action": "BUY",
            "confidence": 0.7,
            "desired_exposure_frac": MAX_EXPOSURE,
            "horizon_hours": MAX_HOLD_BARS,
            "reason": "vwap_stretch_rsi_oversold",
            "meta": {"strategy": "sg_mean_reversion_vwap_rsi_v1", "px": px, "vwap": vwap_i, "rsi": rsi_i, "ema_guard": ema_i, "vol_bucket": vol_bucket_i},
        }

    return {
        "action": "HOLD",
        "confidence": 0.2,
        "desired_exposure_frac": 0.0,
        "horizon_hours": 0,
        "reason": "no_entry",
        "meta": {"strategy": "sg_mean_reversion_vwap_rsi_v1", "px": px, "vwap": vwap_i, "rsi": rsi_i, "ema_guard": ema_i, "vol_bucket": vol_bucket_i, "vol_gate_ok": bool(vol_gate_ok)},
    }

