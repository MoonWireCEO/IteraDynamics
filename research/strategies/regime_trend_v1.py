from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class RegimeTrendV1Params:
    # Regime filter (trend must be "on")
    sma_slow: int = 200
    sma_slow_slope_lookback: int = 24  # slope over last N bars (hourly => 24 = ~1 day)

    # Entry logic
    sma_fast: int = 20
    sma_mid: int = 50
    breakout_lookback: int = 168  # hourly => 168 = 7 days
    use_breakout: bool = True
    use_golden_cross: bool = True

    # Risk management
    hard_stop_pct: float = 0.10       # -10% from entry
    trail_stop_pct: float = 0.12      # -12% from peak since entry
    atr_len: int = 14
    atr_stop_mult: float = 0.0        # 0 disables ATR stop; e.g. 3.0 to enable
    exit_on_close_below_mid: bool = True  # close < SMA(mid) => exit

    # Position sizing
    target_fraction: float = 1.0      # deploy 100% of cash by default (backtester caps by cash)
    min_bars: int = 300               # safety: don’t trade too early


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    """
    Simple ATR using True Range rolling mean.
    Expects columns: High, Low, Close
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n, min_periods=n).mean()


def build_signals_regime_trend_v1(
    df: pd.DataFrame,
    params: Optional[RegimeTrendV1Params] = None,
) -> pd.DataFrame:
    """
    Returns signals DataFrame aligned 1:1 with df with columns:
      - enter_long: bool
      - exit_long: bool
      - target_usd: float (optional)
    """
    if params is None:
        params = RegimeTrendV1Params()

    out = pd.DataFrame(index=df.index)
    out["enter_long"] = False
    out["exit_long"] = False

    close = df["Close"].astype(float)

    # Indicators
    sma_slow = _sma(close, params.sma_slow)
    sma_fast = _sma(close, params.sma_fast)
    sma_mid = _sma(close, params.sma_mid)

    atr = _atr(df, params.atr_len) if params.atr_stop_mult and params.atr_stop_mult > 0 else None

    # Regime: price above slow SMA AND slow SMA rising
    slow_slope = sma_slow - sma_slow.shift(params.sma_slow_slope_lookback)
    regime_on = (close > sma_slow) & (slow_slope > 0)

    # Golden cross entry trigger (fast crosses above mid)
    fast_above = (sma_fast > sma_mid).astype(bool)
    prev_fast_above = fast_above.shift(1).fillna(False).astype("boolean").fillna(False).astype(bool)
    cross_up = fast_above & (~prev_fast_above)

    # Breakout entry trigger: close exceeds prior N-bar high (excluding current bar)
    breakout = pd.Series(False, index=df.index)
    if params.use_breakout:
        hh = close.shift(1).rolling(params.breakout_lookback, min_periods=params.breakout_lookback).max()
        breakout = (close > hh).fillna(False).astype(bool)

    # Exit condition baseline: close below mid SMA
    exit_trend_break = (close < sma_mid).fillna(False).astype(bool) if params.exit_on_close_below_mid else pd.Series(False, index=df.index)

    # Stateful trade management: enforce hard stop / trailing stop / optional ATR stop
    in_pos = False
    entry_px = np.nan
    peak_px = np.nan

    # We’ll write exits into out["exit_long"] as we go
    for i in range(len(df)):
        # Don’t act until indicators are available
        if i < params.min_bars:
            continue

        px = float(close.iloc[i])
        reg = bool(regime_on.iloc[i]) if not pd.isna(regime_on.iloc[i]) else False

        # Entry setup
        entry_signal = False
        if reg:
            if params.use_golden_cross and bool(cross_up.iloc[i]):
                entry_signal = True
            if params.use_breakout and bool(breakout.iloc[i]):
                entry_signal = True

        if not in_pos:
            if entry_signal:
                out.iat[i, out.columns.get_loc("enter_long")] = True
                in_pos = True
                entry_px = px
                peak_px = px
            continue

        # In position: update peak
        peak_px = max(peak_px, px)

        # Risk exits
        hard_stop = (px <= entry_px * (1.0 - params.hard_stop_pct)) if params.hard_stop_pct > 0 else False
        trail_stop = (px <= peak_px * (1.0 - params.trail_stop_pct)) if params.trail_stop_pct > 0 else False

        atr_stop = False
        if atr is not None and not pd.isna(atr.iloc[i]) and params.atr_stop_mult > 0:
            # ATR stop below peak (more conservative) or below entry (less sensitive).
            # Here: below peak to act like volatility-adjusted trailing stop.
            atr_stop_level = peak_px - float(atr.iloc[i]) * float(params.atr_stop_mult)
            atr_stop = (px <= atr_stop_level)

        # Trend break / regime off exits
        trend_break = bool(exit_trend_break.iloc[i]) if params.exit_on_close_below_mid else False
        regime_off = not reg

        if hard_stop or trail_stop or atr_stop or trend_break or regime_off:
            out.iat[i, out.columns.get_loc("exit_long")] = True
            in_pos = False
            entry_px = np.nan
            peak_px = np.nan

    # Optional sizing hint for backtester
    if params.target_fraction is not None and params.target_fraction > 0:
        # backtester will cap to available cash anyway; this is just a target
        out["target_usd"] = np.nan
        # runner will replace NaN with "cash" if needed; we can set a fraction placeholder
        # (leave NaN; runner can ignore or populate)

    return out


def params_dict(params: RegimeTrendV1Params) -> Dict[str, Any]:
    return {k: getattr(params, k) for k in params.__dataclass_fields__.keys()}
