"""
Research-Only Directional Backtest Runner for Layer 2 Strategies
================================================================

Purpose
-------
Provide a research-safe harness that can evaluate long / flat / short strategy
logic without modifying the existing long-only backtest runner or any live
execution code.

Design principles
-----------------
- Deterministic, closed-bar walk-forward evaluation
- No broker / runtime dependencies
- Research-only: does NOT imply live short support
- Backward-compatible aliases for long-only actions
- Explicit directional action vocabulary

Directional contract
--------------------
Strategy module must implement:
    generate_intent(df, ctx, *, closed_only=True) -> dict

Required/expected dict keys:
    - action: ENTER_LONG | EXIT_LONG | ENTER_SHORT | EXIT_SHORT | HOLD | FLAT
      (aliases accepted: BUY/LONG/ENTER, SELL/CLOSE/EXIT, SHORT/ENTER_SHORT, COVER/BUY_TO_COVER)
    - confidence: float
    - desired_exposure_frac: float in [-1.0, +1.0]
    - horizon_hours: int
    - reason: str
    - meta: dict

Position model
--------------
- Positive qty  => long
- Negative qty  => short
- Exposure can be in [-1, +1]
- Cash ledger is updated explicitly on buys/sells so short sale proceeds remain
  in cash and buy-to-cover debits cash.

Notes
-----
- This file intentionally does NOT replace backtest_runner.py.
- The implementation assumes spot-like mark-to-market math for research. It is
  good for directional geometry tests, but not a substitute for production
  derivatives / funding / borrow modeling.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Path setup
# ---------------------------

def _setup_path() -> Path:
    this_file = Path(__file__).resolve()
    argus_dir = this_file.parent.parent.parent

    if argus_dir.name == "argus" and (argus_dir / "research").exists():
        if str(argus_dir) not in sys.path:
            sys.path.insert(0, str(argus_dir))
        return argus_dir

    for candidate in [Path.cwd() / "runtime" / "argus", Path.cwd()]:
        if (candidate / "research").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    return Path.cwd()


# ---------------------------
# Env helpers
# ---------------------------

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


# ---------------------------
# Data loading / strategy loading
# ---------------------------

def load_flight_recorder(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError("CSV must contain a Timestamp column")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required OHLC column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0.0

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return df


def load_strategy_func(module_path: str, func_name: str) -> Callable:
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(f"Module '{module_path}' has no attribute '{func_name}'")
    if not callable(fn):
        raise TypeError(f"'{module_path}.{func_name}' is not callable")
    return fn


# ---------------------------
# Action normalization
# ---------------------------

def _normalize_action(action: Optional[str]) -> str:
    if not action:
        return "HOLD"
    a = str(action).strip().upper()

    if a in ("ENTER_LONG", "BUY", "LONG", "ENTER"):
        return "ENTER_LONG"
    if a in ("EXIT_LONG", "SELL", "CLOSE", "EXIT"):
        return "EXIT_LONG"
    if a in ("ENTER_SHORT", "SHORT", "SELL_SHORT"):
        return "ENTER_SHORT"
    if a in ("EXIT_SHORT", "COVER", "BUY_TO_COVER"):
        return "EXIT_SHORT"
    if a in ("FLAT",):
        return "FLAT"
    if a in ("HOLD", "NONE", "NOOP"):
        return "HOLD"
    return "HOLD"


def _clip_exposure(x: float) -> float:
    try:
        return max(-1.0, min(1.0, float(x)))
    except Exception:
        return 0.0


# ---------------------------
# Metrics
# ---------------------------

def compute_sortino(returns: np.ndarray, risk_free: float = 0.0, ann_factor: float = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / ann_factor
    neg_returns = excess[excess < 0]
    if len(neg_returns) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt(np.mean(neg_returns ** 2))
    if downside_std < 1e-12:
        return float("inf") if excess.mean() > 0 else 0.0
    return float((excess.mean() / downside_std) * np.sqrt(ann_factor))


def compute_calmar(cagr: float, max_dd: float) -> float:
    if abs(max_dd) < 1e-12:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / abs(max_dd))


def compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity / np.maximum(peak, 1e-12)) - 1.0


# ---------------------------
# Backtest engine
# ---------------------------

def run_backtest_directional(
    df: pd.DataFrame,
    strategy_func: Callable,
    *,
    lookback: int = 200,
    initial_equity: float = 10000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    closed_only: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Directional walk-forward backtest.

    At each bar i (for i >= lookback):
      1. Call strategy_func(df[:i+1], ctx, closed_only=closed_only)
      2. Normalize directional action + desired exposure in [-1, +1]
      3. Rebalance signed position toward target exposure
      4. Mark-to-market and log long/short/flat behavior
    """
    n = len(df)
    if n <= lookback:
        raise ValueError(f"Insufficient data: {n} rows, need > {lookback}")

    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0

    equity = float(initial_equity)
    cash = float(initial_equity)
    pos_qty = 0.0  # signed quantity; negative means short

    rows: List[Dict[str, Any]] = []
    gross_history: List[float] = []
    net_history: List[float] = []
    long_history: List[float] = []
    short_history: List[float] = []
    long_flags: List[bool] = []
    short_flags: List[bool] = []
    flat_flags: List[bool] = []

    ctx = {
        "mode": "backtest_directional",
        "dry_run": True,
        "now_utc": datetime.now(timezone.utc),
    }

    for i in range(lookback, n):
        ts = df.loc[i, "Timestamp"]
        price = float(df.loc[i, "Close"])

        pos_value = pos_qty * price
        equity = cash + pos_value
        curr_expo = (pos_value / equity) if equity > 0 else 0.0

        df_slice = df.iloc[: i + 1].copy()
        try:
            intent = strategy_func(df_slice, ctx, closed_only=closed_only)
        except Exception as e:
            intent = {
                "action": "HOLD",
                "confidence": 0.0,
                "desired_exposure_frac": curr_expo,
                "horizon_hours": 0,
                "reason": f"strategy_error: {e}",
                "meta": {},
            }

        action = _normalize_action(intent.get("action"))
        desired_expo = _clip_exposure(intent.get("desired_exposure_frac") or 0.0)

        if action == "EXIT_LONG":
            target_expo = min(curr_expo, 0.0)  # flatten long; preserve short if already short
        elif action == "EXIT_SHORT":
            target_expo = max(curr_expo, 0.0)  # flatten short; preserve long if already long
        elif action == "FLAT":
            target_expo = 0.0
        elif action == "ENTER_LONG":
            target_expo = max(0.0, desired_expo)
        elif action == "ENTER_SHORT":
            target_expo = min(0.0, desired_expo)
        else:  # HOLD
            if abs(desired_expo) > 1e-12:
                target_expo = desired_expo
            else:
                target_expo = curr_expo

        target_expo = _clip_exposure(target_expo)

        target_pos_value = equity * target_expo
        target_qty = target_pos_value / price if price > 0 else 0.0
        delta_qty = target_qty - pos_qty
        cost_this_bar = 0.0
        rebalanced = False

        if abs(delta_qty * price) > 1.0:
            rebalanced = True
            if delta_qty > 0:
                # Buy more / buy to cover
                fill_price = price * (1.0 + slip_rate)
                gross = delta_qty * fill_price
                fee = gross * fee_rate
                cash -= (gross + fee)
                pos_qty += delta_qty
                cost_this_bar = (delta_qty * (fill_price - price)) + fee
            else:
                # Sell down / sell short
                qty_sell = abs(delta_qty)
                fill_price = price * (1.0 - slip_rate)
                gross = qty_sell * fill_price
                fee = gross * fee_rate
                cash += (gross - fee)
                pos_qty -= qty_sell
                cost_this_bar = (qty_sell * (price - fill_price)) + fee

        pos_value = pos_qty * price
        equity = cash + pos_value

        net_expo = (pos_value / equity) if equity > 0 else 0.0
        gross_expo = abs(net_expo)
        long_expo = max(net_expo, 0.0)
        short_expo = abs(min(net_expo, 0.0))

        gross_history.append(gross_expo)
        net_history.append(net_expo)
        long_history.append(long_expo)
        short_history.append(short_expo)
        long_flags.append(long_expo > 0.01)
        short_flags.append(short_expo > 0.01)
        flat_flags.append(gross_expo <= 0.01)

        rows.append(
            {
                "Timestamp": ts,
                "equity": equity,
                "cash": cash,
                "pos_qty": pos_qty,
                "pos_value": pos_value,
                "price": price,
                "net_exposure": net_expo,
                "gross_exposure": gross_expo,
                "long_exposure": long_expo,
                "short_exposure": short_expo,
                "desired_exposure_frac": desired_expo,
                "target_exposure": target_expo,
                "action": action,
                "fee_slippage_this_bar": cost_this_bar,
                "rebalanced": rebalanced,
            }
        )

    equity_df = pd.DataFrame(rows)
    eq = equity_df["equity"].to_numpy(dtype=float)
    if len(eq) > 1:
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    else:
        returns = np.array([0.0])

    total_return = (eq[-1] / initial_equity) - 1.0
    first_ts = equity_df["Timestamp"].iloc[0]
    last_ts = equity_df["Timestamp"].iloc[-1]
    years = max((last_ts - first_ts).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (eq[-1] / initial_equity) ** (1.0 / years) - 1.0
    dd_series = compute_drawdown_series(eq)
    max_dd = float(dd_series.min())
    calmar = compute_calmar(cagr, max_dd)
    sortino = compute_sortino(returns, risk_free=0.0, ann_factor=8760)

    metrics = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(abs(max_dd)),
        "calmar": float(calmar) if np.isfinite(calmar) else 0.0,
        "sortino": float(sortino) if np.isfinite(sortino) else 0.0,
        "avg_gross_exposure": float(np.mean(gross_history)) if gross_history else 0.0,
        "avg_net_exposure": float(np.mean(net_history)) if net_history else 0.0,
        "avg_long_exposure": float(np.mean(long_history)) if long_history else 0.0,
        "avg_short_exposure": float(np.mean(short_history)) if short_history else 0.0,
        "time_long": float(np.mean(long_flags)) if long_flags else 0.0,
        "time_short": float(np.mean(short_flags)) if short_flags else 0.0,
        "time_flat": float(np.mean(flat_flags)) if flat_flags else 0.0,
        "final_equity": float(eq[-1]),
        "bars": int(len(equity_df)),
        "years": float(years),
    }
    return equity_df, metrics


# ---------------------------
# Main
# ---------------------------

def main() -> Dict[str, Any]:
    argus_dir = _setup_path()

    module_path = _env_str("ARGUS_STRATEGY_MODULE", "research.strategies.sg_bear_trend_short_probe_v1")
    func_name = _env_str("ARGUS_STRATEGY_FUNC", "generate_intent")
    default_data = str(argus_dir / "flight_recorder.csv")
    data_file = _env_str("ARGUS_DATA_FILE", default_data)
    lookback = _env_int("ARGUS_LOOKBACK", 200)
    initial_equity = _env_float("ARGUS_INITIAL_EQUITY", 10000.0)
    fee_bps = _env_float("ARGUS_FEE_BPS", 10.0)
    slippage_bps = _env_float("ARGUS_SLIPPAGE_BPS", 5.0)

    print("=" * 60)
    print("DIRECTIONAL BACKTEST RUNNER (Research-Only)")
    print("=" * 60)
    print(f"Strategy module: {module_path}")
    print(f"Strategy func:   {func_name}")
    print(f"Data file:       {data_file}")
    print(f"Lookback:        {lookback}")
    print(f"Initial equity:  ${initial_equity:,.2f}")
    print(f"Fee (bps):       {fee_bps}")
    print(f"Slippage (bps):  {slippage_bps}")
    print("-" * 60)

    strategy_func = load_strategy_func(module_path, func_name)
    df = load_flight_recorder(data_file)
    equity_df, metrics = run_backtest_directional(
        df,
        strategy_func,
        lookback=lookback,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        closed_only=True,
    )

    repo_root = Path(__file__).resolve().parents[4]
    debug_dir = repo_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    trace_path = debug_dir / "harness_directional_trace.csv"
    equity_df.to_csv(trace_path, index=False)
    print(f"Trace written: {trace_path}")

    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Return:      {metrics['total_return'] * 100:>10.2f}%")
    print(f"CAGR:              {metrics['cagr'] * 100:>10.2f}%")
    print(f"Max Drawdown:      {metrics['max_drawdown'] * 100:>10.2f}%")
    print(f"Calmar:            {metrics['calmar']:>10.2f}")
    print(f"Sortino:           {metrics['sortino']:>10.2f}")
    print(f"Avg Gross Exp:     {metrics['avg_gross_exposure'] * 100:>10.2f}%")
    print(f"Avg Net Exp:       {metrics['avg_net_exposure'] * 100:>10.2f}%")
    print(f"Avg Long Exp:      {metrics['avg_long_exposure'] * 100:>10.2f}%")
    print(f"Avg Short Exp:     {metrics['avg_short_exposure'] * 100:>10.2f}%")
    print(f"Time Long:         {metrics['time_long'] * 100:>10.2f}%")
    print(f"Time Short:        {metrics['time_short'] * 100:>10.2f}%")
    print(f"Time Flat:         {metrics['time_flat'] * 100:>10.2f}%")
    print(f"Final Equity:      ${metrics['final_equity']:>10,.2f}")
    print(f"Bars:              {metrics['bars']:>10,}")
    print(f"Years:             {metrics['years']:>10.2f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
