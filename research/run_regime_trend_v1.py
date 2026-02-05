from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from research.engine.backtest_core import (
    BacktestConfig,
    load_flight_recorder,
    run_backtest_long_only,
    buy_and_hold_baseline,
)

from research.strategies.regime_trend_v1 import (
    RegimeTrendV1Params,
    build_signals_regime_trend_v1,
    params_dict,
)


RESULTS_DIR = os.path.join("research", "backtests", "results")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _print_summary(title: str, summary: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:>18}: {v}")
        else:
            print(f"{k:>18}: {v}")


def _walk_forward_slices(df: pd.DataFrame, test_days: int = 90, step_days: int = 30, warmup_days: int = 365) -> List[Tuple[int, int]]:
    """
    Simple walk-forward slicing for evaluation (no parameter fitting).
    - warmup_days: skip first N days so indicators stabilize
    - Each fold is [start_idx, end_idx) for a test window
    """
    ts = pd.to_datetime(df["Timestamp"], utc=True)
    start_time = ts.min() + pd.Timedelta(days=warmup_days)
    end_time = ts.max()

    slices = []
    cur_start = start_time
    while cur_start + pd.Timedelta(days=test_days) <= end_time:
        cur_end = cur_start + pd.Timedelta(days=test_days)
        i0 = int(ts.searchsorted(cur_start))
        i1 = int(ts.searchsorted(cur_end))
        if i1 - i0 > 0:
            slices.append((i0, i1))
        cur_start = cur_start + pd.Timedelta(days=step_days)
    return slices


def run_single_backtest(df: pd.DataFrame, params: RegimeTrendV1Params, cfg: BacktestConfig):
    sig = build_signals_regime_trend_v1(df, params)

    # Optional target sizing: allocate a fraction of cash on entry
    # Our backtester expects target_usd per-row; simplest is to omit it.
    # If you want it: uncomment below and make it explicit.
    # sig["target_usd"] = np.where(sig["enter_long"].values, cfg.initial_cash * params.target_fraction, np.nan)

    res = run_backtest_long_only(df, sig, cfg)
    return res, sig


def run_walk_forward(df: pd.DataFrame, params: RegimeTrendV1Params, cfg: BacktestConfig) -> Dict[str, Any]:
    folds = _walk_forward_slices(df, test_days=90, step_days=30, warmup_days=365)
    fold_summaries = []

    for (i0, i1) in folds:
        dfi = df.iloc[i0:i1].reset_index(drop=True)
        res, _ = run_single_backtest(dfi, params, cfg)
        s = res.summary.copy()
        s["start"] = str(dfi["Timestamp"].iloc[0])
        s["end"] = str(dfi["Timestamp"].iloc[-1])
        fold_summaries.append(s)

    if not fold_summaries:
        return {"folds": 0, "note": "Not enough data for walk-forward slices."}

    # Aggregate: median fold return and worst fold DD, plus % profitable folds
    rets = np.array([fs["total_return_pct"] for fs in fold_summaries], dtype=float)
    dds = np.array([fs["max_drawdown_pct"] for fs in fold_summaries], dtype=float)

    profitable = float((rets > 0).mean() * 100.0)

    return {
        "folds": int(len(fold_summaries)),
        "profitable_folds_pct": profitable,
        "median_fold_return_pct": float(np.median(rets)),
        "mean_fold_return_pct": float(np.mean(rets)),
        "worst_fold_drawdown_pct": float(np.min(dds)),  # most negative
        "best_fold_return_pct": float(np.max(rets)),
    }


def main():
    _ensure_dir(RESULTS_DIR)

    # ---- Config (costs matter; keep conservative) ----
    cfg = BacktestConfig(
        fee_bps=6.0,
        slippage_bps=10.0,
        initial_cash=100.0,
        trade_on_close=True,
        min_notional_usd=5.0,
    )

    # ---- Strategy Params (baseline) ----
    params = RegimeTrendV1Params(
        sma_slow=200,
        sma_slow_slope_lookback=24,
        sma_fast=20,
        sma_mid=50,
        breakout_lookback=168,
        use_breakout=True,
        use_golden_cross=True,
        hard_stop_pct=0.10,
        trail_stop_pct=0.12,
        atr_len=14,
        atr_stop_mult=0.0,  # enable later if needed
        exit_on_close_below_mid=True,
        target_fraction=1.0,
        min_bars=300,
    )

    # ---- Load data ----
    csv_path = os.path.join("research", "backtests", "data", "flight_recorder.csv")

    df = load_flight_recorder(csv_path)

    # ---- Run backtest ----
    res, sig = run_single_backtest(df, params, cfg)
    _print_summary("REGIME_TREND_V1 SUMMARY", res.summary)

    bh = buy_and_hold_baseline(df, cfg)
    _print_summary("BUY & HOLD (costed)", bh)

    # ---- Save outputs ----
    trades_path = os.path.join(RESULTS_DIR, "regime_trend_v1_trades.csv")
    equity_path = os.path.join(RESULTS_DIR, "regime_trend_v1_equity.csv")
    params_path = os.path.join(RESULTS_DIR, "regime_trend_v1_params.json")

    res.trades.to_csv(trades_path, index=False)
    res.equity_curve.to_csv(equity_path, index=False)

    # Save params for provenance
    try:
        import json
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump({"cfg": asdict(cfg), "params": params_dict(params)}, f, indent=2)
    except Exception:
        pass

    print("\nSaved:")
    print(f" - {trades_path}")
    print(f" - {equity_path}")
    print(f" - {params_path}")

    # ---- Optional: Walk-forward sanity check ----
    wf = run_walk_forward(df, params, cfg)
    _print_summary("WALK-FORWARD (no tuning, fixed rules)", wf)


if __name__ == "__main__":
    main()
