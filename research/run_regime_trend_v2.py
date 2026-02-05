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

from research.strategies.regime_trend_v2 import (
    RegimeTrendV2Params,
    build_signals_regime_trend_v2,
    params_dict,
)

RESULTS_DIR = os.path.join("research", "backtests", "results")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _print_summary(title: str, summary: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")


def _walk_forward_slices(
    df: pd.DataFrame, test_days: int = 90, step_days: int = 30, warmup_days: int = 365
) -> List[Tuple[int, int]]:
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


def run_single_backtest(df: pd.DataFrame, params: RegimeTrendV2Params, cfg: BacktestConfig):
    sig = build_signals_regime_trend_v2(df, params)
    res = run_backtest_long_only(df, sig, cfg)
    return res, sig


def run_walk_forward(df: pd.DataFrame, params: RegimeTrendV2Params, cfg: BacktestConfig) -> Dict[str, Any]:
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

    rets = np.array([fs["total_return_pct"] for fs in fold_summaries], dtype=float)
    dds = np.array([fs["max_drawdown_pct"] for fs in fold_summaries], dtype=float)

    return {
        "folds": int(len(fold_summaries)),
        "profitable_folds_pct": float((rets > 0).mean() * 100.0),
        "median_fold_return_pct": float(np.median(rets)),
        "mean_fold_return_pct": float(np.mean(rets)),
        "worst_fold_drawdown_pct": float(np.min(dds)),
        "best_fold_return_pct": float(np.max(rets)),
    }


def main():
    _ensure_dir(RESULTS_DIR)

    cfg = BacktestConfig(
        fee_bps=6.0,
        slippage_bps=10.0,
        initial_cash=100.0,
        trade_on_close=True,
        min_notional_usd=5.0,
    )

    params = RegimeTrendV2Params()  # keep defaults in the strategy file

    csv_path = os.path.join("research", "backtests", "data", "flight_recorder.csv")
    df = load_flight_recorder(csv_path)

    res, _ = run_single_backtest(df, params, cfg)
    _print_summary("REGIME_TREND_V2 SUMMARY", res.summary)

    bh = buy_and_hold_baseline(df, cfg)
    _print_summary("BUY & HOLD (costed)", bh)

    trades_path = os.path.join(RESULTS_DIR, "regime_trend_v2_trades.csv")
    equity_path = os.path.join(RESULTS_DIR, "regime_trend_v2_equity.csv")
    params_path = os.path.join(RESULTS_DIR, "regime_trend_v2_params.json")

    res.trades.to_csv(trades_path, index=False)
    res.equity_curve.to_csv(equity_path, index=False)

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

    wf = run_walk_forward(df, params, cfg)
    _print_summary("WALK-FORWARD (no tuning, fixed rules)", wf)


if __name__ == "__main__":
    main()
