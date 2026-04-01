"""
Trade-level diagnostics for harness output CSVs.

Purpose:
- Convert bar-level backtest output into trade segments.
- Bucket trades by entry regime (trend + volatility) to identify failure modes.

Input expected from run_vb_harness_asset.py output:
  Timestamp, equity, exposure, price
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"Timestamp", "equity", "exposure", "price"}
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    out = df.copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out["exposure"] = pd.to_numeric(out["exposure"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["Timestamp", "equity", "exposure", "price"]).sort_values("Timestamp").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows after parsing input CSV.")
    return out


def _trade_segments(exposure: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    in_trade = exposure > threshold
    segs: List[Tuple[int, int]] = []
    start = None
    for i, on in enumerate(in_trade):
        if start is None and on:
            start = i
        elif start is not None and not on:
            if i > start:
                segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(in_trade) - 1))
    return segs


def _compute_regimes(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    out = df.copy()
    out["ema200"] = out["price"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["ema200_prev"] = out["ema200"].shift(1)
    out["ret1"] = out["price"].pct_change()
    out["rv"] = out["ret1"].rolling(vol_window, min_periods=vol_window).std()
    out["rv_med"] = out["rv"].rolling(500, min_periods=100).median()

    def _trend_bucket(r: pd.Series) -> str:
        if not np.isfinite(r["ema200"]):
            return "unknown"
        px = r["price"]
        ema = r["ema200"]
        ema_prev = r["ema200_prev"]
        if np.isfinite(ema_prev) and px >= ema and ema >= ema_prev:
            return "trend_up"
        if np.isfinite(ema_prev) and px < ema and ema < ema_prev:
            return "trend_down"
        if px >= ema:
            return "above_ema_mixed"
        return "below_ema_mixed"

    def _vol_bucket(r: pd.Series) -> str:
        rv = r["rv"]
        med = r["rv_med"]
        if not np.isfinite(rv):
            return "unknown"
        if np.isfinite(med):
            if rv >= 1.25 * med:
                return "high_vol"
            if rv <= 0.80 * med:
                return "low_vol"
        return "normal_vol"

    out["trend_bucket"] = out.apply(_trend_bucket, axis=1)
    out["vol_bucket"] = out.apply(_vol_bucket, axis=1)
    return out


def _build_trade_table(df: pd.DataFrame, segs: List[Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for idx, (s, e) in enumerate(segs, start=1):
        eq0 = float(df["equity"].iloc[s])
        eq1 = float(df["equity"].iloc[e])
        ret = (eq1 / eq0 - 1.0) if eq0 != 0 else np.nan
        rows.append(
            {
                "trade_id": idx,
                "entry_idx": s,
                "exit_idx": e,
                "entry_ts": df["Timestamp"].iloc[s],
                "exit_ts": df["Timestamp"].iloc[e],
                "bars_held": int(max(e - s, 0)),
                "entry_price": float(df["price"].iloc[s]),
                "exit_price": float(df["price"].iloc[e]),
                "entry_equity": eq0,
                "exit_equity": eq1,
                "trade_return": float(ret),
                "entry_trend_bucket": df["trend_bucket"].iloc[s],
                "entry_vol_bucket": df["vol_bucket"].iloc[s],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze trade outcomes by entry regime buckets.")
    ap.add_argument("--input_csv", type=str, required=True, help="Path to run_vb_harness_asset output CSV.")
    ap.add_argument("--out_summary_csv", type=str, default="", help="Output path for regime summary CSV.")
    ap.add_argument("--out_trades_csv", type=str, default="", help="Output path for per-trade CSV.")
    ap.add_argument("--exposure_threshold", type=float, default=0.01, help="Exposure threshold for in-trade detection.")
    ap.add_argument("--vol_window", type=int, default=24, help="Rolling window for realized vol proxy.")
    ap.add_argument("--worst_n", type=int, default=10, help="How many worst trades to print.")
    args = ap.parse_args()

    in_path = Path(args.input_csv).expanduser()
    df = _compute_regimes(_load_results(in_path), vol_window=int(args.vol_window))
    segs = _trade_segments(df["exposure"].to_numpy(dtype=float), threshold=float(args.exposure_threshold))
    trades = _build_trade_table(df, segs)
    if trades.empty:
        print("No trades found with current exposure threshold.")
        return

    summary = (
        trades.groupby(["entry_trend_bucket", "entry_vol_bucket"], as_index=False)
        .agg(
            trades=("trade_id", "size"),
            win_rate=("trade_return", lambda x: float(np.mean(np.array(x) > 0))),
            avg_trade_return=("trade_return", "mean"),
            median_trade_return=("trade_return", "median"),
            worst_trade_return=("trade_return", "min"),
            best_trade_return=("trade_return", "max"),
            avg_bars_held=("bars_held", "mean"),
        )
        .sort_values(["avg_trade_return", "trades"], ascending=[False, False])
        .reset_index(drop=True)
    )

    out_summary = Path(args.out_summary_csv).expanduser() if args.out_summary_csv else in_path.with_name(f"{in_path.stem}_regime_summary.csv")
    out_trades = Path(args.out_trades_csv).expanduser() if args.out_trades_csv else in_path.with_name(f"{in_path.stem}_trades.csv")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_trades.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    trades.to_csv(out_trades, index=False)

    print("=" * 64)
    print("TRADE REGIME DIAGNOSTICS")
    print("=" * 64)
    print(f"Input: {in_path}")
    print(f"Trades: {len(trades)}")
    print(f"Wrote summary: {out_summary}")
    print(f"Wrote trades:  {out_trades}")
    print("-" * 64)
    print("Top regime buckets by avg trade return:")
    for _, r in summary.head(8).iterrows():
        print(
            f"{r['entry_trend_bucket']} | {r['entry_vol_bucket']} | "
            f"trades={int(r['trades'])} | win={float(r['win_rate']) * 100:5.1f}% | "
            f"avg={float(r['avg_trade_return']) * 100:6.2f}%"
        )
    print("-" * 64)
    print(f"Worst {max(1, int(args.worst_n))} trades:")
    worst = trades.sort_values("trade_return", ascending=True).head(max(1, int(args.worst_n)))
    for _, r in worst.iterrows():
        print(
            f"id={int(r['trade_id'])} | {r['entry_ts']} -> {r['exit_ts']} | "
            f"ret={float(r['trade_return']) * 100:7.2f}% | "
            f"{r['entry_trend_bucket']} / {r['entry_vol_bucket']}"
        )


if __name__ == "__main__":
    main()

