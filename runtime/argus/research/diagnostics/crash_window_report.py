"""
crash_window_report.py

One-command crash-window diagnostics + peak->trough DD proxy.

What it does:
1) Loads a BTC hourly CSV (long dataset)
2) Slices a crash window (default 2021-07-01 to 2022-12-31)
3) Iterates bar-by-bar (growing history, closed_only=True) and records:
   - ts, price, regime, exposure, action, reason, macro_bull (if present)
4) Saves:
   - crash_window_diagnostics_*.csv
   - crash_window_worst_dd_segment_*.csv (peak->trough segment)
5) Prints:
   - basic exposure stats
   - time exposure > 0.5
   - peak->trough DD proxy (exposure-weighted, no fees/slippage)
   - regime mix + exposure stats for peak->trough segment

Design notes:
- Deterministic
- No external deps beyond pandas/numpy (already in your stack)
- Does not touch Layer 1 or Prime; calls:
    classify_regime(sub, closed_only=True)
    strategy.generate_intent(sub, {}, closed_only=True)

Usage (repo root):
  python .\runtime\argus\research\diagnostics\crash_window_report.py ^
      --data data\btcusd_3600s_2019-01-01_to_2025-12-30.csv ^
      --strategy research.strategies.sg_core_exposure_v2 ^
      --start 2021-07-01 --end 2022-12-31

Or use env injection:
  $env:ARGUS_STRATEGY_MODULE="research.strategies.sg_core_exposure_v2"
  $env:ARGUS_STRATEGY_FUNC="generate_intent"
  python .\runtime\argus\research\diagnostics\crash_window_report.py --data <...>
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Bootstrapping sys.path
# ---------------------------

def _ensure_runtime_argus_on_path() -> None:
    """
    Allow running this file directly from repo root with:
      python runtime/argus/research/diagnostics/crash_window_report.py

    Adds ./runtime/argus to sys.path so imports like:
      from research.regime import classify_regime
    work consistently.
    """
    here = Path(__file__).resolve()
    # .../runtime/argus/research/diagnostics/crash_window_report.py
    runtime_argus = here.parents[2]  # .../runtime/argus
    if str(runtime_argus) not in sys.path:
        sys.path.insert(0, str(runtime_argus))


_ensure_runtime_argus_on_path()

from research.regime import classify_regime  # noqa: E402


# ---------------------------
# Data loading helpers
# ---------------------------

def _pick_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None


def _load_ohlcv_csv(path: str) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)

    ts_col = _pick_col(df, "timestamp", "time", "date", "datetime")
    close_col = _pick_col(df, "close", "c", "price", "last", "mid")

    if ts_col is None:
        raise ValueError(f"Missing Timestamp column. Found columns: {list(df.columns)}")
    if close_col is None:
        raise ValueError(f"Missing Close column. Found columns: {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # Ensure standard OHLCV column names exist if present
    open_col = _pick_col(df, "open", "o")
    high_col = _pick_col(df, "high", "h")
    low_col = _pick_col(df, "low", "l")
    vol_col = _pick_col(df, "volume", "vol", "v")

    # Create canonical columns without deleting originals
    df["Timestamp"] = df[ts_col]
    df["Close"] = df[close_col].astype(float)

    if open_col is not None:
        df["Open"] = df[open_col].astype(float)
    if high_col is not None:
        df["High"] = df[high_col].astype(float)
    if low_col is not None:
        df["Low"] = df[low_col].astype(float)
    if vol_col is not None:
        df["Volume"] = df[vol_col].astype(float)

    return df, "Timestamp", "Close"


# ---------------------------
# Strategy loading
# ---------------------------

@dataclass(frozen=True)
class StrategySpec:
    module: str
    func: str


def _load_strategy(spec: StrategySpec):
    mod = importlib.import_module(spec.module)
    fn = getattr(mod, spec.func, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Strategy function not found/callable: {spec.module}.{spec.func}")
    return fn


def _default_strategy_spec_from_env() -> StrategySpec:
    module = str(os.getenv("ARGUS_STRATEGY_MODULE", "")).strip() or "research.strategies.sg_core_exposure_v2"
    func = str(os.getenv("ARGUS_STRATEGY_FUNC", "")).strip() or "generate_intent"
    return StrategySpec(module=module, func=func)


# ---------------------------
# Core diagnostics
# ---------------------------

def _extract_macro_bull(out: Dict[str, Any]) -> Optional[bool]:
    meta = out.get("meta", {}) if isinstance(out, dict) else {}
    macro = meta.get("macro", {}) if isinstance(meta, dict) else {}
    mb = macro.get("macro_bull", None) if isinstance(macro, dict) else None
    if mb is None:
        return None
    return bool(mb)


def run_crash_window_diagnostics(
    *,
    data_path: str,
    strategy_spec: StrategySpec,
    start: str,
    end: str,
    lookback: int,
    out_csv: str,
) -> pd.DataFrame:
    df, ts_col, close_col = _load_ohlcv_csv(data_path)

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    win = df[(df[ts_col] >= start_dt) & (df[ts_col] <= end_dt)].reset_index(drop=True)
    if len(win) < (lookback + 5):
        raise ValueError(f"Crash window too small: len={len(win)} (need > {lookback + 5})")

    generate_intent = _load_strategy(strategy_spec)

    records = []
    # Growing-history loop (matches your earlier forensic methodology)
    for i in range(lookback, len(win)):
        sub = win.iloc[:i]

        reg = classify_regime(sub, closed_only=True)
        out = generate_intent(sub, {}, closed_only=True)

        records.append(
            {
                "ts": sub[ts_col].iloc[-1].to_pydatetime(),
                "price": float(sub[close_col].iloc[-1]),
                "regime": getattr(reg, "label", None),
                "exposure": float(out.get("desired_exposure_frac", 0.0)),
                "action": out.get("action", None),
                "reason": out.get("reason", None),
                "macro_bull": _extract_macro_bull(out),
            }
        )

    res = pd.DataFrame(records).sort_values("ts").reset_index(drop=True)
    res.to_csv(out_csv, index=False)
    return res


def compute_peak_trough_dd_proxy(res: pd.DataFrame) -> Dict[str, Any]:
    """
    Exposure-weighted equity curve proxy:
      strat_ret[t] = exposure[t-1] * price_ret[t]
    No fees/slippage (this is forensic, not official harness).
    """
    df = res.copy()
    df["ret"] = df["price"].pct_change().fillna(0.0)
    df["strat_ret"] = df["exposure"].shift(1).fillna(0.0) * df["ret"]
    df["equity"] = (1.0 + df["strat_ret"]).cumprod()
    roll_max = df["equity"].cummax()
    df["dd"] = df["equity"] / roll_max - 1.0

    i_trough = int(df["dd"].idxmin())
    peak_idx = int(df.loc[:i_trough, "equity"].idxmax())

    peak_ts = df.loc[peak_idx, "ts"]
    trough_ts = df.loc[i_trough, "ts"]

    out = {
        "peak_ts": peak_ts,
        "peak_equity": float(df.loc[peak_idx, "equity"]),
        "trough_ts": trough_ts,
        "trough_equity": float(df.loc[i_trough, "equity"]),
        "dd": float(df.loc[i_trough, "dd"]),
        "df": df,
        "peak_idx": peak_idx,
        "trough_idx": i_trough,
    }
    return out


def _print_basic_summary(res: pd.DataFrame) -> None:
    print("\n=== BASIC SUMMARY ===")
    print("Max exposure:", float(res["exposure"].max()))
    print("Avg exposure:", float(res["exposure"].mean()))
    print("Time exposed >0:", float((res["exposure"] > 0).mean()))
    print("Time exposure >0.5:", float((res["exposure"] > 0.5).mean()))


def _print_peak_trough_summary(dd_out: Dict[str, Any], seg: pd.DataFrame) -> None:
    print("\n=== PEAK -> TROUGH (DD PROXY, no fees) ===")
    print("peak:", dd_out["peak_ts"], "equity:", dd_out["peak_equity"])
    print("trough:", dd_out["trough_ts"], "equity:", dd_out["trough_equity"])
    print("DD:", dd_out["dd"])

    print("\nRegime mix during peak->trough segment:")
    print(seg["regime"].value_counts(normalize=True).head(10))

    print("\nAvg exposure during segment:", float(seg["exposure"].mean()))
    print("Max exposure during segment:", float(seg["exposure"].max()))
    print("Time exposure >0.5 during segment:", float((seg["exposure"] > 0.5).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV path (hourly OHLCV)")
    ap.add_argument("--strategy", default=os.getenv("ARGUS_STRATEGY_MODULE", ""), help="Strategy module (env: ARGUS_STRATEGY_MODULE)")
    ap.add_argument("--func", default=os.getenv("ARGUS_STRATEGY_FUNC", "generate_intent"), help="Strategy function (env: ARGUS_STRATEGY_FUNC)")
    ap.add_argument("--start", default="2021-07-01", help="Crash window start (UTC)")
    ap.add_argument("--end", default="2022-12-31", help="Crash window end (UTC)")
    ap.add_argument("--lookback", type=int, default=200, help="Warmup bars before first evaluation")
    ap.add_argument("--out", default="", help="Output CSV base name (optional). Default auto-named.")
    args = ap.parse_args()

    spec = _default_strategy_spec_from_env()
    if args.strategy.strip():
        spec = StrategySpec(module=args.strategy.strip(), func=args.func.strip() or "generate_intent")
    else:
        spec = StrategySpec(module=spec.module, func=args.func.strip() or spec.func)

    # Auto output names
    out_base = args.out.strip()
    if not out_base:
        safe_mod = spec.module.split(".")[-1]
        out_base = f"crash_window_diagnostics_{safe_mod}"

    out_csv = f"{out_base}.csv"
    out_seg_csv = f"{out_base}_worst_dd_segment.csv"

    print("============================================================")
    print("CRASH WINDOW REPORT")
    print("============================================================")
    print("Data:", args.data)
    print("Strategy:", f"{spec.module}.{spec.func}")
    print("Window:", args.start, "to", args.end)
    print("Lookback:", args.lookback)
    print("Output:", out_csv)
    print("------------------------------------------------------------")

    res = run_crash_window_diagnostics(
        data_path=args.data,
        strategy_spec=spec,
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        out_csv=out_csv,
    )

    _print_basic_summary(res)
    print("\nSaved:", out_csv)

    dd_out = compute_peak_trough_dd_proxy(res)
    df = dd_out["df"]
    peak_ts = dd_out["peak_ts"]
    trough_ts = dd_out["trough_ts"]

    seg = df[(df["ts"] >= peak_ts) & (df["ts"] <= trough_ts)].copy()
    seg.to_csv(out_seg_csv, index=False)
    _print_peak_trough_summary(dd_out, seg)
    print("\nSaved:", out_seg_csv)

    # Optional: macro_bull prevalence in 2022 if column exists
    if "macro_bull" in res.columns and res["macro_bull"].notna().any():
        r2022 = res[(res["ts"] >= "2022-01-01") & (res["ts"] <= "2022-12-31")].copy()
        if len(r2022) > 0:
            mb = r2022["macro_bull"].fillna(False).astype(bool)
            print("\n2022 macro_bull %:", float(mb.mean()))
            print("2022 time exposure >0.5 %:", float((r2022["exposure"] > 0.5).mean()))


if __name__ == "__main__":
    main()
