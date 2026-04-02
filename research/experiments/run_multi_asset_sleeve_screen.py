"""
Run multi-asset sleeve screening against BTC baseline.

Purpose:
  Automate portfolio additivity checks so candidate sleeves are accepted/rejected
  mechanically rather than ad-hoc.

Workflow:
  1) Run BTC sleeve and candidate sleeve on aligned timelines.
  2) Simulate combined portfolio under one or more allocator modes.
  3) Compare Combined metrics vs BTC-only baseline for each run.
  4) Emit:
     - detailed run table (one row per asset x cost x allocator)
     - candidate scoreboard with pass/fail decision.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARGUS_DIR = REPO_ROOT / "runtime" / "argus"
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ARGUS_DIR) not in sys.path:
    sys.path.insert(0, str(ARGUS_DIR))
# Argus is on path first so `research.*` resolves to runtime/argus/research (harness, etc.).
# Portfolio simulation lives in this folder; import by module name to avoid `research.experiments`
# shadowing (there is no argus/research/experiments).
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from research.harness.backtest_runner import load_strategy_func, run_backtest
from run_vb_multi_sleeve_portfolio import (
    _simulate_combined,
    _simulate_combined_with_risk_overlay,
)


@dataclass
class SleeveConfig:
    name: str
    asset: str
    data: Path
    strategy_module: str
    strategy_func: str


def _infer_data_path(asset: str) -> Path:
    return REPO_ROOT / "data" / f"{asset.strip().lower()}usd_3600s_2019-01-01_to_2025-12-30.csv"


def _load_ohlcv_flexible(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    lower_to_actual = {str(c).strip().lower(): c for c in raw.columns}

    def _pick(*names: str) -> str | None:
        for n in names:
            c = lower_to_actual.get(n.lower())
            if c is not None:
                return c
        return None

    col_ts = _pick("Timestamp", "timestamp", "ts", "datetime", "date")
    col_o = _pick("Open", "open", "o")
    col_h = _pick("High", "high", "h")
    col_l = _pick("Low", "low", "l")
    col_c = _pick("Close", "close", "c")
    col_v = _pick("Volume", "volume", "v")
    if any(c is None for c in (col_ts, col_o, col_h, col_l, col_c)):
        raise ValueError(f"{csv_path} missing required OHLCV columns. Found: {list(raw.columns)}")

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(raw[col_ts], utc=True, errors="coerce"),
            "Open": pd.to_numeric(raw[col_o], errors="coerce"),
            "High": pd.to_numeric(raw[col_h], errors="coerce"),
            "Low": pd.to_numeric(raw[col_l], errors="coerce"),
            "Close": pd.to_numeric(raw[col_c], errors="coerce"),
            "Volume": pd.to_numeric(raw[col_v], errors="coerce") if col_v is not None else 0.0,
        }
    )
    df = df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def _run_sleeve(cfg: SleeveConfig, *, lookback: int, initial_equity: float, fee_bps: float, slippage_bps: float) -> pd.DataFrame:
    data_path = cfg.data if cfg.data.is_absolute() else (REPO_ROOT / cfg.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"{cfg.name}: data file not found: {data_path}")
    df = _load_ohlcv_flexible(data_path)
    fn = load_strategy_func(cfg.strategy_module, cfg.strategy_func)
    eq_df, _ = run_backtest(
        df,
        fn,
        lookback=lookback,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        closed_only=True,
        moonwire_feed=None,
    )
    out = eq_df[["Timestamp", "equity", "price", "desired_exposure_frac", "exposure"]].copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    out["return"] = out["equity"].pct_change().fillna(0.0)
    return out.rename(
        columns={
            "equity": f"{cfg.name}_equity",
            "price": f"{cfg.name}_price",
            "desired_exposure_frac": f"{cfg.name}_desired_exposure",
            "exposure": f"{cfg.name}_exposure",
            "return": f"{cfg.name}_return",
        }
    )


def _align_sleeves(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(btc_df, eth_df, on="Timestamp", how="inner").sort_values("Timestamp").reset_index(drop=True)
    if len(merged) < 2:
        raise ValueError("Aligned BTC/asset timeline has fewer than 2 rows.")
    return merged


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(peak, 1e-12)) - 1.0
    return float(np.nanmin(dd))


def _compute_profit_factor_from_returns(returns: np.ndarray) -> float:
    r = returns[~np.isnan(returns)]
    wins = r[r > 0]
    losses = r[r < 0]
    if wins.size == 0 and losses.size == 0:
        return 0.0
    if wins.size == 0:
        return 0.0
    if losses.size == 0:
        return float("inf")
    return float(wins.sum()) / float(abs(losses.sum()))


def _compute_metrics(ts: pd.Series, returns: np.ndarray, equity: np.ndarray, avg_exposure: float) -> Dict[str, float]:
    tsu = pd.to_datetime(ts, utc=True)
    years = (tsu.max() - tsu.min()).total_seconds() / (365.25 * 24 * 3600.0)
    years = years if years > 0 else float("nan")
    if years == years and years > 0 and equity[0] > 0 and equity[-1] > 0:
        cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")
    maxdd = abs(_max_drawdown(equity))
    calmar = cagr / maxdd if maxdd > 1e-12 and np.isfinite(cagr) else float("nan")
    return {
        "CAGR": cagr,
        "MaxDD": maxdd,
        "Calmar": calmar,
        "PF": _compute_profit_factor_from_returns(returns),
        "exposure": float(avg_exposure),
        "final_equity": float(equity[-1]),
    }


def _run_portfolio_simulation(
    merged: pd.DataFrame,
    *,
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
    allocator_mode: str,
    max_secondary_exposure: float | None,
    risk_overlay: str,
    overlay_mode: str,
    dd_overlay_soft: float,
    dd_overlay_hard: float,
    dd_overlay_min_mult: float,
    dd_soft: float,
    dd_med: float,
    dd_hard: float,
    dd_kill: float,
    dd_soft_mult: float,
    dd_med_mult: float,
    dd_hard_mult: float,
    dd_kill_mult: float,
    vol_overlay_target_ann: float | None,
    vol_overlay_window: int,
) -> pd.DataFrame:
    effective_overlay = (overlay_mode or risk_overlay or "none").strip().lower()
    if effective_overlay != "none":
        return _simulate_combined_with_risk_overlay(
            merged,
            initial_equity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            allocator_mode=allocator_mode,
            allow_gross_above_one=False,
            max_secondary_exposure=max_secondary_exposure,
            risk_overlay=effective_overlay,
            overlay_mode=effective_overlay,
            dd_soft_frac=float(dd_overlay_soft),
            dd_hard_frac=float(dd_overlay_hard),
            dd_min_mult=float(dd_overlay_min_mult),
            dd_soft=float(dd_soft),
            dd_med=float(dd_med),
            dd_hard=float(dd_hard),
            dd_kill=float(dd_kill),
            dd_soft_mult=float(dd_soft_mult),
            dd_med_mult=float(dd_med_mult),
            dd_hard_mult=float(dd_hard_mult),
            dd_kill_mult=float(dd_kill_mult),
            vol_target_ann=vol_overlay_target_ann,
            vol_window_bars=int(vol_overlay_window),
            enable_sanity_checks=True,
        )
    return _simulate_combined(
        merged,
        initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        allocator_mode=allocator_mode,
        allow_gross_above_one=False,
        max_secondary_exposure=max_secondary_exposure,
        enable_sanity_checks=True,
    )


@dataclass
class Candidate:
    asset: str
    data: Path
    strategy_module: str
    strategy_func: str


def _parse_assets(raw: str) -> List[str]:
    items = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise ValueError("No assets provided.")
    return items


def _parse_cost_pairs(raw: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if ":" not in token:
            raise ValueError(f"Invalid cost token '{token}'. Use fee:slippage format, e.g. 10:5")
        fee_s, slip_s = token.split(":", 1)
        pairs.append((float(fee_s), float(slip_s)))
    if not pairs:
        raise ValueError("No cost pairs provided.")
    return pairs


def _parse_allocators(raw: str) -> List[str]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise ValueError("No allocator modes provided.")
    return items


def _build_candidates(
    *,
    assets: List[str],
    data_template: str,
    strategy_module_template: str,
    strategy_func: str,
) -> List[Candidate]:
    out: List[Candidate] = []
    for asset in assets:
        data = Path(data_template.format(asset=asset))
        if not data.is_absolute():
            data = (REPO_ROOT / data).resolve()
        out.append(
            Candidate(
                asset=asset,
                data=data,
                strategy_module=strategy_module_template.format(asset=asset),
                strategy_func=strategy_func,
            )
        )
    return out


def _evaluate_row(
    *,
    combined_cagr: float,
    combined_maxdd: float,
    combined_calmar: float,
    btc_cagr: float,
    btc_maxdd: float,
    btc_calmar: float,
    cagr_min_ratio: float,
    maxdd_additive_cap: float,
    calmar_min_ratio: float,
) -> Dict[str, bool]:
    cagr_ok = combined_cagr >= (btc_cagr * cagr_min_ratio)
    maxdd_ok = combined_maxdd <= (btc_maxdd + maxdd_additive_cap)
    calmar_ok = combined_calmar >= (btc_calmar * calmar_min_ratio)
    return {
        "cagr_ok": bool(cagr_ok),
        "maxdd_ok": bool(maxdd_ok),
        "calmar_ok": bool(calmar_ok),
        "all_ok": bool(cagr_ok and maxdd_ok and calmar_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen candidate sleeves vs BTC baseline with pass/fail gating.")
    ap.add_argument("--btc_data", type=str, default=str(_infer_data_path("btc")))
    ap.add_argument("--btc_strategy_module", type=str, default="research.strategies.sg_volatility_breakout_v1")
    ap.add_argument("--btc_strategy_func", type=str, default="generate_intent")
    ap.add_argument("--assets", type=str, default="eth", help="Comma-separated assets, e.g. eth,sol,xrp")
    ap.add_argument(
        "--data_template",
        type=str,
        default="data/{asset}usd_3600s_2019-01-01_to_2025-12-30.csv",
        help="Template for candidate data paths.",
    )
    ap.add_argument(
        "--strategy_module_template",
        type=str,
        default="research.strategies.sg_{asset}_state_machine_v1",
        help="Template for candidate strategy module path.",
    )
    ap.add_argument("--strategy_func", type=str, default="generate_intent")
    ap.add_argument("--lookback", type=int, default=200)
    ap.add_argument("--initial_equity", type=float, default=10000.0)
    ap.add_argument(
        "--costs",
        type=str,
        default="10:5,25:10",
        help="Comma list of fee:slippage pairs, e.g. 10:5,25:10",
    )
    ap.add_argument(
        "--allocator_modes",
        type=str,
        default="normalize,btc_priority,btc_capped_eth,eth_when_btc_flat",
        help="Comma list of allocator modes.",
    )
    ap.add_argument(
        "--cagr_min_ratio",
        type=float,
        default=0.90,
        help="Require combined CAGR >= btc CAGR * this ratio.",
    )
    ap.add_argument(
        "--maxdd_additive_cap",
        type=float,
        default=0.10,
        help="Require combined MaxDD <= btc MaxDD + this absolute cap.",
    )
    ap.add_argument(
        "--calmar_min_ratio",
        type=float,
        default=0.85,
        help="Require combined Calmar >= btc Calmar * this ratio.",
    )
    ap.add_argument(
        "--out_detailed",
        type=str,
        default="research/experiments/output/multi_asset_sleeve_screen_detailed.csv",
    )
    ap.add_argument(
        "--out_scoreboard",
        type=str,
        default="research/experiments/output/multi_asset_sleeve_screen_scoreboard.csv",
    )
    ap.add_argument(
        "--max_secondary_exposure",
        type=float,
        default=None,
        help="Cap secondary sleeve weight after allocator (0..1), e.g. 0.02.",
    )
    ap.add_argument(
        "--risk_overlay",
        type=str,
        choices=["none", "drawdown", "vol", "both", "dd_ladder"],
        default="none",
        help="Portfolio risk overlay (forward-only gross scaling).",
    )
    ap.add_argument(
        "--overlay_mode",
        type=str,
        choices=["none", "drawdown", "vol", "both", "dd_ladder"],
        default="",
        help="Optional alias/override for overlay mode; defaults to --risk_overlay.",
    )
    ap.add_argument("--dd_overlay_soft", type=float, default=0.12)
    ap.add_argument("--dd_overlay_hard", type=float, default=0.25)
    ap.add_argument("--dd_overlay_min_mult", type=float, default=0.35)
    ap.add_argument("--dd_soft", type=float, default=0.05)
    ap.add_argument("--dd_med", type=float, default=0.10)
    ap.add_argument("--dd_hard", type=float, default=0.15)
    ap.add_argument("--dd_kill", type=float, default=0.20)
    ap.add_argument("--dd_soft_mult", type=float, default=0.85)
    ap.add_argument("--dd_med_mult", type=float, default=0.65)
    ap.add_argument("--dd_hard_mult", type=float, default=0.35)
    ap.add_argument("--dd_kill_mult", type=float, default=0.00)
    ap.add_argument(
        "--vol_overlay_target_ann",
        type=float,
        default=None,
        help="Annual vol target for overlay (e.g. 0.15); use with vol/both.",
    )
    ap.add_argument("--vol_overlay_window", type=int, default=504)
    args = ap.parse_args()
    effective_overlay = (args.overlay_mode or args.risk_overlay or "none").strip().lower()

    assets = _parse_assets(args.assets)
    cost_pairs = _parse_cost_pairs(args.costs)
    allocator_modes = _parse_allocators(args.allocator_modes)
    candidates = _build_candidates(
        assets=assets,
        data_template=args.data_template,
        strategy_module_template=args.strategy_module_template,
        strategy_func=args.strategy_func,
    )

    btc_cfg = SleeveConfig(
        name="btc",
        asset="btc",
        data=Path(args.btc_data),
        strategy_module=args.btc_strategy_module,
        strategy_func=args.btc_strategy_func,
    )

    rows: List[Dict[str, object]] = []

    for candidate in candidates:
        for fee_bps, slippage_bps in cost_pairs:
            btc_df = _run_sleeve(
                btc_cfg,
                lookback=args.lookback,
                initial_equity=args.initial_equity,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            eth_cfg = SleeveConfig(
                name="eth",
                asset=candidate.asset,
                data=candidate.data,
                strategy_module=candidate.strategy_module,
                strategy_func=candidate.strategy_func,
            )
            eth_df = _run_sleeve(
                eth_cfg,
                lookback=args.lookback,
                initial_equity=args.initial_equity,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            merged = _align_sleeves(btc_df, eth_df)

            for allocator_mode in allocator_modes:
                sim = _run_portfolio_simulation(
                    merged,
                    initial_equity=args.initial_equity,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    allocator_mode=allocator_mode,
                    max_secondary_exposure=args.max_secondary_exposure,
                    risk_overlay=args.risk_overlay,
                    overlay_mode=effective_overlay,
                    dd_overlay_soft=args.dd_overlay_soft,
                    dd_overlay_hard=args.dd_overlay_hard,
                    dd_overlay_min_mult=args.dd_overlay_min_mult,
                    dd_soft=args.dd_soft,
                    dd_med=args.dd_med,
                    dd_hard=args.dd_hard,
                    dd_kill=args.dd_kill,
                    dd_soft_mult=args.dd_soft_mult,
                    dd_med_mult=args.dd_med_mult,
                    dd_hard_mult=args.dd_hard_mult,
                    dd_kill_mult=args.dd_kill_mult,
                    vol_overlay_target_ann=args.vol_overlay_target_ann,
                    vol_overlay_window=args.vol_overlay_window,
                )

                m_btc = _compute_metrics(
                    sim["Timestamp"],
                    sim["btc_return"].to_numpy(dtype=float),
                    sim["btc_equity"].to_numpy(dtype=float),
                    avg_exposure=float(sim["btc_exposure"].mean()),
                )
                m_eth = _compute_metrics(
                    sim["Timestamp"],
                    sim["eth_return"].to_numpy(dtype=float),
                    sim["eth_equity"].to_numpy(dtype=float),
                    avg_exposure=float(sim["eth_exposure"].mean()),
                )
                m_comb = _compute_metrics(
                    sim["Timestamp"],
                    sim["combined_return"].to_numpy(dtype=float),
                    sim["combined_equity"].to_numpy(dtype=float),
                    avg_exposure=float((sim["w_btc"] + sim["w_eth"]).mean()),
                )

                checks = _evaluate_row(
                    combined_cagr=float(m_comb["CAGR"]),
                    combined_maxdd=float(m_comb["MaxDD"]),
                    combined_calmar=float(m_comb["Calmar"]),
                    btc_cagr=float(m_btc["CAGR"]),
                    btc_maxdd=float(m_btc["MaxDD"]),
                    btc_calmar=float(m_btc["Calmar"]),
                    cagr_min_ratio=args.cagr_min_ratio,
                    maxdd_additive_cap=args.maxdd_additive_cap,
                    calmar_min_ratio=args.calmar_min_ratio,
                )

                rows.append(
                    {
                        "asset": candidate.asset,
                        "strategy_module": candidate.strategy_module,
                        "strategy_func": candidate.strategy_func,
                        "fee_bps": fee_bps,
                        "slippage_bps": slippage_bps,
                        "allocator_mode": allocator_mode,
                        "max_secondary_exposure": args.max_secondary_exposure,
                        "risk_overlay": effective_overlay,
                        "btc_cagr": float(m_btc["CAGR"]),
                        "btc_maxdd": float(m_btc["MaxDD"]),
                        "btc_calmar": float(m_btc["Calmar"]),
                        "combined_cagr": float(m_comb["CAGR"]),
                        "combined_maxdd": float(m_comb["MaxDD"]),
                        "combined_calmar": float(m_comb["Calmar"]),
                        "eth_cagr": float(m_eth["CAGR"]),
                        "eth_maxdd": float(m_eth["MaxDD"]),
                        "eth_calmar": float(m_eth["Calmar"]),
                        **checks,
                    }
                )

    detailed = pd.DataFrame(rows).sort_values(
        by=["asset", "strategy_module", "fee_bps", "slippage_bps", "allocator_mode"]
    ).reset_index(drop=True)

    grouped = detailed.groupby(["asset", "strategy_module", "strategy_func"], as_index=False).agg(
        runs=("all_ok", "size"),
        passed_runs=("all_ok", "sum"),
        min_combined_cagr=("combined_cagr", "min"),
        max_combined_maxdd=("combined_maxdd", "max"),
        min_combined_calmar=("combined_calmar", "min"),
        min_btc_cagr=("btc_cagr", "min"),
        max_btc_maxdd=("btc_maxdd", "max"),
        min_btc_calmar=("btc_calmar", "min"),
    )
    grouped["decision"] = grouped["passed_runs"].eq(grouped["runs"]).map({True: "PROMOTE", False: "REJECT"})

    out_detailed = Path(args.out_detailed)
    out_scoreboard = Path(args.out_scoreboard)
    if not out_detailed.is_absolute():
        out_detailed = (REPO_ROOT / out_detailed).resolve()
    if not out_scoreboard.is_absolute():
        out_scoreboard = (REPO_ROOT / out_scoreboard).resolve()
    out_detailed.parent.mkdir(parents=True, exist_ok=True)
    out_scoreboard.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out_detailed, index=False)
    grouped.to_csv(out_scoreboard, index=False)

    print("=" * 70)
    print("MULTI-ASSET SLEEVE SCREEN")
    print("=" * 70)
    print(f"Candidates tested: {len(candidates)}")
    print(f"Run rows: {len(detailed)}")
    print(
        f"Gates: CAGR>={args.cagr_min_ratio:.2f}x BTC | "
        f"MaxDD<=BTC+{args.maxdd_additive_cap:.2f} | "
        f"Calmar>={args.calmar_min_ratio:.2f}x BTC"
    )
    if args.max_secondary_exposure is not None:
        print(f"Max secondary exposure: {args.max_secondary_exposure:.4f}")
    if effective_overlay != "none":
        print(
            f"Risk overlay: {effective_overlay} | dd {args.dd_overlay_soft}/{args.dd_overlay_hard} | "
            f"vol_target_ann={args.vol_overlay_target_ann} window={args.vol_overlay_window}"
        )
    if effective_overlay == "dd_ladder":
        print(
            f"DD ladder: thresholds={args.dd_soft}/{args.dd_med}/{args.dd_hard}/{args.dd_kill} | "
            f"mult={args.dd_soft_mult}/{args.dd_med_mult}/{args.dd_hard_mult}/{args.dd_kill_mult}"
        )
    print("-" * 70)
    for _, row in grouped.iterrows():
        print(
            f"{row['asset']} | {row['strategy_module']} | "
            f"passed={int(row['passed_runs'])}/{int(row['runs'])} | decision={row['decision']}"
        )
    print("-" * 70)
    print(f"Wrote detailed:   {out_detailed}")
    print(f"Wrote scoreboard: {out_scoreboard}")


if __name__ == "__main__":
    main()

