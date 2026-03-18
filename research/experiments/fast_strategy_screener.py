# research/experiments/fast_strategy_screener.py
"""
Fast Deterministic Strategy Screener (Stage 1 of hybrid quant research pipeline)

Purpose:
  Quickly reject obviously bad strategy ideas before promoting them into the full
  Argus research harness and portfolio geometry. Uses only vectorized pandas/numpy.
  Does NOT replace the harness; it is a cheap first-pass filter.

Architecture:
  - Data: single BTC hourly OHLCV CSV → DataFrame with DatetimeIndex.
  - Backtest engine: compute_position_series(entry, exit) → position (0/1);
    compute_equity_curve(df, position, fee_bps, slippage_bps) → equity;
    get_trade_list(df, position) → list of {entry_idx, exit_idx, entry_price, exit_price, hold_bars};
    summary_metrics(df, equity, trades) → dict of total_return_pct, cagr, max_drawdown, win_rate,
    profit_factor, n_trades, avg_hold_hours. run_backtest() ties them together.
  - Templates: each is a function (df, **params) -> (entry_signal, exit_signal) (boolean Series).
  - Runner: for each (strategy_name, param_set) in TEMPLATES × PARAM_GRIDS, run_backtest → one row;
    add survives_filter (cagr>0, profit_factor>1, max_drawdown<60%%, n_trades>=30); write full CSV and
    filtered CSV; print top N from filtered, sorted by cagr desc, profit_factor desc, max_drawdown asc.

Example command:
  python research/experiments/fast_strategy_screener.py
  python research/experiments/fast_strategy_screener.py --data data/btcusd_3600s_2019-01-01_to_2025-12-30.csv --fee_bps 25 --slippage_bps 10 --top 10

Output CSV locations:
  research/experiments/output/fast_strategy_screener_results.csv (all candidates + survives_filter)
  research/experiments/output/fast_strategy_screener_filtered_results.csv (survivors only)

Adding new templates later:
  1. Implement a function (df: pd.DataFrame, **kwargs) -> Tuple[pd.Series, pd.Series]
     returning (entry_signal, exit_signal) as boolean Series aligned to df.index.
  2. Register it in TEMPLATES: TEMPLATES["my_template"] = my_template_signals.
  3. Add a small parameter grid in PARAM_GRIDS["my_template"] = [{"param": value}, ...].
  4. Re-run the screener; the new template will be evaluated with the rest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "btcusd_3600s_2019-01-01_to_2025-12-30.csv"
OUTPUT_DIR = REPO_ROOT / "research" / "experiments" / "output"
OUTPUT_CSV = OUTPUT_DIR / "fast_strategy_screener_results.csv"
OUTPUT_FILTERED_CSV = OUTPUT_DIR / "fast_strategy_screener_filtered_results.csv"

# Pre-filter thresholds for "top candidates" (only survivors are printed and in filtered CSV)
FILTER_CAGR_MIN = 0.0
FILTER_PROFIT_FACTOR_MIN = 1.0
FILTER_MAX_DRAWDOWN_MAX = 0.60   # reject if drawdown >= 60%
FILTER_N_TRADES_MIN = 30


# -----------------------------------------------------------------------------
# Backtest engine: equity curve, trade list, summary metrics
# -----------------------------------------------------------------------------

def compute_position_series(
    entry_signal: pd.Series,
    exit_signal: pd.Series,
) -> pd.Series:
    """
    Long-only position (0 or 1) from entry/exit signals.
    Entry at close of bar t => position 1 from bar t+1.
    Exit at close of bar t => position 0 from bar t+1.
    """
    n = len(entry_signal)
    position = np.zeros(n, dtype=np.float64)
    pos = 0.0
    for i in range(n):
        if pos == 0 and entry_signal.iloc[i]:
            pos = 1.0
        elif pos == 1 and exit_signal.iloc[i]:
            pos = 0.0
        position[i] = pos
    return pd.Series(position, index=entry_signal.index)


def compute_equity_curve(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """
    Equity curve starting at 1.0. Assumes position is known at start of each bar.
    Bar return = position_prev * (close[t]/close[t-1] - 1). Costs applied on entry/exit.
    """
    close = df["Close"].values
    n = len(close)
    equity = np.ones(n)
    pos_prev = 0.0
    cost_bps = (fee_bps + slippage_bps) / 1e4

    for i in range(1, n):
        pos_curr = position.iloc[i]
        bar_ret = (close[i] / close[i - 1]) - 1.0
        cost = 0.0
        if pos_prev == 0 and pos_curr == 1:
            cost = cost_bps
        elif pos_prev == 1 and pos_curr == 0:
            cost = cost_bps
        equity[i] = equity[i - 1] * (1.0 + pos_prev * bar_ret - cost)
        pos_prev = pos_curr

    return pd.Series(equity, index=df.index)


def get_trade_list(
    df: pd.DataFrame,
    position: pd.Series,
) -> List[Dict[str, Any]]:
    """List of trades: entry_bar, exit_bar, entry_price, exit_price, hold_bars."""
    pos = position.values
    n = len(pos)
    trades: List[Dict[str, Any]] = []
    in_trade = False
    entry_idx = 0

    for i in range(n):
        if not in_trade and pos[i] == 1:
            in_trade = True
            entry_idx = i
        elif in_trade and pos[i] == 0:
            in_trade = False
            exit_idx = i - 1  # exit at close of previous bar
            if exit_idx >= entry_idx:
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "entry_price": df["Close"].iloc[entry_idx],
                    "exit_price": df["Close"].iloc[exit_idx],
                    "hold_bars": exit_idx - entry_idx,
                })
    if in_trade:
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "entry_price": df["Close"].iloc[entry_idx],
            "exit_price": df["Close"].iloc[n - 1],
            "hold_bars": (n - 1) - entry_idx,
        })
    return trades


def summary_metrics(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute total_return_pct, cagr, max_drawdown, win_rate, profit_factor, n_trades, avg_hold_hours."""
    n = len(equity)
    if n == 0:
        return _empty_metrics()

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    total_return_pct = total_return * 100.0

    index = df.index
    if isinstance(index, pd.DatetimeIndex):
        years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    else:
        years = (n - 1) / (24 * 365.25)  # assume hourly
    years = max(years, 1e-6)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0

    cummax = np.maximum.accumulate(equity.values)
    drawdowns = (cummax - equity.values) / np.where(cummax > 0, cummax, 1.0)
    max_drawdown = float(np.max(drawdowns))

    n_trades = len(trades)
    if n_trades == 0:
        win_rate = 0.0
        profit_factor = 0.0
        avg_hold_hours = 0.0
    else:
        pnls = [t["exit_price"] / t["entry_price"] - 1.0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_trades
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        avg_hold_hours = float(np.mean([t["hold_bars"] for t in trades]))

    return {
        "total_return_pct": total_return_pct,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_hold_hours": avg_hold_hours,
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        "total_return_pct": 0.0,
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "n_trades": 0,
        "avg_hold_hours": 0.0,
    }


def run_backtest(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    fee_bps: float = 25.0,
    slippage_bps: float = 10.0,
) -> Tuple[pd.Series, List[Dict[str, Any]], Dict[str, float]]:
    """
    Run long-only backtest from entry/exit signals.
    Returns (equity_curve, trade_list, summary_metrics).
    """
    position = compute_position_series(entry_signal, exit_signal)
    equity = compute_equity_curve(df, position, fee_bps, slippage_bps)
    trades = get_trade_list(df, position)
    metrics = summary_metrics(df, equity, trades)
    return equity, trades, metrics


# -----------------------------------------------------------------------------
# Strategy template interface
# Each template is a function: (df, **params) -> (entry_signal, exit_signal)
# -----------------------------------------------------------------------------

def breakout_continuation_signals(
    df: pd.DataFrame,
    lookback: int = 24,
    recent_return_bars: int = 6,
    min_recent_return_pct: float = 0.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter on breakout above rolling high with positive recent return.
    Exit on close below rolling low (or simple trailing).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    roll_high = high.rolling(lookback, min_periods=1).max()
    roll_low = low.rolling(lookback, min_periods=1).min()
    recent_ret = close.pct_change(recent_return_bars)
    entry = (
        (close >= roll_high.shift(1)) &
        (roll_high.shift(1).notna()) &
        (recent_ret >= min_recent_return_pct / 100.0)
    )
    # Exit: close below recent low or below entry-period low
    exit_roll_low = low.rolling(max(lookback // 2, 4), min_periods=1).min()
    exit_sig = close < exit_roll_low.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def trend_pullback_signals(
    df: pd.DataFrame,
    trend_ma_bars: int = 48,
    pullback_lookback: int = 12,
    pullback_pct: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter when price above trend filter but short-term pullback occurred.
    Exit when price crosses below trend MA.
    """
    close = df["Close"]
    trend_ma = close.rolling(trend_ma_bars, min_periods=1).mean()
    above_trend = close > trend_ma.shift(1)
    roll_high = close.rolling(pullback_lookback, min_periods=1).max()
    pullback = (roll_high.shift(1) - close) / roll_high.shift(1).replace(0, np.nan) >= (pullback_pct / 100.0)
    entry = above_trend & pullback
    exit_sig = close < trend_ma.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def rsi_mean_reversion_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30.0,
    exit_rsi: float = 50.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter when RSI oversold, exit on RSI recovery.
    """
    close = df["Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    entry = rsi < oversold
    exit_sig = rsi >= exit_rsi
    return entry.fillna(False), exit_sig.fillna(False)


def volatility_spike_reversion_signals(
    df: pd.DataFrame,
    atr_period: int = 14,
    drop_pct: float = 3.0,
    atr_mult: float = 1.5,
    exit_bars: int = 24,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter after large drop + volatility spike (ATR expansion).
    Exit after fixed bars or recovery (simplified: exit after exit_bars or when price > entry-level).
    """
    close = df["Close"]
    high, low = df["High"], df["Low"]
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = pd.Series(tr, index=df.index).rolling(atr_period, min_periods=1).mean()
    ret = close.pct_change()
    # Large drop: close below close of N bars ago by at least drop_pct
    lookback = max(atr_period, 6)
    prev_close = close.shift(lookback)
    drop = (prev_close - close) / prev_close.replace(0, np.nan) >= (drop_pct / 100.0)
    atr_above_avg = atr > atr.rolling(atr_period * 2, min_periods=1).mean().shift(1)
    entry = drop & atr_above_avg
    # Exit: time-based would require state; use simple exit when close > rolling mean
    roll_mean = close.rolling(exit_bars, min_periods=1).mean()
    exit_sig = close >= roll_mean.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def volatility_breakout_signals(
    df: pd.DataFrame,
    atr_period: int = 14,
    breakout_lookback: int = 24,
    atr_expansion_min: float = 1.2,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter after ATR expansion + breakout above recent high.
    Exit when close below recent low.
    """
    close = df["Close"]
    high, low = df["High"], df["Low"]
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = pd.Series(tr, index=df.index).rolling(atr_period, min_periods=1).mean()
    atr_avg = atr.rolling(atr_period * 2, min_periods=1).mean().shift(1)
    roll_high = high.rolling(breakout_lookback, min_periods=1).max().shift(1)
    roll_low = low.rolling(breakout_lookback, min_periods=1).min()
    atr_expanded = atr > (atr_avg * atr_expansion_min)
    entry = (close >= roll_high) & atr_expanded
    exit_sig = close < roll_low.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


# -----------------------------------------------------------------------------
# Template registry and parameter grids
# -----------------------------------------------------------------------------

TEMPLATES: Dict[str, Callable[..., Tuple[pd.Series, pd.Series]]] = {
    "breakout_continuation": breakout_continuation_signals,
    "trend_pullback": trend_pullback_signals,
    "rsi_mean_reversion": rsi_mean_reversion_signals,
    "volatility_spike_reversion": volatility_spike_reversion_signals,
    "volatility_breakout": volatility_breakout_signals,
}

PARAM_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "breakout_continuation": [
        {"lookback": 24, "recent_return_bars": 6, "min_recent_return_pct": 0.5},
        {"lookback": 48, "recent_return_bars": 12, "min_recent_return_pct": 0.3},
        {"lookback": 12, "recent_return_bars": 4, "min_recent_return_pct": 1.0},
    ],
    "trend_pullback": [
        {"trend_ma_bars": 48, "pullback_lookback": 12, "pullback_pct": 1.0},
        {"trend_ma_bars": 24, "pullback_lookback": 6, "pullback_pct": 1.5},
        {"trend_ma_bars": 72, "pullback_lookback": 18, "pullback_pct": 0.8},
    ],
    "rsi_mean_reversion": [
        {"rsi_period": 14, "oversold": 30.0, "exit_rsi": 50.0},
        {"rsi_period": 7, "oversold": 25.0, "exit_rsi": 45.0},
        {"rsi_period": 21, "oversold": 35.0, "exit_rsi": 55.0},
    ],
    "volatility_spike_reversion": [
        {"atr_period": 14, "drop_pct": 3.0, "atr_mult": 1.5, "exit_bars": 24},
        {"atr_period": 14, "drop_pct": 5.0, "atr_mult": 1.2, "exit_bars": 48},
        {"atr_period": 7, "drop_pct": 2.0, "atr_mult": 1.8, "exit_bars": 12},
    ],
    "volatility_breakout": [
        {"atr_period": 14, "breakout_lookback": 24, "atr_expansion_min": 1.2},
        {"atr_period": 14, "breakout_lookback": 48, "atr_expansion_min": 1.1},
        {"atr_period": 7, "breakout_lookback": 12, "atr_expansion_min": 1.3},
    ],
}


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load BTC OHLCV CSV; set Timestamp as index."""
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp").sort_index()
    return df


def survives_filter(row: pd.Series) -> bool:
    """
    True if candidate passes pre-filter: cagr > 0, profit_factor > 1.0,
    max_drawdown < 0.60 (drawdown less severe than 60%), n_trades >= 30.
    """
    return (
        row["cagr"] > FILTER_CAGR_MIN
        and row["profit_factor"] > FILTER_PROFIT_FACTOR_MIN
        and row["max_drawdown"] < FILTER_MAX_DRAWDOWN_MAX
        and row["n_trades"] >= FILTER_N_TRADES_MIN
    )


# -----------------------------------------------------------------------------
# Main: run all templates × param sets, collect metrics, write CSV, print top
# -----------------------------------------------------------------------------

def run_screener(
    data_path: Path = DEFAULT_DATA,
    fee_bps: float = 25.0,
    slippage_bps: float = 10.0,
    output_path: Path = OUTPUT_CSV,
    filtered_output_path: Path = OUTPUT_FILTERED_CSV,
) -> pd.DataFrame:
    """
    Run all strategy templates over their parameter grids; return results DataFrame
    with survives_filter column. Writes full CSV and filtered CSV.
    """
    df = load_ohlcv(data_path)
    rows: List[Dict[str, Any]] = []

    for strategy_name, signal_fn in TEMPLATES.items():
        param_list = PARAM_GRIDS.get(strategy_name, [{}])
        for param_set in param_list:
            entry_signal, exit_signal = signal_fn(df, **param_set)
            _, _, metrics = run_backtest(df, entry_signal, exit_signal, fee_bps, slippage_bps)
            rows.append({
                "strategy_name": strategy_name,
                "parameter_set": str(param_set),
                "total_return_pct": metrics["total_return_pct"],
                "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "n_trades": metrics["n_trades"],
                "avg_hold_hours": metrics["avg_hold_hours"],
            })

    results = pd.DataFrame(rows)
    results["survives_filter"] = results.apply(survives_filter, axis=1)

    # Ranking: cagr desc, profit_factor desc, max_drawdown asc (less severe = better)
    results = results.sort_values(
        by=["cagr", "profit_factor", "max_drawdown"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    filtered = results[results["survives_filter"]].copy()
    filtered.to_csv(filtered_output_path, index=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast strategy screener: evaluate deterministic templates on BTC hourly OHLCV.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to BTC OHLCV CSV",
    )
    parser.add_argument(
        "--fee_bps",
        type=float,
        default=25.0,
        help="Fee in basis points (default 25)",
    )
    parser.add_argument(
        "--slippage_bps",
        type=float,
        default=10.0,
        help="Slippage in basis points (default 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--filtered_output",
        type=Path,
        default=OUTPUT_FILTERED_CSV,
        help="Filtered results CSV path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to print (default 10)",
    )
    args = parser.parse_args()

    results = run_screener(
        data_path=args.data,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        output_path=args.output,
        filtered_output_path=args.filtered_output,
    )

    n_total = len(results)
    n_surviving = results["survives_filter"].sum()

    print("Fast Strategy Screener")
    print(f"  Total candidates tested: {n_total}")
    print(f"  Candidates surviving filter: {n_surviving}")
    print("-" * 80)
    print("Top candidates (filtered: cagr>0, PF>1, max_dd<60%%, n_trades>=30); sort: cagr desc, PF desc, max_dd asc")
    print("-" * 80)
    filtered = results[results["survives_filter"]]
    top = filtered.head(args.top)
    if top.empty:
        print("  (none)")
    else:
        for _, row in top.iterrows():
            ps = row["parameter_set"]
            ps_short = (ps[:57] + "...") if len(ps) > 60 else ps
            print(
                f"  {row['strategy_name']} | {ps_short} | "
                f"CAGR={row['cagr']*100:.1f}% PF={row['profit_factor']:.2f} "
                f"DD={row['max_drawdown']*100:.1f}% n={row['n_trades']}"
            )
    print("-" * 80)
    print(f"Full results: {args.output}")
    print(f"Filtered results: {args.filtered_output}")


if __name__ == "__main__":
    main()
