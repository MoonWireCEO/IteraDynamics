"""
Live market data for VB dry-run (no auth, no trading).

Fetches BTC-USD hourly candles from Coinbase Exchange public API and maintains
a rolling CSV store: init from existing, fetch newer candles, append, dedupe by timestamp, UTC only.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import requests

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRANULARITY_HOUR = 3600
MAX_CANDLES_PER_REQUEST = 300  # Coinbase limit


def fetch_recent_hourly_candles(
    product_id: str = "BTC-USD",
    granularity: int = GRANULARITY_HOUR,
    timeout_sec: int = 15,
) -> List[List[Any]]:
    """
    Fetch recent hourly candles from Coinbase Exchange (public, no API key).
    Returns list of [time_unix_sec, low, high, open, close, volume].
    """
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": granularity}
    resp = requests.get(url, params=params, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Coinbase returned non-list: {type(data)}")
    return data


def _candle_row_to_series(c: list) -> dict:
    """Convert Coinbase candle [time, low, high, open, close, volume] to harness row."""
    ts = pd.Timestamp(c[0], unit="s", tz="UTC")
    return {
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": float(c[3]),
        "High": float(c[2]),
        "Low": float(c[1]),
        "Close": float(c[4]),
        "Volume": float(c[5]) if len(c) > 5 else 0.0,
    }


def load_existing_store(csv_path: Path) -> pd.DataFrame:
    """Load existing CSV; return empty DataFrame with correct columns if missing/invalid."""
    cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    if not csv_path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(csv_path)
        if "Timestamp" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame(columns=cols)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        for c in ("Open", "High", "Low", "Close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        else:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        return df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=cols)


def update_btc_store(
    csv_path: Path,
    product_id: str = "BTC-USD",
) -> pd.DataFrame:
    """
    Initialize from existing CSV if present, fetch newer hourly candles from Coinbase,
    append and deduplicate by timestamp (UTC), then write back to csv_path.
    Returns the updated DataFrame (sorted by Timestamp ascending).
    """
    df = load_existing_store(csv_path)
    try:
        raw = fetch_recent_hourly_candles(product_id=product_id, granularity=GRANULARITY_HOUR)
    except Exception as e:
        if df.empty:
            raise
        # Fetch failed; return existing store so runner can still use last known data
        return df

    if not raw:
        return df.sort_values("Timestamp").reset_index(drop=True) if not df.empty else df

    raw.sort(key=lambda x: x[0])
    new_rows = [_candle_row_to_series(c) for c in raw]
    new_df = pd.DataFrame(new_rows)
    new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], utc=True, errors="coerce")

    if df.empty:
        combined = new_df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    else:
        combined = pd.concat([df, new_df], ignore_index=True)
        combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], utc=True, errors="coerce")
        combined = combined.drop_duplicates(subset=["Timestamp"], keep="last")
        combined = combined.sort_values("Timestamp").reset_index(drop=True)

    combined = combined.dropna(subset=["Open", "High", "Low", "Close"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = combined.copy()
    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(csv_path, index=False)
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], utc=True)
    return combined.sort_values("Timestamp").reset_index(drop=True)
