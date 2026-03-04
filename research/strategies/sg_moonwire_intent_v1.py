"""
MoonWire Signal Integration Strategy (Layer-2)

Consumes pre-generated MoonWire signal feeds (JSONL format) and maps
probability scores to StrategyIntent actions.

Configuration via environment variables:
- MOONWIRE_SIGNAL_FILE: Path to signals.jsonl
- MOONWIRE_LONG_THRESH: Probability threshold for LONG (default 0.65)
- MOONWIRE_SHORT_THRESH: Probability threshold for SHORT (default 0.35)
- MOONWIRE_ALLOW_SHORT: Enable SHORT signals (default 0)
- MOONWIRE_REQUIRE_EXACT_TS: Fail if timestamp not found (default 1)

Architecture:
- Deterministic: Same feed + same params = same intents
- Closed-bar semantics: Decision at bar t uses data through bar t
- Fail-fast: Missing timestamp errors by default (no silent fallbacks)
- Module-level caching: Feed loaded once per Python process

Author: Itera Dynamics
Date: 2026-03-04
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
from runtime.argus.apex_core.strategy_intent import Intent, Action


# ============================
# Configuration
# ============================

class MoonWireConfig:
    """Environment-based configuration for MoonWire strategy."""
    
    def __init__(self):
        self.signal_file = os.getenv("MOONWIRE_SIGNAL_FILE")
        self.long_thresh = float(os.getenv("MOONWIRE_LONG_THRESH", "0.65"))
        self.short_thresh = float(os.getenv("MOONWIRE_SHORT_THRESH", "0.35"))
        self.allow_short = os.getenv("MOONWIRE_ALLOW_SHORT", "0") == "1"
        self.require_exact_ts = os.getenv("MOONWIRE_REQUIRE_EXACT_TS", "1") == "1"
    
    def validate(self) -> None:
        """Validate configuration and raise clear errors."""
        if not self.signal_file:
            raise ValueError(
                "MOONWIRE_SIGNAL_FILE not set. "
                "Export signals from MoonWire and set env var to path."
            )
        
        if not Path(self.signal_file).exists():
            raise FileNotFoundError(
                f"Signal file not found: {self.signal_file}\n"
                f"Export signals using: python scripts/export_signal_feed.py"
            )
        
        if not (0 <= self.long_thresh <= 1):
            raise ValueError(f"MOONWIRE_LONG_THRESH must be 0-1, got {self.long_thresh}")
        
        if not (0 <= self.short_thresh <= 1):
            raise ValueError(f"MOONWIRE_SHORT_THRESH must be 0-1, got {self.short_thresh}")


# ============================
# Signal Feed Loader
# ============================

_SIGNAL_CACHE: Optional[Dict[int, float]] = None
_CONFIG_CACHE: Optional[MoonWireConfig] = None


def load_signal_feed(config: MoonWireConfig) -> Dict[int, float]:
    """
    Load MoonWire signal feed from JSONL.
    
    Returns:
        Dict mapping timestamp (Unix seconds) -> probability (0-1)
    """
    global _SIGNAL_CACHE
    
    if _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE
    
    signals = {}
    with open(config.signal_file, "r") as f:
        for line in f:
            record = json.loads(line)
            ts = int(record["timestamp"])
            prob = float(record["probability"])
            signals[ts] = prob
    
    _SIGNAL_CACHE = signals
    return signals


def get_config() -> MoonWireConfig:
    """Get or create cached config."""
    global _CONFIG_CACHE
    
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = MoonWireConfig()
        _CONFIG_CACHE.validate()
    
    return _CONFIG_CACHE


# ============================
# Signal Mapping
# ============================

def map_probability_to_intent(
    probability: float,
    config: MoonWireConfig,
) -> Intent:
    """
    Map MoonWire probability to StrategyIntent.
    
    Thresholds:
    - probability >= LONG_THRESH → ENTER_LONG
    - probability <= SHORT_THRESH (and ALLOW_SHORT=1) → EXIT (short proxy)
    - else → FLAT
    
    Args:
        probability: ML model probability (0-1)
        config: MoonWire configuration
    
    Returns:
        Intent with action, confidence, reason
    """
    if probability >= config.long_thresh:
        return Intent(
            action=Action.ENTER_LONG,
            confidence=probability,
            reason=f"MoonWire LONG signal (p={probability:.3f} >= {config.long_thresh})",
            meta={"source": "moonwire", "probability": probability},
        )
    
    elif config.allow_short and probability <= config.short_thresh:
        # Note: Using EXIT as SHORT proxy until SHORT is validated
        # When SHORT is enabled, Layer-3 governance will handle position sizing
        return Intent(
            action=Action.EXIT,
            confidence=1.0 - probability,  # Invert for short confidence
            reason=f"MoonWire SHORT signal (p={probability:.3f} <= {config.short_thresh})",
            meta={"source": "moonwire", "probability": probability, "short_signal": True},
        )
    
    else:
        return Intent(
            action=Action.FLAT,
            confidence=0.5,
            reason=f"MoonWire neutral (p={probability:.3f} in [{config.short_thresh}, {config.long_thresh}])",
            meta={"source": "moonwire", "probability": probability},
        )


# ============================
# External Strategy API
# ============================

def generate_intent(df: pd.DataFrame, ctx) -> Intent:
    """
    External Strategy API for MoonWire signals.
    
    Args:
        df: DataFrame of flight_recorder candles (tail = most recent bar)
        ctx: StrategyContext from SG (contains timestamp, product, etc.)
    
    Returns:
        Intent with action, confidence, reason
    """
    config = get_config()
    signals = load_signal_feed(config)
    
    # Get current bar timestamp (most recent closed bar)
    last_bar = df.iloc[-1]
    bar_ts = int(pd.Timestamp(last_bar.name).timestamp())
    
    # Lookup signal for this timestamp
    if bar_ts in signals:
        probability = signals[bar_ts]
        return map_probability_to_intent(probability, config)
    
    # Handle missing timestamp
    if config.require_exact_ts:
        raise KeyError(
            f"Timestamp {bar_ts} not found in signal feed.\n"
            f"Feed may be stale or misaligned.\n"
            f"Set MOONWIRE_REQUIRE_EXACT_TS=0 to use nearest prior signal (NOT RECOMMENDED)."
        )
    
    # Fallback: Find nearest prior signal (optional)
    prior_timestamps = [ts for ts in signals.keys() if ts <= bar_ts]
    if not prior_timestamps:
        return Intent(
            action=Action.FLAT,
            reason="No MoonWire signals available before current timestamp",
            meta={"source": "moonwire", "fallback": True},
        )
    
    nearest_ts = max(prior_timestamps)
    probability = signals[nearest_ts]
    
    return Intent(
        action=Action.FLAT,
        confidence=0.0,
        reason=f"MoonWire signal stale (using ts={nearest_ts}, age={(bar_ts - nearest_ts) // 3600}h)",
        meta={
            "source": "moonwire",
            "probability": probability,
            "fallback": True,
            "age_hours": (bar_ts - nearest_ts) // 3600,
        },
    )


# ============================
# Diagnostics
# ============================

def validate_feed_alignment(
    df: pd.DataFrame,
    signal_file: str,
) -> Dict[str, Any]:
    """
    Diagnostic tool to check timestamp alignment between strategy bars and signal feed.
    
    Args:
        df: Strategy bar DataFrame
        signal_file: Path to signals.jsonl
    
    Returns:
        Dict with alignment metrics
    """
    signals = {}
    with open(signal_file, "r") as f:
        for line in f:
            record = json.loads(line)
            signals[int(record["timestamp"])] = float(record["probability"])
    
    bar_timestamps = [int(pd.Timestamp(idx).timestamp()) for idx in df.index]
    
    matched = sum(1 for ts in bar_timestamps if ts in signals)
    missing = [ts for ts in bar_timestamps if ts not in signals]
    
    return {
        "total_bars": len(bar_timestamps),
        "matched": matched,
        "missing": len(missing),
        "coverage": matched / len(bar_timestamps) if bar_timestamps else 0,
        "missing_timestamps": missing[:10],  # First 10 for debugging
        "signal_count": len(signals),
        "first_signal_ts": min(signals.keys()) if signals else None,
        "last_signal_ts": max(signals.keys()) if signals else None,
    }
