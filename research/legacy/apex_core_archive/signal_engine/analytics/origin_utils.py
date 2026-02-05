# signal_engine/analytics/origin_utils.py
"""
Origin and Data Utilities for Analytics.

Provides utilities for normalizing data source origins, parsing timestamps,
and reading JSONL files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional
from datetime import datetime, timedelta, timezone
import json


# Default origin alias map (can be extended by products)
DEFAULT_ALIAS_MAP = {
    "twitter_api": "twitter",
    "twitter": "twitter",
    "Twitter": "twitter",
    "rss": "rss_news",
    "rss_news": "rss_news",
    "reddit": "reddit",
    "Reddit": "reddit",
}


def normalize_origin(
    raw: Any,
    alias_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Normalize a data source origin string.

    Args:
        raw: Raw origin value (string, or any type that can be converted)
        alias_map: Optional custom alias mapping (default: DEFAULT_ALIAS_MAP)

    Returns:
        Normalized lowercase origin string, or "unknown" if invalid
    """
    if alias_map is None:
        alias_map = DEFAULT_ALIAS_MAP

    if raw is None:
        return "unknown"

    s = str(raw).strip()
    if not s:
        return "unknown"

    return alias_map.get(s, s.lower())


# Backward-compatible alias
def extract_origin(raw: Any, alias_map: Optional[Dict[str, str]] = None) -> str:
    """Backward-compatible alias for normalize_origin."""
    return normalize_origin(raw, alias_map)


def parse_timestamp(val: Any) -> Optional[datetime]:
    """
    Parse various timestamp formats to UTC datetime.

    Accepts:
    - float/int epoch seconds
    - numeric strings (e.g., "1723558387.598")
    - ISO-8601 strings (with or without Z)

    Args:
        val: Timestamp value in various formats

    Returns:
        Timezone-aware UTC datetime or None on failure
    """
    if val is None:
        return None

    # Handle numeric types
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except Exception:
            return None

    # Handle strings
    try:
        s = str(val).strip()

        # Try numeric string as epoch seconds
        if s.replace(".", "", 1).isdigit():
            try:
                return datetime.fromtimestamp(float(s), tz=timezone.utc)
            except Exception:
                pass

        # Handle ISO-8601 with 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        dt = datetime.fromisoformat(s)

        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return dt
    except Exception:
        return None


# Backward-compatible alias
parse_ts = parse_timestamp


def is_within_window(
    ts: Optional[datetime],
    reference_time: datetime,
    days: int
) -> bool:
    """
    Check if timestamp is within a time window.

    Args:
        ts: Timestamp to check
        reference_time: Reference time (usually now)
        days: Number of days in the window

    Returns:
        True if timestamp is within the window, False otherwise
    """
    if ts is None:
        return False
    return ts >= (reference_time - timedelta(days=days))


def stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Stream JSONL file line by line, tolerating malformed lines.

    Args:
        path: Path to JSONL file

    Yields:
        Parsed JSON objects (dicts) from each line
    """
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


# Backward-compatible alias
tolerant_jsonl_stream = stream_jsonl


def compute_origin_breakdown(
    data_files: List[Path],
    days: int,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp",
    alias_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compute breakdown of events by origin from JSONL files.

    Args:
        data_files: List of JSONL file paths to analyze
        days: Number of days to look back
        origin_field: Field name for origin in data
        timestamp_field: Field name for timestamp in data
        alias_map: Optional custom alias mapping

    Returns:
        Tuple of (rows, totals):
        - rows: List of dicts with origin, count, pct, percent (sorted by count desc)
        - totals: Dict with total_events count
    """
    now_utc = datetime.now(timezone.utc)

    # Count events per origin
    origin_counts: Dict[str, int] = {}
    total_events = 0

    for data_file in data_files:
        for row in stream_jsonl(data_file):
            # Check timestamp
            ts = parse_timestamp(row.get(timestamp_field))
            if not is_within_window(ts, now_utc, days):
                continue

            # Extract origin (try multiple possible fields)
            org = normalize_origin(
                row.get(origin_field)
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source"),
                alias_map
            )

            origin_counts[org] = origin_counts.get(org, 0) + 1
            total_events += 1

    # Build rows with percentages
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for org, count in origin_counts.items():
            pct = round(100.0 * count / total_events, 2)
            rows.append({
                "origin": org,
                "count": count,
                "pct": pct,
                "percent": pct,
            })

        # Sort: count desc, origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {"total_events": total_events}

    return rows, totals


__all__ = [
    'DEFAULT_ALIAS_MAP',
    'normalize_origin',
    'extract_origin',
    'parse_timestamp',
    'parse_ts',
    'is_within_window',
    'stream_jsonl',
    'tolerant_jsonl_stream',
    'compute_origin_breakdown',
]
