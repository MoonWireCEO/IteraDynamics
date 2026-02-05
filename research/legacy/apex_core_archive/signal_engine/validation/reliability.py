# signal_engine/validation/reliability.py
"""
Feedback Reliability Scoring Module.

Provides functions for assessing the reliability and quality of user feedback
on model predictions.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta


def compute_basic_reliability(
    feedback_entries: List[Dict[str, Any]],
    confidence_weight: float = 0.9,
    base_score: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Compute basic reliability scores for feedback entries.

    Uses a simple confidence-weighted formula:
    reliability_score = confidence * weight + base

    Args:
        feedback_entries: List of feedback dictionaries containing:
            - type: Entry type (typically "user_feedback")
            - confidence: User's confidence in their feedback (0-1)
            - asset: Asset identifier
            - user_feedback: Feedback text
            - timestamp: When feedback was provided
        confidence_weight: Weight for confidence in scoring (default: 0.9)
        base_score: Baseline score (default: 0.1)

    Returns:
        List of dictionaries with added reliability_score field
    """
    results = []

    for entry in feedback_entries:
        if entry.get("type") != "user_feedback":
            continue

        confidence = entry.get("confidence", 0.0)
        reliability_score = round(confidence * confidence_weight + base_score, 3)

        results.append({
            "asset": entry.get("asset"),
            "user_feedback": entry.get("user_feedback"),
            "confidence": confidence,
            "reliability_score": reliability_score,
            "timestamp": entry.get("timestamp"),
        })

    return results


def compute_temporal_reliability(
    feedback_entries: List[Dict[str, Any]],
    recency_weight: float = 0.5,
    max_age_days: int = 30,
) -> List[Dict[str, Any]]:
    """
    Compute reliability scores with temporal weighting.

    More recent feedback is weighted more heavily than older feedback.

    Args:
        feedback_entries: List of feedback dictionaries
        recency_weight: Weight for recency factor (default: 0.5)
        max_age_days: Maximum age in days before feedback gets zero weight

    Returns:
        List of dictionaries with temporal_reliability_score field
    """
    results = []
    now = datetime.utcnow()

    for entry in feedback_entries:
        if entry.get("type") != "user_feedback":
            continue

        timestamp_str = entry.get("timestamp")
        confidence = entry.get("confidence", 0.0)

        # Parse timestamp
        try:
            if isinstance(timestamp_str, str):
                # Remove 'Z' and parse
                timestamp_str = timestamp_str.rstrip('Z')
                feedback_time = datetime.fromisoformat(timestamp_str)
            else:
                feedback_time = now
        except Exception:
            feedback_time = now

        # Calculate age factor
        age_days = (now - feedback_time).days
        if age_days > max_age_days:
            recency_factor = 0.0
        else:
            recency_factor = 1.0 - (age_days / max_age_days)

        # Combine confidence and recency
        base_score = confidence * (1 - recency_weight)
        recency_score = recency_factor * recency_weight
        temporal_reliability = round(base_score + recency_score, 3)

        results.append({
            "asset": entry.get("asset"),
            "user_feedback": entry.get("user_feedback"),
            "confidence": confidence,
            "temporal_reliability_score": temporal_reliability,
            "age_days": age_days,
            "timestamp": entry.get("timestamp"),
        })

    return results


def aggregate_feedback_reliability(
    feedback_entries: List[Dict[str, Any]],
    group_by: str = "asset",
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate reliability scores by asset or other grouping.

    Args:
        feedback_entries: List of feedback dictionaries with reliability scores
        group_by: Field to group by (default: "asset")

    Returns:
        Dictionary mapping group keys to aggregated statistics:
        - count: Number of feedback entries
        - avg_confidence: Average confidence
        - avg_reliability: Average reliability score
        - max_reliability: Maximum reliability score
        - min_reliability: Minimum reliability score
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for entry in feedback_entries:
        key = entry.get(group_by, "unknown")
        groups[key].append(entry)

    results = {}
    for key, entries in groups.items():
        confidences = [e.get("confidence", 0.0) for e in entries]
        reliabilities = [
            e.get("reliability_score", e.get("temporal_reliability_score", 0.0))
            for e in entries
        ]

        results[key] = {
            "count": len(entries),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "avg_reliability": round(sum(reliabilities) / len(reliabilities), 3) if reliabilities else 0.0,
            "max_reliability": round(max(reliabilities), 3) if reliabilities else 0.0,
            "min_reliability": round(min(reliabilities), 3) if reliabilities else 0.0,
        }

    return results


def filter_reliable_feedback(
    feedback_entries: List[Dict[str, Any]],
    min_reliability: float = 0.7,
    reliability_key: str = "reliability_score",
) -> List[Dict[str, Any]]:
    """
    Filter feedback entries to only include reliable ones.

    Args:
        feedback_entries: List of feedback dictionaries
        min_reliability: Minimum reliability threshold
        reliability_key: Key for reliability score in entries

    Returns:
        Filtered list of reliable feedback entries
    """
    return [
        entry for entry in feedback_entries
        if entry.get(reliability_key, 0.0) >= min_reliability
    ]


__all__ = [
    'compute_basic_reliability',
    'compute_temporal_reliability',
    'aggregate_feedback_reliability',
    'filter_reliable_feedback',
]
