# signal_engine/threshold/simulator.py
"""
Threshold Simulation Module.

Simulates threshold-based signal filtering and feedback analysis to determine
optimal confidence thresholds for signal generation.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict


def simulate_thresholds(
    feedback_data: List[Dict[str, Any]],
    min_confidence: float = 0.8,
    feedback_type_key: str = "type",
    asset_key: str = "asset",
    confidence_key: str = "confidence",
    agrees_key: str = "agrees_with_signal",
) -> Dict[str, Dict[str, Any]]:
    """
    Simulate threshold-based filtering on feedback data.

    Analyzes user feedback to understand:
    - How often high-confidence signals were disagreed with
    - Average confidence levels per asset
    - Potential signal adjustments needed

    Args:
        feedback_data: List of feedback entries, each containing:
            - type: Entry type (e.g., "user_feedback")
            - asset: Asset symbol/ticker
            - confidence: Model confidence (0-1)
            - agrees_with_signal: Whether user agreed with the signal
        min_confidence: Minimum confidence threshold for flagging disagreements
        feedback_type_key: Key for entry type in data
        asset_key: Key for asset identifier in data
        confidence_key: Key for confidence value in data
        agrees_key: Key for agreement boolean in data

    Returns:
        Dictionary mapping each asset to statistics:
        - total_feedback: Total feedback entries
        - adjusted_signals: Number of high-confidence disagreements
        - avg_confidence: Average confidence across all feedback
        - high_confidence_disagreements: Count of disagreements above threshold
    """
    result = defaultdict(lambda: {
        "total_feedback": 0,
        "adjusted_signals": 0,
        "avg_confidence": 0.0,
        "high_confidence_disagreements": 0,
    })

    for entry in feedback_data:
        # Filter by entry type if specified
        if feedback_type_key in entry and entry.get(feedback_type_key) != "user_feedback":
            continue

        asset = entry.get(asset_key, "unknown")
        confidence = entry.get(confidence_key, 0.0)
        agrees = entry.get(agrees_key, True)

        result[asset]["total_feedback"] += 1
        result[asset]["avg_confidence"] += confidence

        # Flag high-confidence disagreements for potential adjustment
        if confidence >= min_confidence and not agrees:
            result[asset]["adjusted_signals"] += 1
            result[asset]["high_confidence_disagreements"] += 1

    # Compute average confidence
    for asset in result:
        total_feedback = result[asset]["total_feedback"]
        if total_feedback > 0:
            result[asset]["avg_confidence"] = round(
                result[asset]["avg_confidence"] / total_feedback, 3
            )

    return dict(result)


def analyze_threshold_range(
    feedback_data: List[Dict[str, Any]],
    threshold_range: Optional[List[float]] = None,
) -> Dict[float, Dict[str, Any]]:
    """
    Analyze multiple threshold values to find optimal settings.

    Args:
        feedback_data: List of feedback entries
        threshold_range: List of thresholds to test (default: [0.5, 0.6, ..., 0.95])

    Returns:
        Dictionary mapping each threshold to its simulation results
    """
    if threshold_range is None:
        threshold_range = [round(x * 0.05, 2) for x in range(10, 20)]  # 0.5 to 0.95

    results = {}
    for threshold in threshold_range:
        results[threshold] = simulate_thresholds(
            feedback_data,
            min_confidence=threshold
        )

    return results


def find_optimal_threshold(
    feedback_data: List[Dict[str, Any]],
    threshold_range: Optional[List[float]] = None,
    target_disagreement_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Find optimal confidence threshold based on disagreement rate.

    Args:
        feedback_data: List of feedback entries
        threshold_range: List of thresholds to test
        target_disagreement_rate: Target rate of high-confidence disagreements

    Returns:
        Dictionary with:
        - optimal_threshold: Recommended threshold
        - threshold_analysis: Results for all tested thresholds
        - statistics: Overall statistics
    """
    analysis = analyze_threshold_range(feedback_data, threshold_range)

    # Calculate total feedback across all assets
    total_feedback = sum(
        sum(asset_stats["total_feedback"] for asset_stats in threshold_results.values())
        for threshold_results in analysis.values()
    )

    if total_feedback == 0:
        return {
            "optimal_threshold": 0.8,
            "threshold_analysis": analysis,
            "statistics": {"total_feedback": 0, "reason": "No feedback data"},
        }

    # Find threshold closest to target disagreement rate
    best_threshold = None
    best_diff = float('inf')

    for threshold, results in analysis.items():
        total_disagreements = sum(
            asset_stats["high_confidence_disagreements"]
            for asset_stats in results.values()
        )
        disagreement_rate = total_disagreements / max(1, total_feedback)

        diff = abs(disagreement_rate - target_disagreement_rate)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

    return {
        "optimal_threshold": best_threshold or 0.8,
        "threshold_analysis": analysis,
        "statistics": {
            "total_feedback": total_feedback,
            "target_disagreement_rate": target_disagreement_rate,
        },
    }


__all__ = [
    'simulate_thresholds',
    'analyze_threshold_range',
    'find_optimal_threshold',
]
