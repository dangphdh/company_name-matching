"""
Metrics and quality monitoring utilities for the pipeline.

This module provides functions for computing quality metrics, logging stage
performance, and tracking matching quality.
"""

import time
from functools import wraps
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, count, avg, stddev, min as min_, max as max_,
    countDistinct, lit, when
)
from typing import Dict, Any, Optional


def log_stage_metrics(stage_name: str):
    """
    Decorator to log stage execution metrics.

    Args:
        stage_name: Name of the stage for logging

    Usage:
        @log_stage_metrics("Stage 1: Extract")
        def run_stage1(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n{'='*80}")
            print(f"[{stage_name}] Starting...")
            print(f"{'='*80}")

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                print(f"[✅] {stage_name} completed in {elapsed:.2f}s")

                # Record DataFrame metrics if result is a DataFrame
                if isinstance(result, DataFrame):
                    record_count = result.count()
                    print(f"[📊] Records: {record_count:,}")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[❌] {stage_name} failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper
    return decorator


def compute_quality_metrics(matches_df: DataFrame) -> Dict[str, Any]:
    """
    Compute quality metrics for matching results.

    Args:
        matches_df: DataFrame with matching results (must have 'score' column)

    Returns:
        Dictionary of quality metrics
    """
    metrics = {}

    # Basic stats
    total_matches = matches_df.count()
    metrics['total_matches'] = total_matches

    if total_matches == 0:
        print("[⚠️]  No matches found!")
        return metrics

    # Score statistics
    score_stats = matches_df.agg(
        avg('score').alias('avg_score'),
        stddev('score').alias('std_score'),
        min_('score').alias('min_score'),
        max_('score').alias('max_score')
    ).collect()[0]

    metrics['avg_score'] = float(score_stats['avg_score']) if score_stats['avg_score'] else 0.0
    metrics['std_score'] = float(score_stats['std_score']) if score_stats['std_score'] else 0.0
    metrics['min_score'] = float(score_stats['min_score']) if score_stats['min_score'] else 0.0
    metrics['max_score'] = float(score_stats['max_score']) if score_stats['max_score'] else 0.0

    # Confidence distribution
    if 'match_confidence' in matches_df.columns:
        confidence_dist = matches_df.groupBy('match_confidence').count()
        metrics['confidence_distribution'] = {
            row['match_confidence']: row['count']
            for row in confidence_dist.collect()
        }

    # High confidence rate (score >= 0.90)
    high_conf_count = matches_df.filter(col('score') >= 0.90).count()
    metrics['high_confidence_count'] = high_conf_count
    metrics['high_confidence_rate'] = high_conf_count / total_matches if total_matches > 0 else 0.0

    # Medium confidence rate (0.75 <= score < 0.90)
    medium_conf_count = matches_df.filter((col('score') >= 0.75) & (col('score') < 0.90)).count()
    metrics['medium_confidence_count'] = medium_conf_count
    metrics['medium_confidence_rate'] = medium_conf_count / total_matches if total_matches > 0 else 0.0

    # Low confidence rate (score < 0.75)
    low_conf_count = matches_df.filter(col('score') < 0.75).count()
    metrics['low_confidence_count'] = low_conf_count
    metrics['low_confidence_rate'] = low_conf_count / total_matches if total_matches > 0 else 0.0

    return metrics


def print_quality_metrics(metrics: Dict[str, Any]):
    """
    Print quality metrics in a formatted way.

    Args:
        metrics: Dictionary of quality metrics from compute_quality_metrics()
    """
    print(f"\n{'='*80}")
    print("QUALITY METRICS")
    print(f"{'='*80}")

    print(f"Total Matches: {metrics.get('total_matches', 0):,}")

    if 'avg_score' in metrics:
        print(f"\nScore Statistics:")
        print(f"  Average: {metrics['avg_score']:.4f}")
        print(f"  Std Dev:  {metrics['std_score']:.4f}")
        print(f"  Min:      {metrics['min_score']:.4f}")
        print(f"  Max:      {metrics['max_score']:.4f}")

    if 'high_confidence_rate' in metrics:
        print(f"\nConfidence Distribution:")
        print(f"  High (>=0.90):   {metrics['high_confidence_count']:,} ({metrics['high_confidence_rate']:.1%})")
        print(f"  Medium (0.75-0.90): {metrics['medium_confidence_count']:,} ({metrics['medium_confidence_rate']:.1%})")
        print(f"  Low (<0.75):     {metrics['low_confidence_count']:,} ({metrics['low_confidence_rate']:.1%})")

    if 'confidence_distribution' in metrics:
        print(f"\nBy Confidence Label:")
        for label, count in metrics['confidence_distribution'].items():
            print(f"  {label}: {count:,}")

    print(f"{'='*80}\n")


def check_quality_alerts(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> list:
    """
    Check if quality metrics fall below alert thresholds.

    Args:
        metrics: Dictionary of quality metrics
        thresholds: Dictionary of threshold values (e.g., {'avg_score': 0.85})

    Returns:
        List of alert messages (empty if no alerts)
    """
    alerts = []

    # Check average score
    if 'avg_score' in metrics and 'avg_score' in thresholds:
        if metrics['avg_score'] < thresholds['avg_score']:
            alerts.append(
                f"⚠️  ALERT: Average score ({metrics['avg_score']:.4f}) below threshold "
                f"({thresholds['avg_score']:.4f})"
            )

    # Check high confidence rate
    if 'high_confidence_rate' in metrics and 'high_confidence_rate' in thresholds:
        if metrics['high_confidence_rate'] < thresholds['high_confidence_rate']:
            alerts.append(
                f"⚠️  ALERT: High confidence rate ({metrics['high_confidence_rate']:.1%}) below threshold "
                f"({thresholds['high_confidence_rate']:.1%})"
            )

    return alerts


def save_metrics(metrics: Dict[str, Any], path: str, stage_name: str = "pipeline"):
    """
    Save metrics to a file for tracking over time.

    Args:
        metrics: Dictionary of metrics
        path: Directory to save metrics file
        stage_name: Name of the stage/pipeline
    """
    import json
    from datetime import datetime
    from pathlib import Path

    # Create metrics directory if needed
    metrics_dir = Path(path)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['stage'] = stage_name

    # Save to file
    metrics_file = metrics_dir / f"{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[Metrics] Saved to: {metrics_file}")
