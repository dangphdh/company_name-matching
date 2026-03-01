"""
Data validation utilities for the pipeline.

This module provides functions for validating company names, matches, and
ensuring data quality throughout the pipeline.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, length, trim, regexp_extract, when, lit
from typing import Tuple


def validate_company_names(df: DataFrame, name_column: str = "company_name") -> Tuple[DataFrame, dict]:
    """
    Validate company names and apply filtering rules.

    Args:
        df: Input DataFrame with company names
        name_column: Name of the column containing company names

    Returns:
        Tuple of (validated DataFrame, validation stats dict)
    """
    stats = {
        'input_count': 0,
        'null_count': 0,
        'empty_count': 0,
        'too_short_count': 0,
        'too_long_count': 0,
        'special_only_count': 0,
        'valid_count': 0
    }

    print(f"[Validation] Validating company names in column: {name_column}")

    # Count input
    stats['input_count'] = df.count()

    # Check for nulls
    null_count = df.filter(col(name_column).isNull()).count()
    stats['null_count'] = null_count

    # Remove nulls
    df = df.filter(col(name_column).isNotNull())

    # Trim whitespace
    df = df.withColumn(name_column, trim(col(name_column)))

    # Check for empty strings (after trim)
    empty_count = df.filter(col(name_column) == "").count()
    stats['empty_count'] = empty_count

    # Remove empty strings
    df = df.filter(col(name_column) != "")

    # Check length constraints
    # Minimum: 3 characters (after cleaning)
    # Maximum: 200 characters
    df = df.withColumn(f"{name_column}_length", length(col(name_column)))

    too_short_count = df.filter(col(f"{name_column}_length") < 3).count()
    stats['too_short_count'] = too_short_count

    too_long_count = df.filter(col(f"{name_column}_length") > 200).count()
    stats['too_long_count'] = too_long_count

    # Apply length filters
    df = df.filter((col(f"{name_column}_length") >= 3) & (col(f"{name_column}_length") <= 200))
    df = df.drop(f"{name_column}_length")

    # Check for names with only special characters
    # A valid name should contain at least some alphanumeric or Vietnamese characters
    special_only_pattern = r"^[^a-zA-Z0-9\u00c0-\u00ff\u0102-\u0103\u0110-\u0111\u1ea0-\u1ef9]+$"

    df = df.withColumn(
        "is_special_only",
        regexp_extract(col(name_column), special_only_pattern, 0) != ""
    )

    special_only_count = df.filter(col("is_special_only") == True).count()
    stats['special_only_count'] = special_only_count

    # Remove special-only names
    df = df.filter(col("is_special_only") == False).drop("is_special_only")

    # Count valid records
    stats['valid_count'] = df.count()

    # Print validation summary
    print(f"[Validation] Summary:")
    print(f"  Input:        {stats['input_count']:,}")
    print(f"  Null:         {stats['null_count']:,} (removed)")
    print(f"  Empty:        {stats['empty_count']:,} (removed)")
    print(f"  Too short:    {stats['too_short_count']:,} (removed)")
    print(f"  Too long:     {stats['too_long_count']:,} (removed)")
    print(f"  Special only: {stats['special_only_count']:,} (removed)")
    print(f"  Valid:        {stats['valid_count']:,} ({stats['valid_count']/stats['input_count']*100:.1f}%)")

    # Calculate invalid count
    stats['invalid_count'] = (
        stats['null_count'] +
        stats['empty_count'] +
        stats['too_short_count'] +
        stats['too_long_count'] +
        stats['special_only_count']
    )

    return df, stats


def validate_matches(matches_df: DataFrame) -> Tuple[DataFrame, dict]:
    """
    Validate matching results and ensure data quality.

    Args:
        matches_df: DataFrame with matching results

    Returns:
        Tuple of (validated DataFrame, validation stats dict)
    """
    stats = {
        'input_count': 0,
        'null_score_count': 0,
        'invalid_score_count': 0,
        'no_match_count': 0,
        'valid_count': 0
    }

    print(f"[Validation] Validating matches")

    # Count input
    stats['input_count'] = matches_df.count()

    # Check for null scores
    if 'score' in matches_df.columns:
        null_score_count = matches_df.filter(col('score').isNull()).count()
        stats['null_score_count'] = null_score_count

        # Remove null scores
        matches_df = matches_df.filter(col('score').isNotNull())

        # Check for invalid scores (outside [0, 1])
        invalid_score_count = matches_df.filter(
            (col('score') < 0) | (col('score') > 1)
        ).count()
        stats['invalid_score_count'] = invalid_score_count

        # Remove invalid scores
        matches_df = matches_df.filter((col('score') >= 0) & (col('score') <= 1))

        # Check for zero scores (no matches)
        no_match_count = matches_df.filter(col('score') == 0).count()
        stats['no_match_count'] = no_match_count

    # Count valid records
    stats['valid_count'] = matches_df.count()

    # Print validation summary
    print(f"[Validation] Summary:")
    print(f"  Input:          {stats['input_count']:,}")
    print(f"  Null scores:    {stats['null_score_count']:,} (removed)")
    print(f"  Invalid scores: {stats['invalid_score_count']:,} (removed)")
    print(f"  No matches:     {stats['no_match_count']:,}")
    print(f"  Valid:          {stats['valid_count']:,}")

    return matches_df, stats


def validate_partition_distribution(df: DataFrame, partition_col: str = "norm_key") -> dict:
    """
    Validate data distribution across partitions to detect skew.

    Args:
        df: DataFrame to check
        partition_col: Column to check distribution for

    Returns:
        Dictionary with distribution stats
    """
    print(f"[Validation] Checking partition distribution for: {partition_col}")

    # Get record count per partition
    # (This requires RDD operations)
    partition_counts = df.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()

    if not partition_counts:
        return {'error': 'No partitions found'}

    # Compute statistics
    min_size = min(partition_counts)
    max_size = max(partition_counts)
    avg_size = sum(partition_counts) / len(partition_counts)
    num_partitions = len(partition_counts)

    # Check for skew (max > 3x average is considered skewed)
    is_skewed = max_size > (avg_size * 3)

    stats = {
        'num_partitions': num_partitions,
        'min_partition_size': int(min_size),
        'max_partition_size': int(max_size),
        'avg_partition_size': float(avg_size),
        'is_skewed': is_skewed,
        'skew_ratio': float(max_size / avg_size) if avg_size > 0 else 0
    }

    print(f"[Validation] Partition Distribution:")
    print(f"  Partitions:    {stats['num_partitions']}")
    print(f"  Min size:      {stats['min_partition_size']:,}")
    print(f"  Max size:      {stats['max_partition_size']:,}")
    print(f"  Avg size:      {stats['avg_partition_size']:.0f}")
    print(f"  Skew ratio:    {stats['skew_ratio']:.2f}x")
    print(f"  Skewed:        {'⚠️  YES' if stats['is_skewed'] else '✓ NO'}")

    if stats['is_skewed']:
        print(f"[⚠️]  Data is skewed! Consider repartitioning on: {partition_col}")

    return stats


def validate_schema(df: DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame has all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if all columns present, False otherwise
    """
    existing_columns = set(df.columns)
    missing_columns = set(required_columns) - existing_columns

    if missing_columns:
        print(f"[❌] Missing required columns: {missing_columns}")
        return False

    print(f"[✓] All required columns present: {required_columns}")
    return True
