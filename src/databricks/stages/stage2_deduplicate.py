"""
Stage 2: Deduplicate

This stage removes duplicate company names based on their normalized keys.
It handles various deduplication strategies (first occurrence, longest name, etc.).
"""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (
    col, count, collect_list, first, length, row_number,
    monotonically_increasing_id, when, lit, array_contains, size
)

from src.databricks.utils.delta_utils import write_delta
from src.databricks.utils.metrics import log_stage_metrics


@log_stage_metrics("Stage 2: Deduplicate")
def run_stage2_deduplicate(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    dedup_strategy: str = "first",
    keep_originals: bool = True
) -> DataFrame:
    """
    Stage 2: Deduplicate companies by normalization key.

    This stage:
    1. Groups companies by norm_key (no-accent cleaned name)
    2. Identifies duplicate groups
    3. Selects canonical record based on strategy
    4. Tracks duplicate groups for audit
    5. Writes deduplicated data to output

    Args:
        spark: SparkSession
        input_path: Path to input Delta table (from Stage 1)
        output_path: Path to output Delta table (deduplicated)
        dedup_strategy: "first" (keep first) or "longest" (keep longest name)
        keep_originals: Whether to keep original IDs for tracking

    Returns:
        DataFrame with deduplicated companies (cached)
    """
    print(f"\n[Stage 2] Configuration:")
    print(f"  Input path:       {input_path}")
    print(f"  Output path:      {output_path}")
    print(f"  Strategy:         {dedup_strategy}")
    print(f"  Keep originals:   {keep_originals}")

    # Step 1: Read preprocessed data
    print(f"\n[Stage 2] Step 1: Reading preprocessed data...")
    df = spark.read.format("delta").load(input_path)

    input_count = df.count()
    print(f"[Stage 2] Input records: {input_count:,}")

    # Step 2: Identify duplicate groups
    print(f"\n[Stage 2] Step 2: Identifying duplicate groups...")

    # Group by norm_key and count occurrences
    duplicate_groups = df.groupBy("norm_key").agg(
        count("*").alias("duplicate_count"),
        collect_list("company_name").alias("original_names"),
        collect_list("cleaned_name").alias("cleaned_names")
    )

    # Filter to actual duplicates (count > 1)
    duplicates = duplicate_groups.filter(col("duplicate_count") > 1)
    duplicate_group_count = duplicates.count()

    print(f"[Stage 2] Found {duplicate_group_count:,} duplicate groups")
    print(f"[Stage 2] Total records in duplicate groups: {duplicates.agg({'duplicate_count': 'sum'}).collect()[0]['sum(duplicate_count)']:,}")

    # Step 3: Select canonical record based on strategy
    print(f"\n[Stage 2] Step 3: Applying deduplication strategy: {dedup_strategy}")

    if dedup_strategy == "longest":
        # Keep the longest cleaned name (most complete)
        df_with_len = df.withColumn("name_length", length(col("cleaned_name")))

        window_spec = Window.partitionBy("norm_key").orderBy(
            col("name_length").desc(),
            # If lengths are equal, use ID as tiebreaker
            monotonically_increasing_id().asc()
        )

        df_canonical = df_with_len.withColumn("rank", row_number().over(window_spec))
        df_canonical = df_canonical.filter(col("rank") == 1).drop("rank", "name_length")

    elif dedup_strategy == "first":
        # Keep first occurrence (by existing order or ID)
        window_spec = Window.partitionBy("norm_key").orderBy(
            monotonically_increasing_id().asc()
        )

        df_canonical = df.withColumn("rank", row_number().over(window_spec))
        df_canonical = df_canonical.filter(col("rank") == 1).drop("rank")

    else:
        raise ValueError(f"Unknown dedup strategy: {dedup_strategy}")

    # Add sequential canonical_id
    df_canonical = df_canonical.orderBy("norm_key")
    df_canonical = df_canonical.withColumn(
        "canonical_id",
        monotonically_increasing_id()
    )

    canonical_count = df_canonical.count()
    print(f"[Stage 2] Canonical records: {canonical_count:,}")

    # Step 4: Join duplicate group info (for audit)
    print(f"\n[Stage 2] Step 4: Adding duplicate group metadata...")

    df_dedup = df_canonical.join(
        duplicate_groups.select("norm_key", "duplicate_count", "original_names"),
        on="norm_key",
        how="left"
    )

    # Add flags
    df_dedup = df_dedup.withColumn(
        "is_duplicate",
        when(col("duplicate_count") > 1, lit(True)).otherwise(lit(False))
    )

    df_dedup = df_dedup.withColumn(
        "duplicate_group_size",
        col("duplicate_count")
    )

    # Step 5: Write deduplicated data
    print(f"\n[Stage 2] Step 5: Writing deduplicated data...")

    # Select output columns
    output_columns = [
        "canonical_id",
        "company_name",
        "cleaned_name",
        "no_accent_name",
        "norm_key",
        "is_duplicate",
        "duplicate_group_size",
        "original_names",  # For audit
        "processed_timestamp"
    ]

    # Only include columns that exist
    output_columns = [c for c in output_columns if c in df_dedup.columns]

    df_output = df_dedup.select(*output_columns)

    write_delta(df_output, output_path, mode="overwrite")

    # Cache for downstream use
    df_output = df_output.cache()

    # Step 6: Print summary
    print(f"\n[Stage 2] Summary:")
    print(f"  Input records:        {input_count:,}")
    print(f"  Output records:       {canonical_count:,}")
    print(f"  Duplicates removed:   {input_count - canonical_count:,}")
    print(f"  Reduction rate:       {(1 - canonical_count/input_count)*100:.1f}%")
    print(f"  Duplicate groups:     {duplicate_group_count:,}")

    # Sample duplicate groups
    if duplicate_group_count > 0:
        print(f"\n[Stage 2] Sample duplicate groups:")
        duplicates.select(
            "norm_key",
            "duplicate_count",
            "original_names"
        ).show(5, truncate=False)

    return df_output
