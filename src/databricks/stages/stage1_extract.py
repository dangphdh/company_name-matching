"""
Stage 1: Extract and Transform

This stage extracts company names from the Bronze Delta Lake table and applies
Vietnamese text preprocessing using Pandas UDFs for vectorized processing.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, pandas_udf, current_timestamp, lit
import pandas as pd

from src.databricks.preprocessing.vietnamese_udfs import preprocess_batch_udf
from src.databricks.utils.delta_utils import write_delta
from src.databricks.utils.validation import validate_company_names
from src.databricks.utils.metrics import log_stage_metrics


@log_stage_metrics("Stage 1: Extract & Preprocess")
def run_stage1_extract(
    spark: SparkSession,
    bronze_path: str,
    silver_path: str,
    num_partitions: int = 200,
    validate: bool = True
) -> DataFrame:
    """
    Stage 1: Extract from Bronze and apply Vietnamese preprocessing.

    This stage:
    1. Reads raw company names from Bronze Delta table
    2. Applies Vietnamese text normalization and cleaning
    3. Generates cleaned, no-accent, and normalization key variants
    4. Validates data quality
    5. Writes processed data to Silver layer

    Args:
        spark: SparkSession
        bronze_path: Path to Bronze Delta table (raw input)
        silver_path: Path to Silver Delta table (processed output)
        num_partitions: Target number of partitions for output
        validate: Whether to apply data validation

    Returns:
        DataFrame with processed company names (cached)
    """
    print(f"\n[Stage 1] Configuration:")
    print(f"  Bronze path:     {bronze_path}")
    print(f"  Silver path:     {silver_path}")
    print(f"  Partitions:      {num_partitions}")
    print(f"  Validation:      {validate}")

    # Step 1: Read from Bronze table
    print(f"\n[Stage 1] Step 1: Reading from Bronze...")
    df_bronze = spark.read.format("delta").load(bronze_path)

    input_count = df_bronze.count()
    print(f"[Stage 1] Input records: {input_count:,}")

    # Validate schema
    required_columns = ["company_name"]  # Adjust based on your actual schema
    # Optional: Check if required columns exist
    # from src.databricks.utils.validation import validate_schema
    # validate_schema(df_bronze, required_columns)

    # Step 2: Apply data validation if enabled
    if validate:
        print(f"\n[Stage 1] Step 2: Validating company names...")
        df_bronze, validation_stats = validate_company_names(df_bronze, "company_name")

        # Warn if many records were filtered
        invalid_rate = validation_stats['invalid_count'] / validation_stats['input_count']
        if invalid_rate > 0.10:  # More than 10% invalid
            print(f"[⚠️]  Warning: {invalid_rate:.1%} of records filtered during validation")

    # Step 3: Apply Vietnamese preprocessing with Pandas UDF
    print(f"\n[Stage 1] Step 3: Applying Vietnamese preprocessing...")

    # Apply the preprocessing UDF
    df_processed = df_bronze.select(
        "*",  # Preserve all original columns
        preprocess_batch_udf(col("company_name")).alias("preprocessed")
    )

    # Extract preprocessed fields
    df_cleaned = df_processed.select(
        # Keep original columns
        col("*"),
        # Extract preprocessed fields
        col("preprocessed.cleaned").alias("cleaned_name"),
        col("preprocessed.no_accent").alias("no_accent_name"),
        col("preprocessed.norm_key").alias("norm_key")
    ).drop("preprocessed")

    # Step 4: Add metadata
    print(f"\n[Stage 1] Step 4: Adding metadata...")
    df_cleaned = df_cleaned.withColumn(
        "processed_timestamp",
        current_timestamp()
    )

    # Step 5: Repartition for downstream processing
    print(f"\n[Stage 1] Step 5: Repartitioning to {num_partitions} partitions...")
    df_cleaned = df_cleaned.repartition(num_partitions, col("norm_key"))

    # Cache for reuse in downstream stages
    df_cleaned = df_cleaned.cache()

    print(f"[Stage 1] Cached processed data")

    # Step 6: Write to Silver layer
    print(f"\n[Stage 1] Step 6: Writing to Silver layer...")
    write_delta(df_cleaned, silver_path, mode="overwrite")

    # Print summary
    output_count = df_cleaned.count()
    print(f"\n[Stage 1] Summary:")
    print(f"  Input records:  {input_count:,}")
    print(f"  Output records: {output_count:,}")
    print(f"  Reduction rate: {(1 - output_count/input_count)*100:.1f}%")

    # Sample some results
    print(f"\n[Stage 1] Sample processed records:")
    df_cleaned.select(
        "company_name",
        "cleaned_name",
        "no_accent_name",
        "norm_key"
    ).show(5, truncate=False)

    return df_cleaned
