#!/usr/bin/env python3
"""
Example: Using the Databricks Pipeline for Vietnamese Company Name Matching

This script demonstrates how to:
1. Initialize the pipeline
2. Run all 4 stages
3. Query results
4. Run matching only (against existing index)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from src.databricks.orchestrator import PipelineOrchestrator
from src.databricks.config import load_config


def main():
    """Main example function."""

    print("="*80)
    print("Databricks Pipeline Example")
    print("="*80)

    # ========================================================================
    # Setup 1: Initialize Spark
    # ========================================================================
    print("\n[Step 1] Initializing Spark...")

    spark = SparkSession.builder \
        .appName("DatabricksPipelineExample") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    print(f"Spark version: {spark.version}")

    # ========================================================================
    # Setup 2: Load configuration
    # ========================================================================
    print("\n[Step 2] Loading configuration...")

    # Option A: Load from YAML file (with profile)
    config = load_config(profile="dev")  # Uses dev profile (smaller scale)

    # Option B: Override specific settings
    config.model_name = "tfidf-lsa"
    config.lsa_dims = 128  # Smaller for example
    config.stage1_partitions = 10

    print(f"Model: {config.model_name}")
    print(f"LSA dimensions: {config.lsa_dims}")

    # ========================================================================
    # Setup 3: Create sample data
    # ========================================================================
    print("\n[Step 3] Creating sample data...")

    # Sample Bronze data (companies)
    companies = [
        ("1", "CÔNG TY TNHH SỮA VIỆT NAM"),
        ("2", "Cty TNHH Sữa Việt Nam"),  # Duplicate
        ("3", "Ngân hàng TMCP Ngoại thương"),
        ("4", "Vietcombank"),  # Variant
        ("5", "Tập đoàn Viễn thông Quân đội"),
        ("6", "Viettel"),  # Variant
        ("7", "Tổng công ty CP Đầu tư và Phát triển Điện lực"),
        ("8", "EVN"),  # Abbreviation
        ("9", "CÔNG TY CỔ PHẦN Hàng không Việt Nam"),
        ("10", "Vietnam Airlines")
    ]

    bronze_df = spark.createDataFrame(
        companies,
        ["id", "company_name"]
    )

    # Write to Bronze (in production, this would be your actual data)
    bronze_path = "/tmp/bronze_companies"
    bronze_df.write.format("delta").mode("overwrite").save(bronze_path)
    print(f"Created Bronze table: {bronze_df.count()} companies")

    # Sample queries
    queries = [
        (1, "Vinamilk"),
        (2, "Vietcombank"),
        (3, "Viettel"),
        (4, "EVN"),
        (5, "Vietnam Airlines")
    ]

    queries_df = spark.createDataFrame(
        queries,
        ["query_id", "query_text"]
    )

    print(f"Created {queries_df.count()} queries")

    # ========================================================================
    # Pipeline Run: Full Pipeline (Stages 1-4)
    # ========================================================================
    print("\n" + "="*80)
    print("Running Full Pipeline (Stages 1-4)")
    print("="*80)

    # Initialize pipeline
    pipeline = PipelineOrchestrator(config)

    # Override paths for this example
    silver_path = "/tmp/silver_companies"
    gold_index_path = "/tmp/gold_index"
    gold_matches_path = "/tmp/gold_matches"

    try:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            bronze_path=bronze_path,
            queries_df=queries_df
        )

        # ========================================================================
        # Query Results
        # ========================================================================
        print("\n[Step 4] Querying results...")

        matches_df = spark.read.format("delta").load(gold_matches_path)

        print("\nTop matches per query:")
        matches_df.filter(col("rank") == 1).select(
            "query_id",
            "query_text",
            "matched_company",
            "score",
            "match_confidence"
        ).show(truncate=False)

        # ========================================================================
        # Pipeline Run: Matching Only (Reuse Existing Index)
        # ========================================================================
        print("\n" + "="*80)
        print("Running Matching Only (Stage 4)")
        print("="*80)

        # Create new queries
        new_queries = spark.createDataFrame([
            (6, "Sữa Việt Nam"),
            (7, "Ngân hàng ngoại thương")
        ]).toDF("query_id", "query_text")

        # Run matching only (reuses existing index)
        matches_new = pipeline.run_matching_only(new_queries)

        print("\nNew query results:")
        matches_new.select(
            "query_id",
            "query_text",
            "matched_company",
            "score"
        ).show(truncate=False)

    finally:
        # ========================================================================
        # Cleanup
        # ========================================================================
        print("\n[Step 5] Cleaning up...")

        pipeline.cleanup()
        spark.stop()

        print("\n" + "="*80)
        print("Example completed successfully!")
        print("="*80)


if __name__ == "__main__":
    main()
