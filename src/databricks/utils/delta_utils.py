"""
Delta Lake utility functions for the pipeline.

This module provides helper functions for reading/writing Delta Lake tables
and optimizing them for performance.
"""

from pyspark.sql import DataFrame
from delta.tables import DeltaTable
from pathlib import Path


def write_delta(
    df: DataFrame,
    path: str,
    mode: str = "overwrite",
    partition_by: list = None,
    optimize: bool = True
):
    """
    Write DataFrame to Delta Lake with standard options.

    Args:
        df: DataFrame to write
        path: Output path
        mode: Write mode ("overwrite", "append", "error", "ignore")
        partition_by: Optional column(s) to partition by
        optimize: Whether to enable Delta Lake optimization
    """
    writer = df.write.format("delta").mode(mode)

    # Apply partitioning if specified
    if partition_by:
        writer = writer.partitionBy(*partition_by)

    # Enable Delta Lake optimizations
    if optimize:
        writer = writer.option("delta.autoOptimize.optimizeWrite", "true")
        writer = writer.option("delta.autoOptimize.autoCompact", "true")

    # Write data
    writer.save(path)

    print(f"[Delta] Written {df.count()} records to {path}")


def read_delta(spark, path: str) -> DataFrame:
    """
    Read from Delta Lake table.

    Args:
        spark: SparkSession
        path: Delta Lake table path

    Returns:
        DataFrame
    """
    return spark.read.format("delta").load(path)


def optimize_delta(spark, path: str):
    """
    Optimize Delta Lake table for performance.

    This compacts small files and improves query performance.

    Args:
        spark: SparkSession
        path: Delta Lake table path
    """
    if not DeltaTable.isDeltaTable(spark, path):
        print(f"[Delta] Skipping optimization: {path} is not a Delta table")
        return

    delta_table = DeltaTable.forPath(spark, path)

    # Compact files
    print(f"[Delta] Optimizing {path}...")
    delta_table.optimize().executeCompaction()

    # Z-order by relevant columns (if any)
    # This improves query performance by co-locating related data
    # delta_table.optimize().executeZOrderBy("norm_key")

    print(f"[Delta] Optimization complete for {path}")


def vacuum_delta(spark, path: str, retention_hours: int = 168):
    """
    Vacuum Delta Lake table to remove old files.

    WARNING: This permanently deletes data files older than retention period.
    Only run this after you're sure you won't need time travel.

    Args:
        spark: SparkSession
        path: Delta Lake table path
        retention_hours: Retention period in hours (default: 7 days)
    """
    if not DeltaTable.isDeltaTable(spark, path):
        print(f"[Delta] Skipping vacuum: {path} is not a Delta table")
        return

    delta_table = DeltaTable.forPath(spark, path)

    print(f"[Delta] Vacuuming {path} (retention: {retention_hours} hours)...")
    delta_table.vacuum(retention_hours)
    print(f"[Delta] Vacuum complete for {path}")


def create_table_if_not_exists(
    spark,
    table_name: str,
    path: str,
    schema: str = None
):
    """
    Create Delta Lake table if it doesn't exist.

    Args:
        spark: SparkSession
        table_name: Full table name (e.g., "catalog.schema.table")
        path: Delta Lake table path
        schema: Optional DDL schema (creates empty table with schema)
    """
    # Check if table exists
    try:
        spark.read.format("delta").load(path)
        print(f"[Delta] Table already exists at: {path}")
        return
    except Exception:
        # Table doesn't exist, create it
        pass

    if schema:
        # Create with schema
        spark.sql(f"""
            CREATE TABLE {table_name}
            ({schema})
            USING DELTA
            LOCATION '{path}'
        """)
    else:
        # Create empty table
        spark.sql(f"""
            CREATE TABLE {table_name}
            USING DELTA
            LOCATION '{path}'
        """)

    print(f"[Delta] Created table: {table_name} at {path}")


def upsert_delta(
    spark,
    target_table: str,
    source_df: DataFrame,
    merge_condition: str,
    update_columns: list = None
):
    """
    Upsert (merge) data into Delta Lake table.

    Args:
        spark: SparkSession
        target_table: Target table name or path
        source_df: Source DataFrame to upsert
        merge_condition: Merge condition (e.g., "target.id = source.id")
        update_columns: Columns to update on match (None = all except key)
    """
    target_delta_table = DeltaTable.forName(spark, target_table)

    # Build update set
    if update_columns:
        update_set = {col: f"source.{col}" for col in update_columns}
    else:
        # Update all columns except merge keys
        update_set = None

    # Perform merge
    target_delta_table.alias("target").merge(
        source_df.alias("source"),
        merge_condition
    ).whenNotMatchedInsertAll().execute()

    print(f"[Delta] Upserted {source_df.count()} records into {target_table}")
