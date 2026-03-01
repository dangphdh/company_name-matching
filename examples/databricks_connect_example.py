"""
Example: Databricks Connect for Remote Development
This script demonstrates connecting to a remote Databricks cluster for development.

Prerequisites:
1. Install databricks-connect: pip install databricks-connect==14.0.* (match your Databricks runtime)
2. Configure environment variables in .env:
   - DATABRICKS_WORKSPACE_URL
   - DATABRICKS_CLUSTER_ID
   - DATABRICKS_TOKEN

Or use Databricks CLI:
    databricks configure --configure-cluster
"""
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from config.spark_config import create_spark_session, get_databricks_connect_config


def setup_databricks_connect():
    """
    Setup Databricks Connect connection.

    Note: For Databricks Connect 14.0+, you can use the remote() API.
    Alternatively, configure using databricks CLI or environment variables.
    """
    # Method 1: Using environment variables (recommended)
    # Set in .env or export:
    # export DATABRICKS_ADDRESS=<workspace-url>
    # export DATABRICKS_TOKEN=<personal-access-token>
    # export DATABRICKS_CLUSTER_ID=<cluster-id>

    # Method 2: Direct configuration
    config = get_databricks_connect_config()

    print("Databricks Configuration:")
    print(f"  Workspace URL: {config['workspace_url']}")
    print(f"  Cluster ID: {config['cluster_id']}")
    print(f"  Token: {'*' * 20}{config['token'][-4:] if config['token'] else 'NOT SET'}")

    if not all([config['workspace_url'], config['cluster_id'], config['token']]):
        raise ValueError(
            "Missing Databricks configuration. Please set the following environment variables:\n"
            "  - DATABRICKS_WORKSPACE_URL\n"
            "  - DATABRICKS_CLUSTER_ID\n"
            "  - DATABRICKS_TOKEN\n\n"
            "Or run: databricks configure --configure-cluster"
        )

    return config


def example_databricks_operations(spark):
    """
    Run example operations on Databricks cluster.

    Args:
        spark: SparkSession connected to Databricks
    """
    print("\n=== Databricks Cluster Info ===")
    print(f"Spark version: {spark.version}")
    print(f"Spark master: {spark.conf.get('spark.master')}")

    # Check if running on Databricks
    try:
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
        print("Running on Databricks cluster!")

        # Get cluster info
        print("\n=== Cluster Info ===")
        try:
            cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
            print(f"Cluster ID: {cluster_id}")
        except:
            pass
    except:
        print("Warning: DBUtils not available. May not be on Databricks.")

    # Test basic operations
    print("\n=== Test Operations ===")

    # Create test DataFrame
    test_data = [
        ("CÔNG TY TNHH SỮA VIỆT NAM", "Vinamilk"),
        ("Ngân hàng TMCP Ngoại thương", "Vietcombank"),
        ("Tập đoàn Hòa Phát", "Hoa Phat Group"),
    ]
    df = spark.createDataFrame(test_data, ["full_name", "short_name"])

    print("\nTest DataFrame:")
    df.show(truncate=False)

    # Register as temp view for SQL
    df.createOrReplaceTempView("companies")
    result = spark.sql("SELECT * FROM companies WHERE short_name LIKE 'V%'")
    print("\nSQL Query Result:")
    result.show(truncate=False)


def upload_data_to_dbfs(spark, local_path: str, dbfs_path: str):
    """
    Upload local data to DBFS (Databricks File System).

    Args:
        spark: SparkSession
        local_path: Local file path
        dbfs_path: DBFS path (e.g., "/FileStore/data/companies.txt")

    Note: Requires dbutils (available on Databricks clusters)
    """
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)

    print(f"\nUploading {local_path} to {dbfs_path}...")

    # Create directory if needed
    dbutils.fs.mkdirs(dbfs_path.rsplit('/', 1)[0])

    # Upload file
    dbutils.fs.cp(f"file://{local_path}", dbfs_path)

    print(f"Upload complete!")


def example_delta_table_operations(spark):
    """
    Example operations with Delta Lake tables on Databricks.

    Args:
        spark: SparkSession
    """
    print("\n=== Delta Table Operations ===")

    # Create a Delta table
    data = [
        ("1", "CÔNG TY TNHH SỮA VIỆT NAM", "Vinamilk", "Food & Beverage"),
        ("2", "Ngân hàng TMCP Ngoại thương", "Vietcombank", "Banking"),
        ("3", "Tập đoàn Hòa Phát", "Hoa Phat", "Manufacturing"),
    ]

    df = spark.createDataFrame(data, ["id", "full_name", "brand_name", "industry"])

    # Write to Delta table
    delta_path = "/tmp/companies_delta"
    df.write.format("delta").mode("overwrite").save(delta_path)

    print(f"Delta table written to: {delta_path}")

    # Read back
    delta_df = spark.read.format("delta").load(delta_path)
    print("\nReading from Delta table:")
    delta_df.show(truncate=False)

    # Time travel example (query previous version)
    print("\n=== Time Travel Example ===")
    print(f"Available versions: Check history at {delta_path}")

    # You can query specific versions:
    # spark.read.format("delta").option("versionAsOf", 0).load(delta_path).show()


def main():
    """Main function for Databricks Connect example."""
    print("=== Databricks Connect Example ===\n")

    try:
        # Setup connection
        config = setup_databricks_connect()

        # Connect to remote cluster
        print("\nConnecting to Databricks cluster...")
        spark = create_spark_session(
            use_databricks=True,
            databricks_config=config
        )
        print("Connected successfully!")

        try:
            # Run examples
            example_databricks_operations(spark)
            example_delta_table_operations(spark)

            print("\n=== Examples Complete ===")

        finally:
            spark.stop()
            print("Spark session stopped.")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure databricks-connect is installed: pip install databricks-connect==14.0.*")
        print("2. Configure using: databricks configure --configure-cluster")
        print("3. Or set environment variables in .env file")


if __name__ == "__main__":
    main()
