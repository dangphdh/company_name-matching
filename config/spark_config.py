"""
Spark Configuration for Local and Databricks Experimentation
"""
import os
from pathlib import Path

def get_local_spark_config(
    app_name: str = "CompanyMatching",
    executor_memory: str = "4g",
    driver_memory: str = "4g",
    cores: int = 4,
) -> dict:
    """
    Get Spark configuration for local development.

    Args:
        app_name: Name of the Spark application
        executor_memory: Memory per executor (e.g., "4g", "8g")
        driver_memory: Memory for driver (e.g., "4g", "8g")
        cores: Number of CPU cores to use

    Returns:
        Dictionary of Spark configuration
    """
    return {
        "spark.app.name": app_name,
        "spark.master": f"local[{cores}]",
        "spark.executor.memory": executor_memory,
        "spark.driver.memory": driver_memory,
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        # Vietnamese text handling
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
    }


def get_databricks_connect_config(
    cluster_id: str = None,
    workspace_url: str = None,
    token: str = None,
) -> dict:
    """
    Get Databricks Connect configuration.

    Args:
        cluster_id: Databricks cluster ID (from DATABRICKS_CLUSTER_ID env var)
        workspace_url: Databricks workspace URL (from DATABRICKS_WORKSPACE_URL env var)
        token: Databricks personal access token (from DATABRICKS_TOKEN env var)

    Returns:
        Dictionary of Databricks configuration

    Environment Variables (alternatively):
        - DATABRICKS_CLUSTER_ID
        - DATABRICKS_WORKSPACE_URL
        - DATABRICKS_TOKEN
    """
    return {
        "cluster_id": cluster_id or os.getenv("DATABRICKS_CLUSTER_ID"),
        "workspace_url": workspace_url or os.getenv("DATABRICKS_WORKSPACE_URL"),
        "token": token or os.getenv("DATABRICKS_TOKEN"),
    }


def create_spark_session(
    use_databricks: bool = False,
    databricks_config: dict = None,
    local_config: dict = None,
):
    """
    Create and configure a Spark session.

    Args:
        use_databricks: Whether to use Databricks Connect (requires databricks-connect package)
        databricks_config: Databricks configuration dict
        local_config: Local Spark configuration dict

    Returns:
        SparkSession
    """
    from pyspark.sql import SparkSession

    if use_databricks:
        # For Databricks Connect, you need to install databricks-connect separately
        # Then configure using environment variables or databricks CLI
        try:
            from databricks.connect import DatabricksSession

            # For Databricks Connect, use DatabricksSession
            builder = DatabricksSession.builder

            # Set cluster info if provided
            if databricks_config:
                if databricks_config.get("cluster_id"):
                    builder = builder.clusterId(databricks_config["cluster_id"])
                if databricks_config.get("workspace_url"):
                    builder = builder.workspaceUrl(databricks_config["workspace_url"])
                if databricks_config.get("token"):
                    builder = builder.token(databricks_config["token"])

            return builder.getOrCreate()

        except ImportError:
            raise ImportError(
                "Databricks Connect is not installed. "
                "To use Databricks, install it separately:\n"
                "  pip install databricks-connect==14.0.*  # Match your Databricks runtime version\n"
                "Then configure with: databricks configure --configure-cluster"
            )

    else:
        # Local Spark session
        config = local_config or get_local_spark_config()

        builder = SparkSession.builder

        for key, value in config.items():
            if key != "spark.app.name":
                builder = builder.config(key, value)

        return builder.appName(config["spark.app.name"]).getOrCreate()


def setup_delta_lake(spark: "SparkSession"):
    """
    Configure Delta Lake for Spark session.

    Args:
        spark: SparkSession
    """
    from delta import configure_spark_with_delta_pip

    builder = configure_spark_with_delta_pip(
        SparkSession.builder.builder.config(
            "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
        ).config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    return builder.getOrCreate()
