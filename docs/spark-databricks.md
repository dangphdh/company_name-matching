# Spark & Databricks Environment Setup

This directory contains examples and configuration for experimenting with Apache Spark and Databricks for company name matching at scale.

## Overview

The setup supports two modes:

1. **Local Spark**: Run Spark locally on your machine for development and testing
2. **Databricks Connect**: Connect to a remote Databricks cluster for production-scale experimentation

## Installation

### Step 1: Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Spark and Databricks dependencies
pip install -r requirements-spark.txt
```

**Note for Databricks Connect:**
The version of `databricks-connect` must match your Databricks runtime version.
Check your cluster's runtime version in Databricks UI, then install:
```bash
# For Databricks Runtime 14.0
pip install databricks-connect==14.0.*
```

### Step 2: Java Requirement

Spark requires Java 8 or 11. Install if not present:

```bash
# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# macOS
brew install openjdk@11

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Linux
# export JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home  # macOS
```

## Local Spark Development

### Quick Start

```bash
# Run local Spark example
python examples/spark_local_example.py
```

This script:
- Creates a local Spark session
- Loads company names from `data/sample_system_names.txt`
- Demonstrates basic Spark operations (filtering, aggregation)
- Shows TF-IDF feature extraction using Spark ML
- Performs similarity search on company names

### Spark UI

When running locally, access the Spark UI at:
```
http://localhost:4040
```

This provides:
- Job execution details
- Stage-level metrics
- Executor information
- Storage and memory usage

### Configuration

Edit `config/spark_config.py` to adjust:
- Memory allocation (executor_memory, driver_memory)
- Number of CPU cores
- Spark configurations

Example:
```python
from config.spark_config import create_spark_session, get_local_spark_config

config = get_local_spark_config(
    app_name="MyExperiment",
    executor_memory="8g",
    driver_memory="4g",
    cores=8,
)

spark = create_spark_session(use_databricks=False, local_config=config)
```

## Databricks Connect

### Configuration Options

#### Option 1: Environment Variables (Recommended)

Add to `.env` file:
```bash
DATABRICKS_WORKSPACE_URL=https://your-workspace.cloud.databricks.com
DATABRICKS_CLUSTER_ID=your-cluster-id
DATABRICKS_TOKEN=your-personal-access-token
```

#### Option 2: Databricks CLI

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure interactively
databricks configure --configure-cluster
```

#### Option 3: Direct Configuration

```python
from config.spark_config import get_databricks_connect_config, create_spark_session

config = get_databricks_connect_config(
    cluster_id="your-cluster-id",
    workspace_url="https://your-workspace.cloud.databricks.com",
    token="your-token"
)

spark = create_spark_session(use_databricks=True, databricks_config=config)
```

### Running Databricks Examples

```bash
# Run Databricks Connect example
python examples/databricks_connect_example.py
```

This script:
- Connects to remote Databricks cluster
- Runs example operations on the cluster
- Demonstrates Delta Lake table operations
- Shows time travel queries

## Common Use Cases

### 1. Distributed Company Name Matching

```python
from pyspark.sql import SparkSession
from config.spark_config import create_spark_session

spark = create_spark_session(use_databricks=False)

# Load data
companies_df = spark.read.text("data/sample_system_names.txt")

# Apply preprocessing
from src.preprocess import clean_company_name
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

clean_udf = udf(clean_company_name, StringType())
companies_df = companies_df.withColumn("cleaned", clean_udf("value"))

# Use Spark ML for TF-IDF
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

tokenizer = Tokenizer(inputCol="cleaned", outputCol="tokens")
hashing_tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="idf_features")

# Build pipeline and fit model
```

### 2. Processing Large Datasets

```python
# Read from multiple files
df = spark.read.text("data/large_corpus/*.txt")

# Repartition for parallel processing
df = df.repartition(100)

# Cache frequently used data
df.cache()

# Run transformations
result = df.filter(...).groupBy(...).agg(...)
```

### 3. Export to Delta Lake (Databricks)

```python
# Write to Delta table
df.write.format("delta") \
    .mode("overwrite") \
    .save("/delta/companies")

# Time travel: read previous version
old_version = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("/delta/companies")
```

## Performance Tips

### Local Spark
- Allocate adequate memory: `executor_memory="4g"` or more
- Use appropriate cores: `cores=4` matches most modern CPUs
- Cache DataFrames used multiple times: `df.cache()`
- Use `df.persist()` with appropriate storage level

### Databricks
- Use cluster autoscaling for variable workloads
- Enable Photon engine for faster SQL operations
- Use Delta Lake for ACID transactions and time travel
- Leverage DBFS for scalable storage

### General
- Avoid `collect()` on large datasets - returns all data to driver
- Use DataFrame operations over RDDs when possible
- Filter early to reduce data shuffling
- Use `explain()` to understand query plans:
  ```python
  df.explain(extended=True)
  ```

## Troubleshooting

### Java Not Found
```
Error: Java gateway process exited before sending its driver to Python
```
**Solution**: Install Java 8 or 11 and set JAVA_HOME

### Out of Memory
```
Java heap space
```
**Solution**: Increase memory allocation in `spark_config.py`

### Databricks Connection Failed
```
Error: Cannot connect to cluster
```
**Solutions**:
1. Verify cluster is running in Databricks workspace
2. Check token has proper permissions
3. Ensure databricks-connect version matches cluster runtime
4. Test connection: `databricks-connect test`

### Slow Performance
**Solutions**:
1. Increase partitions: `df.repartition(100)`
2. Enable adaptive query execution (default in Spark 3.x)
3. Use caching for reused DataFrames
4. Check Spark UI for bottlenecks

## Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Databricks Connect Guide](https://docs.databricks.com/dev-tools/databricks-connect.html)
- [Delta Lake Documentation](https://docs.delta.io/latest/)
- [Spark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)

## Next Steps

1. Start with `spark_local_example.py` to understand basics
2. Explore Spark MLlib for distributed machine learning
3. Experiment with Delta Lake for data management
4. Scale to Databricks when needed for production workloads
5. Integrate with existing company matching pipeline in `src/matching/`
