"""
Example: Local Spark Experimentation with Company Name Matching
This script demonstrates using PySpark locally for company name matching experiments.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, udf
from pyspark.sql.types import StringType, FloatType
import pandas as pd

from config.spark_config import create_spark_session, get_local_spark_config


def create_local_spark_session():
    """Create a local Spark session for experimentation."""
    config = get_local_spark_config(
        app_name="CompanyNameMatching",
        executor_memory="2g",
        driver_memory="2g",
        cores=4,
    )

    spark = create_spark_session(use_databricks=False, local_config=config)

    print(f"Spark version: {spark.version}")
    print(f"Spark master: {spark.conf.get('spark.master')}")
    print(f"Spark UI: {spark.sparkContext.uiWebUrl}")

    return spark


def load_company_data(spark, data_path: str = "data/sample_system_names.txt"):
    """
    Load company names from text file into Spark DataFrame.

    Args:
        spark: SparkSession
        data_path: Path to company names file

    Returns:
        DataFrame with columns: id, company_name
    """
    from src.preprocess import clean_company_name

    # Read raw company names
    with open(data_path, 'r', encoding='utf-8') as f:
        companies = [(i, line.strip()) for i, line in enumerate(f) if line.strip()]

    # Create pandas DataFrame first
    pdf = pd.DataFrame(companies, columns=['id', 'company_name'])

    # Clean company names
    pdf['cleaned_name'] = pdf['company_name'].apply(
        lambda x: clean_company_name(x, remove_stopwords=True)
    )

    # Convert to Spark DataFrame
    df = spark.createDataFrame(pdf)

    return df


def example_spark_operations(spark, df):
    """
    Demonstrate basic Spark operations on company data.

    Args:
        spark: SparkSession
        df: Company names DataFrame
    """
    print("\n=== Schema ===")
    df.printSchema()

    print("\n=== Sample Data (first 10) ===")
    df.show(10, truncate=False)

    print("\n=== Count ===")
    print(f"Total companies: {df.count()}")

    print("\n=== Companies starting with 'CÔNG TY' ===")
    df.filter(lower(col('company_name')).like('công ty%')).show(5, truncate=False)

    print("\n=== Company name length statistics ===")
    from pyspark.sql.functions import length
    df.withColumn('name_length', length(col('cleaned_name'))) \
        .select('name_length') \
        .describe() \
        .show()


def example_spark_matching(spark, df):
    """
    Example: Implement simple TF-IDF based matching using Spark ML.

    Args:
        spark: SparkSession
        df: Company names DataFrame
    """
    from pyspark.ml.feature import Tokenizer, HashingTF, IDF
    from pyspark.ml.linalg import Vectors
    from pyspark.sql.functions import rand

    print("\n=== Spark ML-based Matching Example ===")

    # Tokenize cleaned names into character n-grams
    tokenizer = Tokenizer(inputCol="cleaned_name", outputCol="char_tokens")
    tokenized_df = tokenizer.transform(df)

    # Create TF-IDF features
    hashing_tf = HashingTF(
        inputCol="char_tokens",
        outputCol="raw_features",
        numFeatures=10000
    )
    featurized_df = hashing_tf.transform(tokenized_df)

    # Apply IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    idf_model = idf.fit(featurized_df)
    rescaled_df = idf_model.transform(featurized_df)

    # Show features for first company
    print("\n=== TF-IDF Features (first company) ===")
    rescaled_df.select("company_name", "features").show(1, truncate=False)

    # Simple similarity search using cosine similarity
    query = "Vinamilk"
    print(f"\n=== Searching for: '{query}' ===")

    # Preprocess query
    from src.preprocess import clean_company_name
    cleaned_query = clean_company_name(query, remove_stopwords=True)
    print(f"Cleaned query: '{cleaned_query}'")

    # Create query DataFrame
    query_df = spark.createDataFrame([(0, cleaned_query,)], ["id", "cleaned_name"])

    # Transform query through same pipeline
    query_tokenized = tokenizer.transform(query_df)
    query_featurized = hashing_tf.transform(query_tokenized)
    query_rescaled = idf_model.transform(query_featurized)

    # Calculate cosine similarity
    query_vector = query_rescaled.first()["features"]

    def cosine_similarity(v1, v2):
        """Calculate cosine similarity between two vectors."""
        from pyspark.ml.linalg import Vectors as MLVectors
        import numpy as np

        v1_arr = v1.toArray() if hasattr(v1, 'toArray') else v1
        v2_arr = v2.toArray() if hasattr(v2, 'toArray') else v2

        dot_product = float(v1_arr.dot(v2_arr))
        norm1 = float(np.linalg.norm(v1_arr))
        norm2 = float(np.linalg.norm(v2_arr))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    # UDF for cosine similarity
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType

    cosine_udf = udf(lambda v: cosine_similarity(query_vector, v), FloatType())

    # Calculate similarities
    results_df = rescaled_df.withColumn("similarity", cosine_udf(col("features")))

    # Show top matches
    top_matches = results_df.orderBy(col("similarity").desc()).limit(5)
    top_matches.select("company_name", "similarity").show(truncate=False)

    return results_df


def main():
    """Main function to run Spark experimentation."""
    print("=== Local Spark Experimentation ===\n")

    # Create Spark session
    spark = create_local_spark_session()

    try:
        # Load data
        df = load_company_data(spark)

        # Run examples
        example_spark_operations(spark, df)
        result_df = example_spark_matching(spark, df)

        print("\n=== Experiment Complete ===")
        print(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")

    finally:
        # Clean up
        spark.stop()
        print("\nSpark session stopped.")


if __name__ == "__main__":
    main()
