"""
Stage 4: Batch Match

This stage performs distributed batch matching of queries against the company index
using Pandas UDFs for vectorized cosine similarity computation.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, pandas_udf, lit, explode, broadcast, row_number, rank
)
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.sql.window import Window
import pandas as pd
import numpy as np

from src.databricks.utils.delta_utils import write_delta
from src.databricks.utils.metrics import log_stage_metrics
from src.databricks.utils.validation import validate_matches


@log_stage_metrics("Stage 4: Batch Match")
def run_stage4_match(
    spark: SparkSession,
    queries_df: DataFrame,
    matcher,
    index_path: str,
    output_path: str,
    top_k: int = 5,
    min_score: float = 0.0,
    broadcast_index: bool = False
) -> DataFrame:
    """
    Stage 4: Batch match queries against company index.

    This stage:
    1. Preprocesses queries using Vietnamese preprocessing
    2. Collects queries to driver (for matching)
    3. Matches each query against the index using matcher.search()
    4. Formats results with confidence scores
    5. Validates results and writes to Gold layer

    Args:
        spark: SparkSession
        queries_df: DataFrame with queries (must have 'query_id' and 'query_text' columns)
        matcher: CompanyMatcher instance (from Stage 3)
        index_path: Path to indexed data (for metadata lookup)
        output_path: Path to output Delta table (matching results)
        top_k: Number of results per query
        min_score: Minimum confidence threshold
        broadcast_index: Whether to broadcast index (future enhancement)

    Returns:
        DataFrame with matching results
    """
    print(f"\n[Stage 4] Configuration:")
    print(f"  Queries:          {queries_df.count():,}")
    print(f"  Index path:       {index_path}")
    print(f"  Output path:      {output_path}")
    print(f"  Top K:            {top_k}")
    print(f"  Min score:        {min_score}")

    # Step 1: Validate queries schema
    print(f"\n[Stage 4] Step 1: Validating queries schema...")

    required_columns = ["query_id", "query_text"]
    for col_name in required_columns:
        if col_name not in queries_df.columns:
            raise ValueError(f"Missing required column: {col_name}")

    print(f"[Stage 4] Schema validated")

    # Step 2: Preprocess queries
    print(f"\n[Stage 4] Step 2: Preprocessing queries...")

    from src.databricks.preprocessing.vietnamese_udfs import clean_company_name_udf

    queries_processed = queries_df.select(
        "query_id",
        "query_text",
        clean_company_name_udf(col("query_text")).alias("query_cleaned")
    )

    print(f"[Stage 4] Preprocessed {queries_processed.count():,} queries")

    # Step 3: Collect queries to driver
    print(f"\n[Stage 4] Step 3: Collecting queries to driver...")

    queries = queries_processed.select("query_id", "query_cleaned").collect()

    query_ids = [row['query_id'] for row in queries]
    query_texts = [row['query_cleaned'] for row in queries]

    print(f"[Stage 4] Collected {len(queries)} queries")

    # Step 4: Perform matching on driver
    print(f"\n[Stage 4] Step 4: Matching queries against index...")

    results = []

    for i, (query_id, query_text) in enumerate(zip(query_ids, query_texts)):
        if i % 100 == 0:
            print(f"[Stage 4] Processed {i}/{len(queries)} queries...")

        # Search using matcher
        matches = matcher.search(query_text, top_k=top_k, min_score=min_score)

        # Format results
        for rank, match in enumerate(matches, start=1):
            # Look up canonical ID if available
            canonical_id = None
            if hasattr(matcher, 'canonical_ids'):
                # Find the company in the original list
                try:
                    idx = matcher.corpus_names.index(match['company'])
                    canonical_id = matcher.canonical_ids[idx]
                except ValueError:
                    canonical_id = None

            # Determine confidence level
            score = match['score']
            if score >= 0.90:
                confidence = "high"
            elif score >= 0.75:
                confidence = "medium"
            else:
                confidence = "low"

            results.append({
                'query_id': query_id,
                'query_text': query_text,
                'matched_company': match['company'],
                'canonical_id': str(canonical_id) if canonical_id else None,
                'score': float(score),
                'rank': rank,
                'match_confidence': confidence
            })

    print(f"[Stage 4] Generated {len(results)} match results")

    # Step 5: Convert results to DataFrame
    print(f"\n[Stage 4] Step 5: Converting results to DataFrame...")

    results_df = spark.createDataFrame(results)

    # Step 6: Validate results
    print(f"\n[Stage 4] Step 6: Validating results...")
    results_df, validation_stats = validate_matches(results_df)

    # Step 7: Write to Gold layer
    print(f"\n[Stage 4] Step 7: Writing to Gold layer...")
    write_delta(results_df, output_path, mode="overwrite")

    # Step 8: Print summary
    print(f"\n[Stage 4] Summary:")
    print(f"  Queries processed:     {len(queries):,}")
    print(f"  Total matches:        {len(results):,}")
    print(f"  Valid results:        {validation_stats['valid_count']:,}")

    # Compute quality metrics
    from src.databricks.utils.metrics import compute_quality_metrics, print_quality_metrics

    quality_metrics = compute_quality_metrics(results_df)
    print_quality_metrics(quality_metrics)

    # Sample results
    print(f"\n[Stage 4] Sample results:")
    results_df.select(
        "query_id",
        "query_text",
        "matched_company",
        "score",
        "rank",
        "match_confidence"
    ).show(10, truncate=False)

    return results_df


def create_matcher_udf(matcher, top_k: int = 5, min_score: float = 0.0):
    """
    Create a Pandas UDF for distributed matching.

    This is an alternative implementation for future enhancement,
    allowing true distributed processing across workers.

    Args:
        matcher: CompanyMatcher instance (must be serializable)
        top_k: Number of results per query
        min_score: Minimum confidence threshold

    Returns:
        Pandas UDF for matching
    """
    # Note: This is not currently used due to serialization challenges
    # with sklearn models. Kept for future reference.

    @pandas_udf(
        returnType=StructType([
            StructField("matched_company", StringType(), nullable=False),
            StructField("score", FloatType(), nullable=False),
            StructField("rank", IntegerType(), nullable=False)
        ])
    )
    def match_udf(query_text: pd.Series) -> pd.DataFrame:
        """
        Match queries using CompanyMatcher.

        Args:
            query_text: Pandas Series of query texts

        Returns:
            DataFrame with match results
        """
        results = []

        for query in query_text:
            matches = matcher.search(query, top_k=top_k, min_score=min_score)

            for rank, match in enumerate(matches, start=1):
                results.append({
                    'matched_company': match['company'],
                    'score': float(match['score']),
                    'rank': rank
                })

        # If no results, return empty DataFrame with correct schema
        if not results:
            return pd.DataFrame({
                'matched_company': pd.Series(dtype=str),
                'score': pd.Series(dtype=float),
                'rank': pd.Series(dtype=int)
            })

        return pd.DataFrame(results)

    return match_udf
