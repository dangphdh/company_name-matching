"""
Stage 3: Build Index

This stage builds the CompanyMatcher index using LSA for efficient processing
at 2M scale. It uses the enhanced save/load functionality for index persistence.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, collect_set
import numpy as np

from src.matching.matcher import CompanyMatcher
from src.databricks.preprocessing.vietnamese_udfs import preprocess_batch_for_index
from src.databricks.utils.delta_utils import write_delta
from src.databricks.utils.metrics import log_stage_metrics


@log_stage_metrics("Stage 3: Build Index")
def run_stage3_build_index(
    spark: SparkSession,
    input_path: str,
    index_path: str,
    model_path: str,
    model_name: str = "tfidf-lsa",
    lsa_dims: int = 512,
    remove_stopwords: bool = True,
    force_rebuild: bool = False
) -> CompanyMatcher:
    """
    Stage 3: Build CompanyMatcher index from deduplicated companies.

    This stage:
    1. Loads deduplicated companies from Silver layer
    2. Collects company names to driver (for index building)
    3. Builds TF-IDF + LSA index using CompanyMatcher
    4. Saves index to disk for persistence
    5. Returns matcher instance for Stage 4

    Args:
        spark: SparkSession
        input_path: Path to input Delta table (from Stage 2)
        index_path: Path to save indexed data (metadata + vectors)
        model_path: Path to save/load matcher model
        model_name: Model type ("tfidf-lsa" for 2M scale)
        lsa_dims: LSA dimensions (512 recommended for 2M)
        remove_stopwords: Whether to remove stopwords
        force_rebuild: Force rebuild even if cached index exists

    Returns:
        CompanyMatcher instance with built index
    """
    print(f"\n[Stage 3] Configuration:")
    print(f"  Input path:      {input_path}")
    print(f"  Index path:      {index_path}")
    print(f"  Model path:      {model_path}")
    print(f"  Model name:      {model_name}")
    print(f"  LSA dimensions:  {lsa_dims}")
    print(f"  Force rebuild:   {force_rebuild}")

    # Step 1: Check for existing index
    from pathlib import Path

    if not force_rebuild and Path(model_path).exists():
        print(f"\n[Stage 3] Found existing index at: {model_path}")
        print(f"[Stage 3] Loading index...")

        try:
            matcher = CompanyMatcher.load_index(model_path)
            print(f"[Stage 3] Index loaded successfully!")
            return matcher
        except Exception as e:
            print(f"[Stage 3] Failed to load index: {e}")
            print(f"[Stage 3] Will rebuild index...")
            force_rebuild = True

    # Step 2: Load deduplicated companies
    print(f"\n[Stage 3] Step 1: Loading deduplicated companies...")
    df_companies = spark.read.format("delta").load(input_path)

    company_count = df_companies.count()
    print(f"[Stage 3] Loaded {company_count:,} companies")

    # Step 3: Collect company names to driver
    print(f"\n[Stage 3] Step 2: Collecting company names to driver...")

    # Select unique company names (use original name for building index)
    companies_data = df_companies.select("company_name", "canonical_id").distinct()

    # Collect to driver
    company_names = companies_data.select("company_name").rdd.map(lambda r: r[0]).collect()
    canonical_ids = companies_data.select("canonical_id").rdd.map(lambda r: r[0]).collect()

    print(f"[Stage 3] Collected {len(company_names)} unique company names")

    # Estimate memory usage
    avg_name_length = sum(len(name) for name in company_names) / len(company_names)
    estimated_memory = (
        len(company_names) * avg_name_length +  # Names
        len(company_names) * 100 +  # Python overhead
        len(company_names) * lsa_dims * 4  # LSA vectors (float32)
    )

    print(f"[Stage 3] Estimated memory usage: ~{estimated_memory / 1e9:.2f} GB")

    if estimated_memory > 16e9:  # More than 16 GB
        print(f"[⚠️]  Warning: Large index size. Consider reducing LSA dimensions or using more memory.")

    # Step 4: Build matcher index
    print(f"\n[Stage 3] Step 3: Building {model_name} index...")

    matcher = CompanyMatcher(
        model_name=model_name,
        lsa_dims=lsa_dims,
        remove_stopwords=remove_stopwords
    )

    # Build index
    print(f"[Stage 3] Building TF-IDF + LSA index for {len(company_names)} companies...")
    matcher.build_index(company_names)

    print(f"[Stage 3] Index built successfully!")

    # Store canonical IDs for result mapping
    matcher.canonical_ids = canonical_ids

    # Step 5: Save index to disk
    print(f"\n[Stage 3] Step 4: Saving index to disk...")
    matcher.save_index(model_path)

    print(f"[Stage 3] Index saved to: {model_path}")

    # Step 6: Save indexed data to Delta Lake (for reference)
    print(f"\n[Stage 3] Step 5: Writing indexed metadata to Delta Lake...")

    # Create indexed metadata DataFrame
    # We'll save the canonical company info for reference
    indexed_metadata = df_companies.select(
        "canonical_id",
        "company_name",
        "cleaned_name",
        "norm_key"
    )

    write_delta(indexed_metadata, index_path, mode="overwrite")

    # Step 7: Print summary
    print(f"\n[Stage 3] Summary:")
    print(f"  Companies indexed:  {len(company_names):,}")
    print(f"  LSA dimensions:     {lsa_dims}")
    print(f"  Model saved:        {model_path}")
    print(f"  Metadata saved:     {index_path}")

    # Sample index info
    if hasattr(matcher, 'corpus_vectors'):
        print(f"\n[Stage 3] Index Statistics:")
        print(f"  Vector shape:      {matcher.corpus_vectors.shape}")
        print(f"  Memory usage:      {matcher.corpus_vectors.nbytes / 1e9:.2f} GB")

    return matcher
