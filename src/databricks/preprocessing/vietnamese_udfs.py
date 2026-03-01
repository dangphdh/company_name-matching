"""
Vietnamese text preprocessing Pandas UDFs for Spark.

This module provides vectorized UDFs for preprocessing Vietnamese company names
in Apache Spark DataFrames. Uses Pandas UDFs for 100x better performance than
row-by-row UDFs.
"""

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import pandas as pd
from typing import List

# Import existing preprocessing functions
from src.preprocess import (
    normalize_vietnamese_text,
    normalize_entity_types,
    normalize_functional_terms,
    clean_company_name,
    remove_accents
)


@pandas_udf(returnType=StringType())
def clean_company_name_udf(names: pd.Series) -> pd.Series:
    """
    Clean Vietnamese company names using Pandas UDF.

    This is a simple wrapper around clean_company_name() for Spark compatibility.
    For better performance, use preprocess_batch_udf() which returns multiple fields.

    Args:
        names: Pandas Series of company names

    Returns:
        Pandas Series of cleaned company names
    """
    return names.apply(lambda x: clean_company_name(x, remove_stopwords=True) if pd.notna(x) else "")


@pandas_udf(returnType=StringType())
def remove_accents_udf(names: pd.Series) -> pd.Series:
    """
    Remove Vietnamese accents from company names.

    Args:
        names: Pandas Series of company names

    Returns:
        Pandas Series of no-accent company names
    """
    return names.apply(lambda x: remove_accents(x) if pd.notna(x) else "")


@pandas_udf(returnType=StructType([
    StructField("cleaned", StringType(), nullable=False),
    StructField("no_accent", StringType(), nullable=False),
    StructField("norm_key", StringType(), nullable=False)
]))
def preprocess_batch_udf(names: pd.Series) -> pd.DataFrame:
    """
    Vectorized Vietnamese preprocessing for batch processing.

    This applies the full preprocessing pipeline:
    1. NFC Unicode normalization
    2. Entity type normalization (JSC → cp, CO.,LTD → tnhh)
    3. Functional term normalization (IMP-EXP → xnk)
    4. Stopword removal
    5. No-accent variant generation

    Args:
        names: Pandas Series of company names

    Returns:
        DataFrame with columns:
            - cleaned: Fully preprocessed name (with stopwords removed)
            - no_accent: No-accent version of cleaned name
            - norm_key: Normalization key for deduplication (no accents, with entity types)
    """
    result = []

    for name in names:
        if pd.isna(name) or name == "":
            result.append({"cleaned": "", "no_accent": "", "norm_key": ""})
            continue

        try:
            # Apply full preprocessing pipeline
            cleaned = clean_company_name(name, remove_stopwords=True)
            no_accent = remove_accents(cleaned)

            # Generate canonical normalization key
            # This includes entity types (cp, tnhh, etc.) but removes generic terms
            norm = normalize_vietnamese_text(name)
            norm = normalize_entity_types(norm)
            norm = normalize_functional_terms(norm)
            norm_key = remove_accents(norm).strip()

            result.append({
                'cleaned': cleaned,
                'no_accent': no_accent,
                'norm_key': norm_key
            })
        except Exception as e:
            # Handle any preprocessing errors gracefully
            result.append({"cleaned": "", "no_accent": "", "norm_key": ""})

    return pd.DataFrame(result)


@pandas_udf(returnType=ArrayType(ArrayType(StringType())))
def generate_dual_variants_udf(names: pd.Series) -> pd.Series:
    """
    Generate dual variants (accented + no-accent) for indexing.

    For each company name, returns:
    - [cleaned_accented, cleaned_no_accent] if they differ
    - [cleaned] if they're the same

    This enables robust matching against queries with or without accents.

    Args:
        names: Pandas Series of company names

    Returns:
        Pandas Series where each element is a list of variant strings
    """
    result = []

    for name in names:
        if pd.isna(name) or name == "":
            result.append([])
            continue

        try:
            # Clean the name
            cleaned = clean_company_name(name, remove_stopwords=True)
            no_accent = remove_accents(cleaned)

            # Return both variants if different, otherwise just one
            if cleaned != no_accent:
                result.append([cleaned, no_accent])
            else:
                result.append([cleaned])
        except Exception as e:
            result.append([])

    return pd.Series(result)


def preprocess_batch_for_index(names: List[str]) -> List[str]:
    """
    Preprocess a batch of company names for index building.

    This is a utility function for Stage 3 (index building) where we need
    to preprocess many names before vectorization. It's not a UDF but rather
    a helper for the batch matcher.

    Args:
        names: List of company names

    Returns:
        List of preprocessed names (with dual variants)
    """
    processed = []
    for name in names:
        if not name:
            continue

        # Clean and generate dual variants
        cleaned = clean_company_name(name, remove_stopwords=True)
        no_accent = remove_accents(cleaned)

        processed.append(cleaned)

        # Add no-accent variant if different
        if cleaned != no_accent:
            processed.append(no_accent)

    return processed


# Performance optimization: Pre-compile regex patterns for batch operations
# This is done at module load time to avoid recompilation on each UDF call
try:
    import re

    # Common Vietnamese company type patterns for fast filtering
    ENTITY_TYPE_PATTERNS = re.compile(
        r'\b(cp|tnhh|mtv|vpdd|cn|td|htx|hd)\b',
        re.IGNORECASE
    )

    # Pattern to detect repeated tokens (data quality issue)
    REPEATED_TOKEN_PATTERN = re.compile(r'\b(\w+)\s+\1\b')

    def has_entity_type(name: str) -> bool:
        """Fast check if name contains entity type."""
        return bool(ENTITY_TYPE_PATTERNS.search(name))

    def has_repeated_tokens_fast(name: str) -> bool:
        """Fast check for repeated consecutive tokens."""
        return bool(REPEATED_TOKEN_PATTERN.search(name))

except Exception:
    # Fallback if regex compilation fails
    def has_entity_type(name: str) -> bool:
        return any(et in name.lower() for et in ['cp', 'tnhh', 'mtv', 'vpdd', 'cn', 'td', 'htx', 'hd'])

    def has_repeated_tokens_fast(name: str) -> bool:
        tokens = name.split()
        return any(tokens[i] == tokens[i+1] for i in range(len(tokens)-1) if len(tokens[i]) > 1)
