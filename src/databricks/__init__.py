"""
Databricks Batch Processing Pipeline for Vietnamese Company Name Matching

This package provides a production-grade pipeline for matching Vietnamese company names
at scale using Apache Spark on Databricks.

Architecture: Hybrid Pragmatic Approach
- Spark for ETL (preprocessing, deduplication, shuffling)
- Enhanced CompanyMatcher with LSA for 2M-scale indexing
- Pandas UDFs for vectorized batch operations
- Delta Lake for reliable storage

Usage:
    from src.databricks.orchestrator import PipelineOrchestrator
    from src.databricks.config import load_config

    config = load_config()
    pipeline = PipelineOrchestrator(config)
    pipeline.run_pipeline(...)
"""

__version__ = "1.0.0"
__author__ = "Company Matching Team"
