"""
Pipeline Orchestrator

This module provides the main orchestrator for running the company name
matching pipeline end-to-end.
"""

import time
from pyspark.sql import SparkSession, DataFrame
from typing import Optional, Dict, Any

from src.databricks.config import load_config, PipelineConfig
from src.databricks.stages.stage1_extract import run_stage1_extract
from src.databricks.stages.stage2_deduplicate import run_stage2_deduplicate
from src.databricks.stages.stage3_build_index import run_stage3_build_index
from src.databricks.stages.stage4_match import run_stage4_match
from src.databricks.utils.metrics import compute_quality_metrics, print_quality_metrics, check_quality_alerts
from config.spark_config import create_spark_session


class PipelineOrchestrator:
    """
    Orchestrates the 4-stage company name matching pipeline.

    Usage:
        config = load_config()
        pipeline = PipelineOrchestrator(config)
        pipeline.run_full_pipeline(queries_df)
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.spark = None
        self.stage_outputs = {}
        self.matcher = None

    def initialize_spark(self):
        """Initialize or get Spark session."""
        if self.spark is None:
            print("\n[Pipeline] Initializing Spark session...")

            # Determine if running on Databricks
            use_databricks = (
                self.config.databricks_cluster_id is not None or
                self.config.databricks_workspace_url is not None
            )

            # Build Spark config
            databricks_config = None
            local_config = None

            if use_databricks:
                databricks_config = {
                    'cluster_id': self.config.databricks_cluster_id,
                    'workspace_url': self.config.databricks_workspace_url,
                    'token': self.config.databricks_token
                }
            else:
                local_config = {
                    'app_name': "CompanyMatching",
                    'executor_memory': self.config.local_executor_memory,
                    'driver_memory': self.config.local_driver_memory,
                    'cores': self.config.local_cores
                }

            self.spark = create_spark_session(
                use_databricks=use_databricks,
                databricks_config=databricks_config,
                local_config=local_config
            )

            # Configure Spark for optimal performance
            self.spark.conf.set("spark.sql.adaptive.enabled", str(self.config.adaptive_query))
            self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", str(self.config.adaptive_query))
            self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", str(self.config.arrow_enabled))
            self.spark.conf.set("spark.sql.shuffle.partitions", str(self.config.shuffle_partitions))

            print(f"[Pipeline] Spark {self.spark.version} initialized")
            print(f"[Pipeline] App name: {self.spark.conf.get('spark.app.name')}")

        return self.spark

    def run_full_pipeline(
        self,
        bronze_path: Optional[str] = None,
        queries_df: Optional[DataFrame] = None,
        skip_stages: list = None
    ) -> Dict[str, Any]:
        """
        Run all 4 stages of the pipeline.

        Args:
            bronze_path: Override bronze path (default: from config)
            queries_df: DataFrame with queries to match (columns: query_id, query_text)
            skip_stages: List of stage numbers to skip (e.g., [1, 2] to rebuild from Stage 3)

        Returns:
            Dictionary with stage outputs and metrics
        """
        # Initialize Spark
        self.initialize_spark()

        # Use paths from config if not provided
        if bronze_path is None:
            bronze_path = self.config.bronze_path

        # Define pipeline paths
        silver_path = self.config.silver_path
        gold_index_path = self.config.gold_index_path
        gold_matches_path = self.config.gold_matches_path
        model_path = silver_path + "_model"

        # Track timing
        pipeline_start = time.time()

        print(f"\n{'='*80}")
        print("COMPANY NAME MATCHING PIPELINE")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Model:           {self.config.model_name}")
        print(f"  LSA dimensions:  {self.config.lsa_dims}")
        print(f"  Bronze path:     {bronze_path}")
        print(f"  Silver path:     {silver_path}")
        print(f"  Gold index:      {gold_index_path}")
        print(f"  Gold matches:    {gold_matches_path}")
        print(f"{'='*80}")

        try:
            # Stage 1: Extract & Preprocess
            if skip_stages is None or 1 not in skip_stages:
                print(f"\n{'='*80}")
                print("STAGE 1/4: EXTRACT & PREPROCESS")
                print(f"{'='*80}")

                df_cleaned = run_stage1_extract(
                    spark=self.spark,
                    bronze_path=bronze_path,
                    silver_path=silver_path,
                    num_partitions=self.config.stage1_partitions,
                    validate=True
                )

                self.stage_outputs['stage1'] = {
                    'status': 'success',
                    'record_count': df_cleaned.count()
                }
            else:
                print(f"\n[⏭]  Skipping Stage 1")

            # Stage 2: Deduplicate
            if skip_stages is None or 2 not in skip_stages:
                print(f"\n{'='*80}")
                print("STAGE 2/4: DEDUPLICATE")
                print(f"{'='*80}")

                df_dedup = run_stage2_deduplicate(
                    spark=self.spark,
                    input_path=silver_path,
                    output_path=silver_path + "_dedup",
                    dedup_strategy=self.config.stage2_dedup_strategy
                )

                self.stage_outputs['stage2'] = {
                    'status': 'success',
                    'record_count': df_dedup.count()
                }
            else:
                print(f"\n[⏭]  Skipping Stage 2")

            # Stage 3: Build Index
            if skip_stages is None or 3 not in skip_stages:
                print(f"\n{'='*80}")
                print("STAGE 3/4: BUILD INDEX")
                print(f"{'='*80}")

                input_for_stage3 = silver_path + "_dedup" if (skip_stages is None or 2 not in skip_stages) else silver_path

                self.matcher = run_stage3_build_index(
                    spark=self.spark,
                    input_path=input_for_stage3,
                    index_path=gold_index_path,
                    model_path=model_path,
                    model_name=self.config.model_name,
                    lsa_dims=self.config.lsa_dims,
                    remove_stopwords=self.config.remove_stopwords
                )

                self.stage_outputs['stage3'] = {
                    'status': 'success'
                }
            else:
                print(f"\n[⏭]  Skipping Stage 3")
                # Load existing matcher
                if self.matcher is None:
                    print(f"[Pipeline] Loading existing matcher...")
                    self.matcher = CompanyMatcher.load_index(model_path)

            # Stage 4: Batch Match (if queries provided)
            if queries_df is not None:
                print(f"\n{'='*80}")
                print("STAGE 4/4: BATCH MATCH")
                print(f"{'='*80}")

                df_matches = run_stage4_match(
                    spark=self.spark,
                    queries_df=queries_df,
                    matcher=self.matcher,
                    index_path=gold_index_path,
                    output_path=gold_matches_path,
                    top_k=self.config.top_k,
                    min_score=self.config.min_score
                )

                self.stage_outputs['stage4'] = {
                    'status': 'success',
                    'match_count': df_matches.count()
                }

                # Compute quality metrics
                quality_metrics = compute_quality_metrics(df_matches)
                print_quality_metrics(quality_metrics)

                # Check quality alerts
                alerts = check_quality_alerts(quality_metrics, {
                    'avg_score': self.config.alert_threshold_avg_score,
                    'high_confidence_rate': self.config.alert_threshold_high_confidence
                })

                if alerts:
                    print(f"\n[⚠️]  QUALITY ALERTS:")
                    for alert in alerts:
                        print(f"  {alert}")

                self.stage_outputs['quality_metrics'] = quality_metrics
            else:
                print(f"\n[⏭]  Skipping Stage 4 (no queries provided)")

            # Pipeline completed
            elapsed = time.time() - pipeline_start

            print(f"\n{'='*80}")
            print(f"PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")

            self.stage_outputs['pipeline'] = {
                'status': 'success',
                'elapsed_seconds': elapsed
            }

            return self.stage_outputs

        except Exception as e:
            elapsed = time.time() - pipeline_start

            print(f"\n{'='*80}")
            print(f"PIPELINE FAILED")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            print(f"Elapsed: {elapsed:.2f}s")

            self.stage_outputs['pipeline'] = {
                'status': 'error',
                'error': str(e),
                'elapsed_seconds': elapsed
            }

            raise

    def run_matching_only(self, queries_df: DataFrame) -> DataFrame:
        """
        Run only Stage 4 (matching) using existing index.

        This is useful for running new queries against a pre-built index.

        Args:
            queries_df: DataFrame with queries (columns: query_id, query_text)

        Returns:
            DataFrame with matching results
        """
        # Initialize Spark
        self.initialize_spark()

        # Load existing matcher
        model_path = self.config.silver_path + "_model"

        if self.matcher is None:
            print(f"[Pipeline] Loading existing matcher from: {model_path}")
            self.matcher = CompanyMatcher.load_index(model_path)

        # Run Stage 4
        return run_stage4_match(
            spark=self.spark,
            queries_df=queries_df,
            matcher=self.matcher,
            index_path=self.config.gold_index_path,
            output_path=self.config.gold_matches_path,
            top_k=self.config.top_k,
            min_score=self.config.min_score
        )

    def cleanup(self):
        """Cleanup resources (un cache DataFrames, stop Spark)."""
        if self.spark is not None:
            print(f"[Pipeline] Cleaning up...")
            self.spark.stop()
            self.spark = None
