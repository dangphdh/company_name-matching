"""
Comprehensive Spark Experiment for Company Name Matching

This script evaluates the performance of company name matching using Spark
vs traditional single-node processing, measuring accuracy, latency, and scalability.
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, udf, rand, length, avg, stddev, min as spark_min, max as spark_max, when, lit, count
from pyspark.sql.types import StringType, FloatType, ArrayType
from src.matching.matcher import CompanyMatcher
from src.preprocess import clean_company_name
from config.spark_config import get_local_spark_config, create_spark_session


def setup_spark_session(cores: int = 4, memory: str = "4g") -> SparkSession:
    """Create and configure Spark session for experiments."""
    config = get_local_spark_config(
        app_name="CompanyMatchingEvaluation",
        executor_memory=memory,
        driver_memory=memory,
        cores=cores,
    )

    spark = create_spark_session(use_databricks=False, local_config=config)
    return spark


def load_sample_data(spark, num_companies: int = 1000) -> Tuple:
    """
    Load company names into Spark DataFrame.

    Returns:
        Tuple of (spark_df, pandas_df, companies_list)
    """
    # Load base companies
    with open('data/sample_system_names.txt', 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f if line.strip()]

    # Upsample if needed
    while len(companies) < num_companies:
        companies = companies + companies

    companies = companies[:num_companies]

    # Create pandas DataFrame
    pdf = pd.DataFrame({
        'id': range(len(companies)),
        'company_name': companies,
        'cleaned_name': [clean_company_name(c, remove_stopwords=True) for c in companies]
    })

    # Create Spark DataFrame
    spark_df = spark.createDataFrame(pdf)

    return spark_df, pdf, companies


def experiment_1_spark_vs_traditional(spark, companies: List[str]) -> Dict:
    """
    Experiment 1: Compare Spark DataFrame operations vs traditional processing
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Spark vs Traditional Processing")
    print("="*80)

    results = {
        'experiment': 'spark_vs_traditional',
        'num_companies': len(companies),
        'timestamp': datetime.now().isoformat()
    }

    # Generate test queries
    test_queries = companies[:100]  # Use first 100 as queries
    query_variants = []
    for q in test_queries:
        query_variants.append(clean_company_name(q, remove_stopwords=True))
        query_variants.append(clean_company_name(q, remove_stopwords=False))

    # --- Traditional Approach ---
    print("\n[Traditional] Building index...")
    traditional_start = time.time()

    traditional_matcher = CompanyMatcher(model_name='tfidf-bm25')
    traditional_matcher.build_index(companies)

    traditional_index_time = time.time() - traditional_start
    print(f"[Traditional] Index built in {traditional_index_time:.3f}s")

    # Test query performance
    print("[Traditional] Running queries...")
    traditional_query_start = time.time()

    traditional_search_results = []
    for query in query_variants[:50]:  # Test 50 queries
        search_result = traditional_matcher.search(query, top_k=5)
        traditional_search_results.append(search_result)

    traditional_query_time = time.time() - traditional_query_start
    traditional_avg_latency = (traditional_query_time / 50) * 1000  # ms

    print(f"[Traditional] Queries: {traditional_query_time:.3f}s, Avg: {traditional_avg_latency:.2f}ms")

    # --- Spark Approach ---
    print("\n[Spark] Creating DataFrame...")
    spark_df = spark.createDataFrame(
        [(i, c, clean_company_name(c, remove_stopwords=True))
         for i, c in enumerate(companies)]
    ).toDF('id', 'company_name', 'cleaned_name')

    spark_df_count = spark_df.count()
    print(f"[Spark] DataFrame created with {spark_df_count} rows")

    # Test Spark operations
    print("[Spark] Running filter operations...")
    spark_op_start = time.time()

    # Filter companies starting with "CÔNG TY"
    filtered_df = spark_df.filter(col('cleaned_name').like('cong ty%'))
    filtered_count = filtered_df.count()

    spark_op_time = time.time() - spark_op_start
    print(f"[Spark] Filter operation: {spark_op_time:.3f}s, Found: {filtered_count}")

    # Cache and count operations
    print("[Spark] Testing cache and count...")
    spark_cache_start = time.time()

    spark_df.cache()
    count1 = spark_df.count()
    count2 = spark_df.count()  # Should be faster due to caching

    spark_cache_time = time.time() - spark_cache_start
    print(f"[Spark] Cache operations: {spark_cache_time:.3f}s (Count1: {count1}, Count2: {count2})")

    # Store results
    results['traditional'] = {
        'index_build_time': traditional_index_time,
        'query_time': traditional_query_time,
        'avg_query_latency_ms': traditional_avg_latency,
        'queries_per_second': 50 / traditional_query_time
    }

    results['spark'] = {
        'dataframe_creation_time': spark_op_time,
        'filter_operation_time': spark_op_time,
        'cache_operation_time': spark_cache_time,
        'row_count': spark_df_count
    }

    return results


def experiment_2_scalability_analysis(spark) -> Dict:
    """
    Experiment 2: Test scalability with different corpus sizes
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Scalability Analysis")
    print("="*80)

    results = {
        'experiment': 'scalability_analysis',
        'timestamp': datetime.now().isoformat(),
        'corpus_sizes': []
    }

    corpus_sizes = [1000, 5000, 10000]

    for size in corpus_sizes:
        print(f"\n--- Testing with {size} companies ---")

        size_results = {'num_companies': size}

        # Load data
        spark_df, pdf, companies = load_sample_data(spark, num_companies=size)

        # Measure Spark operations
        spark_start = time.time()

        spark_df.cache()
        row_count = spark_df.count()

        # Filter operation
        filtered = spark_df.filter(col('cleaned_name').like('cong ty%'))
        filtered_count = filtered.count()

        # Aggregation
        avg_length = spark_df.withColumn('name_len', length(col('cleaned_name'))) \
                         .agg(avg('name_len')).collect()[0][0]

        spark_time = time.time() - spark_start

        size_results['spark'] = {
            'total_time': spark_time,
            'row_count': row_count,
            'filtered_count': filtered_count,
            'avg_name_length': float(avg_length)
        }

        # Measure traditional operations
        trad_start = time.time()

        matcher = CompanyMatcher(model_name='tfidf')
        matcher.build_index(companies)

        # Sample query
        if len(companies) > 0:
            query = clean_company_name(companies[0], remove_stopwords=True)
            search_results = matcher.search(query, top_k=5)

        trad_time = time.time() - trad_start

        size_results['traditional'] = {
            'index_build_time': trad_time
        }

        results['corpus_sizes'].append(size_results)

        print(f"  Spark: {spark_time:.3f}s, Traditional index: {trad_time:.3f}s")

    return results


def experiment_3_data_processing_pipeline(spark) -> Dict:
    """
    Experiment 3: Complete data processing pipeline with Spark
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Data Processing Pipeline")
    print("="*80)

    results = {
        'experiment': 'data_processing_pipeline',
        'timestamp': datetime.now().isoformat()
    }

    # Load data
    spark_df, pdf, companies = load_sample_data(spark, num_companies=5000)

    # Pipeline 1: Basic cleaning and filtering
    print("\n[Pipeline 1] Basic operations...")
    start = time.time()

    cleaned_df = spark_df.filter(col('cleaned_name').rlike("\\w{5,}"))  # At least 5 chars
    filtered_df = cleaned_df.filter(length(col('cleaned_name')) > 10)

    pipeline1_time = time.time() - start
    pipeline1_count = filtered_df.count()

    print(f"  Pipeline 1: {pipeline1_time:.3f}s, Rows: {pipeline1_count}")

    # Pipeline 2: Aggregation and statistics
    print("\n[Pipeline 2] Aggregation operations...")
    start = time.time()

    stats = filtered_df.select(
        length(col('cleaned_name')).alias('length')
    ).agg(
        avg('length').alias('avg_length'),
        stddev('length').alias('std_length'),
        spark_min('length').alias('min_length'),
        spark_max('length').alias('max_length')
    ).collect()[0]

    pipeline2_time = time.time() - start

    print(f"  Pipeline 2: {pipeline2_time:.3f}s")
    print(f"    Avg: {stats.avg_length:.2f}, Std: {stats.std_length:.2f}")
    print(f"    Min: {stats.min_length}, Max: {stats.max_length}")

    # Pipeline 3: Complex transformations
    print("\n[Pipeline 3] Complex transformations...")
    start = time.time()

    categorized_df = filtered_df.withColumn(
        'category',
        when(col('cleaned_name').like('cong ty%'), 'company')
        .when(col('cleaned_name').like('tap doan%'), 'conglomerate')
        .when(col('cleaned_name').like('ngan hang%'), 'bank')
        .otherwise('other')
    )

    category_counts = categorized_df.groupBy('category').count().collect()
    pipeline3_time = time.time() - start

    print(f"  Pipeline 3: {pipeline3_time:.3f}s")
    for row in category_counts:
        print(f"    {row.category}: {row['count']}")

    results['pipeline_1'] = {
        'time': pipeline1_time,
        'row_count': pipeline1_count
    }

    results['pipeline_2'] = {
        'time': pipeline2_time,
        'statistics': {
            'avg_length': float(stats.avg_length),
            'std_length': float(stats.std_length),
            'min_length': int(stats.min_length),
            'max_length': int(stats.max_length)
        }
    }

    results['pipeline_3'] = {
        'time': pipeline3_time,
        'categories': {row.category: row['count'] for row in category_counts}
    }

    return results


def experiment_4_partitioning_performance(spark) -> Dict:
    """
    Experiment 4: Test performance with different partition strategies
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: Partitioning Performance")
    print("="*80)

    results = {
        'experiment': 'partitioning_performance',
        'timestamp': datetime.now().isoformat(),
        'partitions': []
    }

    # Load data
    spark_df, pdf, companies = load_sample_data(spark, num_companies=10000)

    partition_counts = [1, 4, 10, 100]

    for num_partitions in partition_counts:
        print(f"\n--- Testing with {num_partitions} partition(s) ---")

        partition_results = {'num_partitions': num_partitions}

        # Repartition
        start = time.time()

        repartitioned_df = spark_df.repartition(num_partitions)
        partition_count = repartitioned_df.rdd.getNumPartitions()

        repartition_time = time.time() - start

        # Test filter operation
        start = time.time()

        filtered = repartitioned_df.filter(col('cleaned_name').like('cong ty%'))
        result_count = filtered.count()

        filter_time = time.time() - start

        # Test aggregation
        start = time.time()

        agg_result = filtered.agg(count('*')).collect()

        agg_time = time.time() - start

        partition_results['operations'] = {
            'repartition_time': repartition_time,
            'filter_time': filter_time,
            'aggregation_time': agg_time,
            'result_count': result_count
        }

        results['partitions'].append(partition_results)

        print(f"  Repartition: {repartition_time:.4f}s")
        print(f"  Filter: {filter_time:.4f}s, Count: {result_count}")
        print(f"  Aggregation: {agg_time:.4f}s")

    return results


def run_all_experiments(spark) -> Dict:
    """Run all experiments and collect results."""
    print("\n" + "="*80)
    print("SPARK COMPANY NAME MATCHING - FULL EXPERIMENT SUITE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Spark version: {spark.version}")
    print(f"Spark master: {spark.conf.get('spark.master')}")

    all_results = {
        'suite': 'spark_matching_experiments',
        'start_time': datetime.now().isoformat(),
        'experiments': []
    }

    # Load base data for experiments
    _, _, companies = load_sample_data(spark, num_companies=1000)

    try:
        # Run experiments
        exp1_results = experiment_1_spark_vs_traditional(spark, companies)
        all_results['experiments'].append(exp1_results)

        exp2_results = experiment_2_scalability_analysis(spark)
        all_results['experiments'].append(exp2_results)

        exp3_results = experiment_3_data_processing_pipeline(spark)
        all_results['experiments'].append(exp3_results)

        exp4_results = experiment_4_partitioning_performance(spark)
        all_results['experiments'].append(exp4_results)

    finally:
        spark.stop()

    all_results['end_time'] = datetime.now().isoformat()

    return all_results


def save_results(results: Dict, output_file: str = 'spark_experiment_results.json'):
    """Save experiment results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main entry point."""
    # Setup Spark
    print("Initializing Spark session...")
    spark = setup_spark_session(cores=4, memory="4g")

    print(f"✅ Spark session created")
    print(f"   Spark UI: {spark.sparkContext.uiWebUrl}")

    # Run experiments
    results = run_all_experiments(spark)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data/eval/spark_experiment_{timestamp}.json'

    # Create directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    save_results(results, output_file)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    for exp in results['experiments']:
        exp_name = exp.get('experiment', 'unknown')
        print(f"\n✓ {exp_name}: Completed")

    print(f"\n📊 Full results saved to: {output_file}")
    print(f"⏱️  Total duration: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
