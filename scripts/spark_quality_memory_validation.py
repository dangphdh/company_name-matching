"""
Spark Quality and Memory Validation for Company Name Matching

This script validates:
1. Quality: Accuracy, precision, recall when using Spark
2. Memory: Memory consumption and optimization
3. Comparison: Spark vs Traditional approaches
"""
import sys
import time
import json
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import psutil
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, udf, length
from pyspark.sql.types import StringType, FloatType
from src.matching.matcher import CompanyMatcher
from src.preprocess import clean_company_name
from config.spark_config import get_local_spark_config, create_spark_session


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),    # Percentage of total RAM
    }


def measure_spark_memory(spark: SparkSession) -> Dict[str, float]:
    """Measure Spark's memory consumption."""
    try:
        sc = spark.sparkContext
        status = sc.statusTracker()
        return {
            'executor_memory_mb': spark.conf.get('spark.executor.memory', 'Not set'),
            'driver_memory_mb': spark.conf.get('spark.driver.memory', 'Not set'),
        }
    except:
        return {'error': 'Could not retrieve Spark memory info'}


def quality_validation_1_accuracy_comparison(spark, companies: List[str]) -> Dict:
    """
    Quality Validation 1: Compare matching accuracy between Spark and Traditional
    """
    print("\n" + "="*80)
    print("QUALITY VALIDATION 1: Accuracy Comparison")
    print("="*80)

    results = {
        'validation': 'accuracy_comparison',
        'timestamp': datetime.now().isoformat(),
        'num_companies': len(companies)
    }

    # Generate test queries with known matches
    print("\n[Setup] Generating test queries...")
    test_samples = companies[:100]  # Use first 100 companies
    queries = []
    expected_matches = []

    for company in test_samples:
        # Create query variants
        cleaned = clean_company_name(company, remove_stopwords=True)
        queries.append(cleaned)
        expected_matches.append(company)

    print(f"[Setup] Generated {len(queries)} test queries")

    # --- Traditional Approach ---
    print("\n[Traditional] Building matcher...")
    mem_before = get_memory_usage()

    traditional_matcher = CompanyMatcher(model_name='tfidf-bm25')
    traditional_matcher.build_index(companies)

    mem_after = get_memory_usage()

    print(f"[Traditional] Index built")
    print(f"[Traditional] Memory: RSS {mem_after['rss_mb']:.2f} MB (+{mem_after['rss_mb'] - mem_before['rss_mb']:.2f} MB)")

    # Test queries
    print(f"[Traditional] Running {len(queries)} queries...")
    trad_start = time.time()

    traditional_correct = 0
    traditional_top3_correct = 0
    traditional_scores = []

    for i, query in enumerate(queries):
        results_list = traditional_matcher.search(query, top_k=3)

        if results_list:
            top_result = results_list[0]
            traditional_scores.append(top_result['score'])

            # Check if top-1 matches
            if top_result['company'] == expected_matches[i]:
                traditional_correct += 1

            # Check if top-3 contains match
            if any(r['company'] == expected_matches[i] for r in results_list):
                traditional_top3_correct += 1

    trad_time = time.time() - trad_start
    trad_avg_latency = (trad_time / len(queries)) * 1000

    trad_accuracy_1 = (traditional_correct / len(queries)) * 100
    trad_accuracy_3 = (traditional_top3_correct / len(queries)) * 100

    print(f"[Traditional] Top-1 Accuracy: {trad_accuracy_1:.2f}%")
    print(f"[Traditional] Top-3 Accuracy: {trad_accuracy_3:.2f}%")
    print(f"[Traditional] Avg Score: {sum(traditional_scores)/len(traditional_scores):.4f}")
    print(f"[Traditional] Avg Latency: {trad_avg_latency:.2f}ms")

    results['traditional'] = {
        'top1_accuracy': trad_accuracy_1,
        'top3_accuracy': trad_accuracy_3,
        'avg_score': sum(traditional_scores) / len(traditional_scores),
        'avg_latency_ms': trad_avg_latency,
        'memory_rss_mb': mem_after['rss_mb'],
        'memory_increase_mb': mem_after['rss_mb'] - mem_before['rss_mb']
    }

    # --- Spark Approach (using DataFrame to simulate matching) ---
    print("\n[Spark] Creating DataFrame...")
    spark_mem_before = measure_spark_memory(spark)
    mem_before_spark = get_memory_usage()

    spark_df = spark.createDataFrame(
        [(i, c, clean_company_name(c, remove_stopwords=True))
         for i, c in enumerate(companies)]
    ).toDF('id', 'company_name', 'cleaned_name')

    spark_df.cache()
    spark_df.count()  # Materialize cache

    spark_mem_after = measure_spark_memory(spark)
    mem_after_spark = get_memory_usage()

    print(f"[Spark] DataFrame created and cached")
    print(f"[Spark] Memory: RSS {mem_after_spark['rss_mb']:.2f} MB (+{mem_after_spark['rss_mb'] - mem_before_spark['rss_mb']:.2f} MB)")

    # Simulate matching with Spark operations
    print(f"[Spark] Running {len(queries)} filter operations...")
    spark_start = time.time()

    spark_correct = 0
    spark_scores = []

    for i, query in enumerate(queries):
        # Use Spark SQL to find matching company
        from pyspark.sql.functions import instr

        # Simple similarity check (using substring matching)
        matches = spark_df.filter(
            instr(col('cleaned_name'), query.split()[0] if query.split() else '') > 0
        ).limit(3).collect()

        if matches:
            spark_correct += 1
            spark_scores.append(1.0)  # Binary score for simplicity
        else:
            spark_scores.append(0.0)

    spark_time = time.time() - spark_start
    spark_avg_latency = (spark_time / len(queries)) * 1000

    spark_accuracy = (spark_correct / len(queries)) * 100

    print(f"[Spark] Accuracy (basic matching): {spark_accuracy:.2f}%")
    print(f"[Spark] Avg Latency: {spark_avg_latency:.2f}ms")
    print(f"[Spark] Memory Increase: {mem_after_spark['rss_mb'] - mem_before_spark['rss_mb']:.2f} MB")

    results['spark'] = {
        'accuracy': spark_accuracy,
        'avg_latency_ms': spark_avg_latency,
        'memory_rss_mb': mem_after_spark['rss_mb'],
        'memory_increase_mb': mem_after_spark['rss_mb'] - mem_before_spark['rss_mb'],
        'spark_memory': spark_mem_after
    }

    # Comparison
    print(f"\n[Comparison] Traditional Top-1: {trad_accuracy_1:.2f}% vs Spark: {spark_accuracy:.2f}%")
    print(f"[Comparison] Traditional maintains {(trad_accuracy_1 - spark_accuracy):.2f}% accuracy advantage")

    return results


def quality_validation_2_data_integrity(spark) -> Dict:
    """
    Quality Validation 2: Verify data integrity through Spark pipeline
    """
    print("\n" + "="*80)
    print("QUALITY VALIDATION 2: Data Integrity")
    print("="*80)

    results = {
        'validation': 'data_integrity',
        'timestamp': datetime.now().isoformat()
    }

    # Load data
    print("\n[Setup] Loading companies...")
    with open('data/sample_system_names.txt', 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f if line.strip()]

    print(f"[Setup] Loaded {len(companies)} companies")

    # Test 1: Unicode normalization
    print("\n[Test 1] Unicode normalization...")
    test_unicode = "CÔNG TY TNHH SỮA VIỆT NAM"

    # Traditional
    from src.preprocess import normalize_vietnamese_text
    trad_normalized = normalize_vietnamese_text(test_unicode)

    # Spark
    test_df = spark.createDataFrame([(test_unicode,)], ['original'])
    test_df = test_df.select(
        trim(col('original')).alias('normalized')
    )
    spark_normalized = test_df.collect()[0]['normalized']

    unicode_match = trad_normalized == spark_normalized
    print(f"[Test 1] Traditional: '{trad_normalized}'")
    print(f"[Test 1] Spark: '{spark_normalized}'")
    print(f"[Test 1] Match: {unicode_match}")

    results['unicode_normalization'] = {
        'traditional': trad_normalized,
        'spark': spark_normalized,
        'match': unicode_match
    }

    # Test 2: Stopword removal consistency
    print("\n[Test 2] Stopword removal consistency...")
    test_company = "CÔNG TY TNHH SỮA VIỆT NAM"

    trad_cleaned = clean_company_name(test_company, remove_stopwords=True)

    # Spark UDF for stopword removal
    @udf(returnType=StringType())
    def spark_clean(text: str) -> str:
        return clean_company_name(text, remove_stopwords=True)

    test_df2 = spark.createDataFrame([(test_company,)], ['company'])
    test_df2 = test_df2.select(
        spark_clean(col('company')).alias('cleaned')
    )
    spark_cleaned = test_df2.collect()[0]['cleaned']

    stopword_match = trad_cleaned == spark_cleaned
    print(f"[Test 2] Traditional: '{trad_cleaned}'")
    print(f"[Test 2] Spark: '{spark_cleaned}'")
    print(f"[Test 2] Match: {stopword_match}")

    results['stopword_removal'] = {
        'traditional': trad_cleaned,
        'spark': spark_cleaned,
        'match': stopword_match
    }

    # Test 3: Batch processing consistency
    print("\n[Test 3] Batch processing consistency...")
    sample_companies = companies[:100]

    trad_cleaned_batch = [clean_company_name(c, remove_stopwords=True) for c in sample_companies]

    # Spark batch
    batch_df = spark.createDataFrame(
        [(c,) for c in sample_companies],
        ['company']
    )
    batch_df = batch_df.select(
        spark_clean(col('company')).alias('cleaned')
    )
    spark_cleaned_batch = [row['cleaned'] for row in batch_df.collect()]

    batch_match = trad_cleaned_batch == spark_cleaned_batch
    matches = sum(1 for t, s in zip(trad_cleaned_batch, spark_cleaned_batch) if t == s)
    match_rate = (matches / len(sample_companies)) * 100

    print(f"[Test 3] Processed {len(sample_companies)} companies")
    print(f"[Test 3] Matches: {matches}/{len(sample_companies)} ({match_rate:.2f}%)")
    print(f"[Test 3] All match: {batch_match}")

    results['batch_processing'] = {
        'total': len(sample_companies),
        'matches': matches,
        'match_rate': match_rate,
        'all_match': batch_match
    }

    results['overall_integrity'] = unicode_match and stopword_match and batch_match

    return results


def memory_validation_1_scalability(spark) -> Dict:
    """
    Memory Validation 1: Measure memory consumption across corpus sizes
    """
    print("\n" + "="*80)
    print("MEMORY VALIDATION 1: Scalability Analysis")
    print("="*80)

    results = {
        'validation': 'memory_scalability',
        'timestamp': datetime.now().isoformat(),
        'sizes': []
    }

    corpus_sizes = [1000, 5000, 10000]

    for size in corpus_sizes:
        print(f"\n--- Testing {size} companies ---")

        size_results = {'num_companies': size}

        # Get baseline memory
        baseline_mem = get_memory_usage()

        # Load data
        with open('data/sample_system_names.txt', 'r', encoding='utf-8') as f:
            companies = [line.strip() for line in f]
        while len(companies) < size:
            companies = companies + companies
        companies = companies[:size]

        # Traditional approach memory
        print(f"[Traditional] Building index for {size} companies...")
        trad_mem_before = get_memory_usage()

        trad_matcher = CompanyMatcher(model_name='tfidf-bm25')
        trad_matcher.build_index(companies)

        trad_mem_after = get_memory_usage()
        trad_mem_increase = trad_mem_after['rss_mb'] - trad_mem_before['rss_mb']

        print(f"[Traditional] Memory: {trad_mem_after['rss_mb']:.2f} MB (+{trad_mem_increase:.2f} MB)")

        size_results['traditional'] = {
            'memory_rss_mb': trad_mem_after['rss_mb'],
            'memory_increase_mb': trad_mem_increase,
            'memory_per_company_mb': trad_mem_increase / size
        }

        # Spark approach memory
        print(f"[Spark] Creating DataFrame for {size} companies...")
        spark_mem_before = measure_spark_memory(spark)
        mem_spark_before = get_memory_usage()

        spark_df = spark.createDataFrame(
            [(i, c, clean_company_name(c, remove_stopwords=True))
             for i, c in enumerate(companies)]
        ).toDF('id', 'company_name', 'cleaned_name')

        spark_df.cache()
        spark_df.count()  # Materialize

        # Force garbage collection
        import gc
        gc.collect()

        mem_spark_after = get_memory_usage()
        spark_mem_increase = mem_spark_after['rss_mb'] - mem_spark_before['rss_mb']

        print(f"[Spark] Memory: {mem_spark_after['rss_mb']:.2f} MB (+{spark_mem_increase:.2f} MB)")

        size_results['spark'] = {
            'memory_rss_mb': mem_spark_after['rss_mb'],
            'memory_increase_mb': spark_mem_increase,
            'memory_per_company_mb': spark_mem_increase / size
        }

        # Cleanup
        spark_df.unpersist()
        del trad_matcher

        results['sizes'].append(size_results)

    # Analysis
    print("\n[Analysis] Memory Scalability:")
    for size_result in results['sizes']:
        n = size_result['num_companies']
        trad_per_co = size_result['traditional']['memory_per_company_mb']
        spark_per_co = size_result['spark']['memory_per_company_mb']
        print(f"  {n:5d} companies: Traditional {trad_per_co:.6f} MB/co, Spark {spark_per_co:.6f} MB/co")

    return results


def memory_validation_2_optimization(spark) -> Dict:
    """
    Memory Validation 2: Test memory optimization techniques
    """
    print("\n" + "="*80)
    print("MEMORY VALIDATION 2: Optimization Techniques")
    print("="*80)

    results = {
        'validation': 'memory_optimization',
        'timestamp': datetime.now().isoformat(),
        'techniques': []
    }

    # Load data
    print("\n[Setup] Loading 5000 companies...")
    with open('data/sample_system_names.txt', 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f]
    while len(companies) < 5000:
        companies = companies + companies
    companies = companies[:5000]

    # Technique 1: No caching
    print("\n[Test 1] No caching...")
    baseline_mem = get_memory_usage()

    df1 = spark.createDataFrame(
        [(i, c, clean_company_name(c, remove_stopwords=True))
         for i, c in enumerate(companies)]
    ).toDF('id', 'company_name', 'cleaned_name')

    count1 = df1.count()
    mem1 = get_memory_usage()

    print(f"[Test 1] Count: {count1}, Memory: {mem1['rss_mb']:.2f} MB")

    results['techniques'].append({
        'technique': 'no_cache',
        'memory_rss_mb': mem1['rss_mb'],
        'memory_increase_mb': mem1['rss_mb'] - baseline_mem['rss_mb']
    })

    # Technique 2: With caching
    print("\n[Test 2] With caching...")
    baseline_mem2 = get_memory_usage()

    df2 = spark.createDataFrame(
        [(i, c, clean_company_name(c, remove_stopwords=True))
         for i, c in enumerate(companies)]
    ).toDF('id', 'company_name', 'cleaned_name')

    df2.cache()
    count2a = df2.count()  # First access
    count2b = df2.count()  # Second access (should be cached)

    mem2 = get_memory_usage()

    print(f"[Test 2] Count 1: {count2a}, Count 2: {count2b}, Memory: {mem2['rss_mb']:.2f} MB")

    results['techniques'].append({
        'technique': 'cache',
        'memory_rss_mb': mem2['rss_mb'],
        'memory_increase_mb': mem2['rss_mb'] - baseline_mem2['rss_mb']
    })

    df2.unpersist()

    # Technique 3: Repartitioning
    print("\n[Test 3] Repartitioning (4 partitions)...")
    baseline_mem3 = get_memory_usage()

    df3 = spark.createDataFrame(
        [(i, c, clean_company_name(c, remove_stopwords=True))
         for i, c in enumerate(companies)]
    ).toDF('id', 'company_name', 'cleaned_name')

    df3_repartitioned = df3.repartition(4)
    count3 = df3_repartitioned.count()

    mem3 = get_memory_usage()

    print(f"[Test 3] Count: {count3}, Memory: {mem3['rss_mb']:.2f} MB")

    results['techniques'].append({
        'technique': 'repartition_4',
        'memory_rss_mb': mem3['rss_mb'],
        'memory_increase_mb': mem3['rss_mb'] - baseline_mem3['rss_mb']
    })

    # Comparison
    print("\n[Summary] Memory Optimization Results:")
    for tech in results['techniques']:
        print(f"  {tech['technique']:20s}: {tech['memory_rss_mb']:.2f} MB")

    return results


def run_all_validations(spark) -> Dict:
    """Run all quality and memory validations."""
    print("\n" + "="*80)
    print("SPARK QUALITY AND MEMORY VALIDATION SUITE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Spark version: {spark.version}")

    # Get initial system memory
    initial_mem = get_memory_usage()
    print(f"\nSystem Memory: {initial_mem['rss_mb']:.2f} MB RSS, {initial_mem['percent']:.1f}% of RAM")

    all_results = {
        'suite': 'spark_quality_memory_validation',
        'start_time': datetime.now().isoformat(),
        'initial_memory_mb': initial_mem['rss_mb'],
        'validations': []
    }

    # Load base data
    with open('data/sample_system_names.txt', 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f if line.strip()]

    try:
        # Run validations
        val1 = quality_validation_1_accuracy_comparison(spark, companies)
        all_results['validations'].append(val1)

        val2 = quality_validation_2_data_integrity(spark)
        all_results['validations'].append(val2)

        val3 = memory_validation_1_scalability(spark)
        all_results['validations'].append(val3)

        val4 = memory_validation_2_optimization(spark)
        all_results['validations'].append(val4)

    finally:
        spark.stop()

    # Final memory measurement
    final_mem = get_memory_usage()
    all_results['end_time'] = datetime.now().isoformat()
    all_results['final_memory_mb'] = final_mem['rss_mb']
    all_results['memory_delta_mb'] = final_mem['rss_mb'] - initial_mem['rss_mb']

    return all_results


def save_results(results: Dict, output_file: str):
    """Save validation results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main entry point."""
    # Setup Spark
    print("Initializing Spark session...")
    spark = create_spark_session(
        use_databricks=False,
        local_config=get_local_spark_config(
            app_name="QualityMemoryValidation",
            executor_memory="2g",
            driver_memory="2g",
            cores=4
        )
    )

    print(f"✅ Spark session created")
    print(f"   Spark UI: {spark.sparkContext.uiWebUrl}")

    # Run validations
    results = run_all_validations(spark)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data/eval/spark_validation_{timestamp}.json'

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    save_results(results, output_file)

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\nInitial Memory: {results['initial_memory_mb']:.2f} MB")
    print(f"Final Memory: {results['final_memory_mb']:.2f} MB")
    print(f"Memory Delta: {results['memory_delta_mb']:.2f} MB")

    print(f"\nValidations Completed: {len(results['validations'])}")
    for val in results['validations']:
        val_name = val.get('validation', 'unknown')
        print(f"  ✓ {val_name}")

    print(f"\n📊 Full results: {output_file}")


if __name__ == "__main__":
    main()
