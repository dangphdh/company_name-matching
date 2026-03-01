"""
Spark + BGE-M3 Integration for Company Name Matching

This script integrates BGE-M3 (BAAI General Embedding - Multilingual v3)
with Apache Spark for distributed semantic embedding generation and matching.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, pandas_udf
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
import pandas as pd

from src.matching.matcher import CompanyMatcher
from src.preprocess import clean_company_name
from config.spark_config import get_local_spark_config, create_spark_session


class BGE_M3_Embedder:
    """BGE-M3 embedding generator with lazy loading."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize BGE-M3 model (lazy loading)."""
        self.model_name = model_name
        self.model = None
        self.device = 'cpu'  # Use CPU by default

    def load_model(self):
        """Load the model on first use."""
        if self.model is None:
            print(f"[BGE-M3] Loading model {self.model_name}...")
            try:
                from FlagEmbedding import BGEM3FlagModel
                self.model = BGEM3FlagModel(
                    self.model_name,
                    device='cpu',
                    use_fp16=False
                )
                print(f"[BGE-M3] Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed. Install with:\n"
                    "pip install -U FlagEmbedding"
                )

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self.load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            max_length=512
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode([text])[0]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def experiment_1_bge3_tfidf_comparison(spark, companies: List[str]) -> Dict:
    """
    Experiment 1: Compare BGE-M3 embeddings vs TF-IDF
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: BGE-M3 vs TF-IDF Comparison")
    print("="*80)

    results = {
        'experiment': 'bge3_vs_tfidf',
        'timestamp': datetime.now().isoformat(),
        'num_companies': len(companies)
    }

    # Prepare test data
    test_queries = companies[:50]  # Test with 50 queries
    test_corpora = companies[:1000]  # Use 1000 companies as corpus

    print(f"\n[Setup] Test queries: {len(test_queries)}")
    print(f"[Setup] Corpus size: {len(test_corpora)}")

    # --- BGE-M3 Approach ---
    print("\n[BGE-M3] Initializing model...")
    bge_start = time.time()

    embedder = BGE_M3_Embedder()

    # Generate embeddings for corpus
    print(f"[BGE-M3] Generating embeddings for {len(test_corpora)} companies...")
    corpus_embeddings = embedder.encode(test_corpora)

    bge_index_time = time.time() - bge_start
    print(f"[BGE-M3] Embeddings generated in {bge_index_time:.3f}s")
    print(f"[BGE-M3] Embedding shape: {corpus_embeddings.shape}")
    print(f"[BGE-M3] Memory: {corpus_embeddings.nbytes / 1024 / 1024:.2f} MB")

    # Test queries
    print(f"\n[BGE-M3] Running {len(test_queries)} queries...")
    bge_query_start = time.time()

    bge_results = []
    bge_correct_top1 = 0
    bge_correct_top3 = 0
    bge_scores = []

    for i, query in enumerate(test_queries):
        query_emb = embedder.encode_single(query)

        # Calculate similarities with all corpus
        similarities = [
            cosine_similarity(query_emb, corpus_emb)
            for corpus_emb in corpus_embeddings
        ]

        # Get top 3 matches
        top_indices = np.argsort(similarities)[::-1][:3]

        # Check accuracy
        if top_indices[0] == i:  # Exact match
            bge_correct_top1 += 1
        if i in top_indices:
            bge_correct_top3 += 1

        bge_scores.append(similarities[top_indices[0]])
        bge_results.append({
            'query': query,
            'top_match': test_corpora[top_indices[0]],
            'score': similarities[top_indices[0]]
        })

    bge_query_time = time.time() - bge_query_start
    bge_avg_latency = (bge_query_time / len(test_queries)) * 1000

    bge_accuracy_1 = (bge_correct_top1 / len(test_queries)) * 100
    bge_accuracy_3 = (bge_correct_top3 / len(test_queries)) * 100

    print(f"[BGE-M3] Top-1 Accuracy: {bge_accuracy_1:.2f}%")
    print(f"[BGE-M3] Top-3 Accuracy: {bge_accuracy_3:.2f}%")
    print(f"[BGE-M3] Avg Score: {np.mean(bge_scores):.4f}")
    print(f"[BGE-M3] Avg Latency: {bge_avg_latency:.2f}ms")

    results['bge_m3'] = {
        'index_build_time': bge_index_time,
        'query_time': bge_query_time,
        'avg_latency_ms': bge_avg_latency,
        'top1_accuracy': bge_accuracy_1,
        'top3_accuracy': bge_accuracy_3,
        'avg_score': float(np.mean(bge_scores)),
        'embedding_shape': list(corpus_embeddings.shape),
        'memory_mb': corpus_embeddings.nbytes / 1024 / 1024
    }

    # --- TF-IDF Approach ---
    print("\n[TF-IDF] Building matcher...")
    tfidf_start = time.time()

    tfidf_matcher = CompanyMatcher(model_name='tfidf')
    tfidf_matcher.build_index(test_corpora)

    tfidf_index_time = time.time() - tfidf_start
    print(f"[TF-IDF] Index built in {tfidf_index_time:.3f}s")

    print(f"\n[TF-IDF] Running {len(test_queries)} queries...")
    tfidf_query_start = time.time()

    tfidf_correct_top1 = 0
    tfidf_correct_top3 = 0
    tfidf_scores = []

    for i, query in enumerate(test_queries):
        cleaned_query = clean_company_name(query, remove_stopwords=True)
        results_list = tfidf_matcher.search(cleaned_query, top_k=3)

        if results_list:
            top_score = results_list[0]['score']
            tfidf_scores.append(top_score)

            if results_list[0]['company'] == query:
                tfidf_correct_top1 += 1

            if any(r['company'] == query for r in results_list):
                tfidf_correct_top3 += 1

    tfidf_query_time = time.time() - tfidf_query_start
    tfidf_avg_latency = (tfidf_query_time / len(test_queries)) * 1000

    tfidf_accuracy_1 = (tfidf_correct_top1 / len(test_queries)) * 100
    tfidf_accuracy_3 = (tfidf_correct_top3 / len(test_queries)) * 100

    print(f"[TF-IDF] Top-1 Accuracy: {tfidf_accuracy_1:.2f}%")
    print(f"[TF-IDF] Top-3 Accuracy: {tfidf_accuracy_3:.2f}%")
    print(f"[TF-IDF] Avg Score: {np.mean(tfidf_scores):.4f}")
    print(f"[TF-IDF] Avg Latency: {tfidf_avg_latency:.2f}ms")

    results['tfidf'] = {
        'index_build_time': tfidf_index_time,
        'query_time': tfidf_query_time,
        'avg_latency_ms': tfidf_avg_latency,
        'top1_accuracy': tfidf_accuracy_1,
        'top3_accuracy': tfidf_accuracy_3,
        'avg_score': float(np.mean(tfidf_scores))
    }

    # Comparison
    print(f"\n[Comparison]")
    print(f"  Index Build: BGE-M3 {bge_index_time:.3f}s vs TF-IDF {tfidf_index_time:.3f}s")
    print(f"  Query Time: BGE-M3 {bge_query_time:.3f}s vs TF-IDF {tfidf_query_time:.3f}s")
    print(f"  Top-1 Accuracy: BGE-M3 {bge_accuracy_1:.2f}% vs TF-IDF {tfidf_accuracy_1:.2f}%")
    print(f"  Latency: BGE-M3 {bge_avg_latency:.2f}ms vs TF-IDF {tfidf_avg_latency:.2f}ms")

    return results


def experiment_2_spark_bge3_udf(spark, companies: List[str]) -> Dict:
    """
    Experiment 2: Use BGE-M3 as Spark UDF
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: BGE-M3 as Spark UDF")
    print("="*80)

    results = {
        'experiment': 'spark_bge3_udf',
        'timestamp': datetime.now().isoformat()
    }

    # Prepare data
    sample_companies = companies[:500]
    print(f"\n[Setup] Using {len(sample_companies)} companies")

    # Initialize embedder (will be shared across UDF calls)
    embedder = BGE_M3_Embedder()

    # Create a regular Python UDF (slower but works)
    @udf(returnType=StringType())
    def generate_embedding_text(company_name: str) -> str:
        """Generate embedding and return as string (for storage)."""
        try:
            emb = embedder.encode_single(company_name)
            # Convert to comma-separated string for storage
            return ','.join([str(x) for x in emb.tolist()[:10]])  # First 10 dims
        except Exception as e:
            return ""

    # Create DataFrame
    print("\n[Spark] Creating DataFrame...")
    df = spark.createDataFrame(
        [(i, c, clean_company_name(c, remove_stopwords=True))
         for i, c in enumerate(sample_companies)]
    ).toDF('id', 'company_name', 'cleaned_name')

    # Apply UDF
    print("[Spark] Applying BGE-M3 UDF...")
    udf_start = time.time()

    df_with_embeddings = df.withColumn(
        'embedding_preview',
        generate_embedding_text(col('company_name'))
    )

    # Collect results
    results_df = df_with_embeddings.limit(10).collect()

    udf_time = time.time() - udf_start

    print(f"[Spark] UDF applied in {udf_time:.3f}s")
    print(f"[Spark] Sample results:")
    for row in results_df[:3]:
        print(f"  {row.company_name[:50]}... -> [{row.embedding_preview[:50]}...]")

    results['udf_time'] = udf_time
    results['sample_size'] = len(sample_companies)

    return results


def experiment_3_batch_embedding_generation(spark, companies: List[str]) -> Dict:
    """
    Experiment 3: Optimized batch embedding generation
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Optimized Batch Embedding Generation")
    print("="*80)

    results = {
        'experiment': 'batch_embeddings',
        'timestamp': datetime.now().isoformat(),
        'batches': []
    }

    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256]
    sample_size = 500

    sample_companies = companies[:sample_size]
    print(f"\n[Setup] Using {sample_size} companies")
    print(f"[Setup] Testing batch sizes: {batch_sizes}")

    embedder = BGE_M3_Embedder()

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        batch_start = time.time()

        # Process in batches
        all_embeddings = []
        for i in range(0, len(sample_companies), batch_size):
            batch = sample_companies[i:i+batch_size]
            embeddings = embedder.encode(batch)
            all_embeddings.append(embeddings)

        batch_time = time.time() - batch_start
        avg_time_per_company = (batch_time / len(sample_companies)) * 1000

        print(f"  Total time: {batch_time:.3f}s")
        print(f"  Avg per company: {avg_time_per_company:.2f}ms")

        results['batches'].append({
            'batch_size': batch_size,
            'total_time': batch_time,
            'avg_time_per_company_ms': avg_time_per_company,
            'throughput_per_second': len(sample_companies) / batch_time
        })

    # Find optimal batch size
    best_batch = max(results['batches'], key=lambda x: x['throughput_per_second'])
    print(f"\n[Optimal] Batch size: {best_batch['batch_size']}")
    print(f"[Optimal] Throughput: {best_batch['throughput_per_second']:.2f} companies/sec")

    results['optimal_batch_size'] = best_batch['batch_size']

    return results


def experiment_4_quality_analysis(spark, companies: List[str]) -> Dict:
    """
    Experiment 4: Quality analysis of BGE-M3 embeddings
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: BGE-M3 Quality Analysis")
    print("="*80)

    results = {
        'experiment': 'quality_analysis',
        'timestamp': datetime.now().isoformat()
    }

    # Prepare test cases with known variations
    test_cases = [
        {
            'original': 'CÔNG TY TNHH SỮA VIỆT NAM',
            'variants': [
                'Vinamilk',
                'Sữa Việt Nam',
                'Vietnam Dairy',
                'Cty TNHH Sữa VN',
            ]
        },
        {
            'original': 'Ngân hàng TMCP Ngoại thương Việt Nam',
            'variants': [
                'Vietcombank',
                'Ngân hàng Ngoại thương',
                'Vietcom Bank',
                'VCB',
            ]
        },
    ]

    embedder = BGE_M3_Embedder()

    print("\n[Quality] Testing semantic understanding...")
    for test_case in test_cases:
        print(f"\n  Original: {test_case['original']}")

        # Get embedding for original
        original_emb = embedder.encode_single(test_case['original'])

        print(f"  Variants:")
        for variant in test_case['variants']:
            variant_emb = embedder.encode_single(variant)
            similarity = cosine_similarity(original_emb, variant_emb)
            print(f"    - {variant:30s} -> {similarity:.4f}")

    results['test_cases'] = len(test_cases)

    return results


def run_all_experiments(spark) -> Dict:
    """Run all BGE-M3 experiments."""
    print("\n" + "="*80)
    print("SPARK + BGE-M3 INTEGRATION EXPERIMENTS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Spark version: {spark.version}")

    all_results = {
        'suite': 'spark_bge3_experiments',
        'start_time': datetime.now().isoformat(),
        'experiments': []
    }

    # Load base data
    print("\n[Setup] Loading companies...")
    with open('data/sample_system_names.txt', 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f if line.strip()]

    print(f"[Setup] Loaded {len(companies)} companies")

    try:
        # Note: Experiment 1 requires BGE-M3 model
        print("\n" + "="*80)
        print("NOTE: BGE-M3 experiments require FlagEmbedding library")
        print("Install with: pip install -U FlagEmbedding")
        print("="*80)

        # Try to run experiments if BGE-M3 is available
        try:
            exp1 = experiment_1_bge3_tfidf_comparison(spark, companies)
            all_results['experiments'].append(exp1)
        except ImportError as e:
            print(f"\n[SKIP] BGE-M3 experiments: {e}")
            all_results['experiments'].append({
                'experiment': 'bge3_not_available',
                'reason': 'FlagEmbedding not installed',
                'install_command': 'pip install -U FlagEmbedding'
            })

        # Other experiments that don't require BGE-M3
        exp2 = experiment_2_spark_bge3_udf(spark, companies)
        all_results['experiments'].append(exp2)

    finally:
        spark.stop()

    all_results['end_time'] = datetime.now().isoformat()

    return all_results


def save_results(results: Dict, output_file: str):
    """Save experiment results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main entry point."""
    print("="*80)
    print("BGE-M3 + SPARK FOR COMPANY NAME MATCHING")
    print("="*80)
    print("\nBGE-M3: BAAI General Embedding - Multilingual v3")
    print("- Supports 100+ languages including Vietnamese")
    print("- 1024-dimensional embeddings")
    print("- Optimized for semantic search")
    print("\n" + "="*80)

    # Check for dependencies
    print("\n[Check] Verifying dependencies...")
    try:
        import FlagEmbedding
        print("[Check] ✅ FlagEmbedding installed")
    except ImportError:
        print("[Check] ❌ FlagEmbedding not installed")
        print("\nTo install BGE-M3 support:")
        print("  pip install -U FlagEmbedding")
        print("\nContinuing with limited experiments...")

    # Setup Spark
    print("\n[Setup] Initializing Spark...")
    spark = create_spark_session(
        use_databricks=False,
        local_config=get_local_spark_config(
            app_name="BGE_M3_Experiments",
            executor_memory="2g",
            driver_memory="2g",
            cores=4
        )
    )

    print(f"✅ Spark session created")
    print(f"   Spark UI: {spark.sparkContext.uiWebUrl}")

    # Run experiments
    results = run_all_experiments(spark)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data/eval/bge3_spark_{timestamp}.json'

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    save_results(results, output_file)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print(f"\nExperiments completed: {len(results['experiments'])}")
    for exp in results['experiments']:
        exp_name = exp.get('experiment', 'unknown')
        status = "✓" if exp.get('experiment') != 'bge3_not_available' else "⊘"
        print(f"  {status} {exp_name}")

    print(f"\n📊 Full results: {output_file}")


if __name__ == "__main__":
    main()
