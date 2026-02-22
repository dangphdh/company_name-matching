# Distributed Vietnamese Company Name Matching - Databricks Batch Solution

## Overview
This document outlines a scalable, distributed architecture for the Vietnamese company name matching system designed to run on Databricks. The solution is a **batch-only pipeline**: it matches a set of new/incoming company names against an existing reference corpus and writes results to Delta Lake. There is no real-time serving layer.

## Current Limitations & Requirements

### Current System Constraints
- **Memory-bound**: In-memory indexes limit scalability
- **Single-node**: Cannot leverage distributed computing
- **Sequential processing**: Batch operations are slow
- **No fault tolerance**: Single point of failure

### Target Requirements
- **Scale**: Handle 1M+ reference companies and large input batches efficiently
- **Throughput**: Process 100K+ new company names per job run
- **Accuracy**: ≥90% Top-1 accuracy (production model: `tfidf[sw=F]-rerank(n=5)+bge-m3`, `min_score=0.76` → 92.0% Precision@1, 97.5% coverage)
- **Reliability**: Fault-tolerant distributed batch processing
- **Cost-efficiency**: Run on scheduled or on-demand Databricks jobs; no always-on serving cluster

## Distributed Architecture Design

### Core Components

#### 1. Data Layer (Delta Lake)
```
Reference Data  → Bronze Layer → Silver Layer → Gold Layer (TF-IDF Index)
New Input Batch → Bronze Layer → Silver Layer ─────────────────────┐
                                                                   ↓
                                                      Batch Match  →  Results Table
```
- **Bronze**: Raw company data ingestion (reference + new input)
- **Silver**: Cleaned and preprocessed data (both datasets)
- **Gold**: Vectorized reference data with precomputed TF-IDF matrix
- **Results**: Match output table (`company_matcher.match_results`)

#### 2. Processing Layer (Apache Spark)
```
Preprocess Reference → Build TF-IDF Index → Preprocess New Batch → Batch Match → Write Results
```

## Detailed Implementation

### 1. Data Ingestion & Storage

#### Delta Lake Schema Design
```python
# Bronze Layer - Raw Data
bronze_schema = StructType([
    StructField("company_id", StringType(), True),
    StructField("raw_name", StringType(), True),
    StructField("source", StringType(), True),
    StructField("ingestion_timestamp", TimestampType(), True)
])

# Silver Layer - Preprocessed Data
silver_schema = StructType([
    StructField("company_id", StringType(), False),
    StructField("clean_name", StringType(), False),
    StructField("accented_name", StringType(), False),
    StructField("unaccented_name", StringType(), False),
    StructField("company_type", StringType(), True),
    StructField("brand_name", StringType(), False),
    StructField("processed_timestamp", TimestampType(), False)
])

# Gold Layer - Vectorized Reference Data (precomputed, reused across batch runs)
gold_schema = StructType([
    StructField("company_id", StringType(), False),
    StructField("clean_name", StringType(), False),
    StructField("tfidf_vector", ArrayType(FloatType()), False),
    StructField("partition_key", StringType(), False)   # for distributed index sharding
])

# Results Table - Batch Match Output
results_schema = StructType([
    StructField("input_id", StringType(), False),          # ID of incoming company
    StructField("input_name", StringType(), False),        # Original input name
    StructField("matched_company_id", StringType(), True), # Best match in reference
    StructField("matched_name", StringType(), True),       # Matched company name
    StructField("score", FloatType(), False),              # Cosine similarity score
    StructField("rank", IntegerType(), False),             # 1 = top match
    StructField("batch_id", StringType(), False),          # Job run identifier
    StructField("matched_at", TimestampType(), False)
])
```

#### Distributed Preprocessing Pipeline
```python
def distributed_preprocessing(spark, bronze_df):
    """Distributed preprocessing using Spark UDFs"""

    # Broadcast stopwords for efficiency
    stop_words_broadcast = spark.sparkContext.broadcast(stop_words)

    @pandas_udf(StringType())
    def preprocess_company_names(names):
        preprocessor = CompanyPreprocessor()
        return names.apply(lambda x: preprocessor.clean_company_name(x))

    # Process in parallel across partitions
    silver_df = bronze_df.withColumn(
        "clean_name",
        preprocess_company_names("raw_name")
    ).withColumn(
        "accented_name",
        preprocess_company_names("raw_name")
    ).withColumn(
        "unaccented_name",
        remove_accents_udf("clean_name")
    )

    return silver_df
```

### 2. Distributed Vectorization

#### TF-IDF Vectorization at Scale

**Production model**: `tfidf[sw=F]-rerank(n=5)+bge-m3` with `min_score=0.76`
- First stage: TF-IDF char n-gram (2–5), `remove_stopwords=False`, with entity-type normalization
- Second stage: BGE-M3 (`BAAI/bge-m3`) reranks top-5 TF-IDF candidates per query
- Confidence threshold: `min_score=0.76` — abstains on ~2.5% of low-confidence queries, raises Precision@1 from 90.3% → 92.0%
- Source: [MODEL_EVALUATION_RESULTS.md — best F0.5 threshold sweep, Feb 22 2026]

The TF-IDF model is **fit once on the reference corpus** and saved to DBFS/Delta. Each batch run reloads the fitted model to transform both the reference vectors (if not precomputed) and the new input batch — no retraining needed unless the reference corpus changes significantly.

```python
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
import mlflow

# Production model config (source: MODEL_EVALUATION_RESULTS.md)
PROD_MODEL = "tfidf-dense"          # tfidf first-stage + bge-m3 reranker
PROD_REMOVE_STOPWORDS = False        # sw=F: best sparse retriever (88.8% standalone)
PROD_DENSE_MODEL = "BAAI/bge-m3"    # reranker
PROD_FUSION = "tfidf-rerank"        # 2-stage: tfidf retrieves, bge-m3 reranks
PROD_RERANK_N = 5                   # rerank top-5 TF-IDF candidates
PROD_MIN_SCORE = 0.76               # best F0.5: 92.0% Precision@1, 97.5% coverage


def build_reference_tfidf(spark, silver_ref_df, model_save_path="/dbfs/models/tfidf_pipeline"):
    """Fit TF-IDF on reference corpus and save. Run once (or on corpus refresh).
    
    Uses production config: sw=False + entity-type normalization.
    Entity normalization must be applied in the silver preprocessing step before this runs.
    """

    tokenizer = Tokenizer(inputCol="clean_name", outputCol="words")

    # HashingTF is preferred over CountVectorizer for large distributed datasets
    # char n-gram (2-5) with sw=False: best sparse retriever at 88.8% Top-1
    hashing_tf = HashingTF(
        inputCol="words",
        outputCol="raw_features",
        numFeatures=2**18  # 262K features
    )

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])

    tfidf_model = pipeline.fit(silver_ref_df)
    tfidf_model.save(model_save_path)

    # Vectorize and persist reference gold data
    gold_df = tfidf_model.transform(silver_ref_df)
    gold_df.write.format("delta").mode("overwrite").saveAsTable("company_matcher.gold_reference")

    return tfidf_model


def load_tfidf_model(model_save_path="/dbfs/models/tfidf_pipeline"):
    """Load the pre-fitted TF-IDF pipeline for batch matching runs."""
    from pyspark.ml import PipelineModel
    return PipelineModel.load(model_save_path)
```

### 3. Distributed Index Building

The FAISS index is built **once** from the Gold reference vectors and written to DBFS. Batch matching runs load the index shards without rebuilding them, unless the reference corpus is refreshed.

#### FAISS Index per Partition
```python
import faiss
import numpy as np
from pyspark.sql.functions import collect_list, pandas_udf
from pyspark.sql.types import StringType

def build_faiss_index(spark, gold_df, vector_col="tfidf_features"):
    """Build one FAISS index shard per partition. Run once on corpus refresh."""

    partition_vectors = gold_df.groupBy("partition_key").agg(
        collect_list(vector_col).alias("vectors"),
        collect_list("company_id").alias("ids")
    )

    @pandas_udf(StringType())
    def build_partition_index(vectors_series, ids_series):
        vectors_array = np.array(vectors_series.tolist(), dtype=np.float32)

        if len(vectors_array) < 10_000:
            index = faiss.IndexFlatIP(vectors_array.shape[1])
        else:
            nlist = min(100, max(4, len(vectors_array) // 39))
            quantizer = faiss.IndexFlatIP(vectors_array.shape[1])
            index = faiss.IndexIVFFlat(quantizer, vectors_array.shape[1], nlist)
            index.train(vectors_array)

        index.add(vectors_array)

        # Persist shard alongside its ID mapping
        shard_id = ids_series.iloc[0]
        index_path = f"/dbfs/indexes/{shard_id}_shard.index"
        id_map_path = f"/dbfs/indexes/{shard_id}_ids.pkl"
        faiss.write_index(index, index_path)
        import pickle
        with open(id_map_path, "wb") as f:
            pickle.dump(ids_series.tolist(), f)

        return index_path

    indexed_df = partition_vectors.withColumn(
        "index_path",
        build_partition_index("vectors", "ids")
    )

    # Persist shard metadata
    indexed_df.select("partition_key", "index_path") \
        .write.format("delta").mode("overwrite") \
        .saveAsTable("company_matcher.index_metadata")

    return indexed_df
```

### 4. Batch Matching Job

This is the core job that runs on a schedule or on-demand. It reads a batch of new/incoming company names, vectorizes them using the pre-fitted TF-IDF model, searches the FAISS shards, and writes ranked match results to Delta Lake.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, lit, current_timestamp, col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType, IntegerType
import faiss
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# ── Schema for match results ─────────────────────────────────────────────────
match_result_schema = ArrayType(StructType([
    StructField("matched_company_id", StringType()),
    StructField("matched_name", StringType()),
    StructField("score", FloatType()),
    StructField("rank", IntegerType()),
]))


def run_batch_matching(
    spark,
    input_table: str,          # e.g. "company_matcher.silver_input"
    reference_table: str,      # e.g. "company_matcher.gold_reference"
    index_metadata_table: str, # e.g. "company_matcher.index_metadata"
    results_table: str,        # e.g. "company_matcher.match_results"
    model_path: str = "/dbfs/models/tfidf_pipeline",
    dense_model_name: str = PROD_DENSE_MODEL,  # "BAAI/bge-m3"
    rerank_n: int = PROD_RERANK_N,             # 5
    min_score: float = PROD_MIN_SCORE,         # 0.76 — best F0.5 threshold
    top_k: int = 3,
    batch_id: str = None,
):
    """
    Production batch matching: tfidf[sw=F]-rerank(n=5)+bge-m3 with min_score=0.76.

    Pipeline:
      1. TF-IDF (sw=False, entity-normalized) retrieves top `rerank_n` candidates per query
      2. BGE-M3 reranks the candidates using dense cosine similarity
      3. Top-1 result is kept only if score >= min_score (0.76 → 92.0% Precision@1, 97.5% coverage)
      4. Rows where score < min_score are written with matched_company_id=NULL and score=NULL
         (abstained — route to human review or fallback pipeline)

    Source: MODEL_EVALUATION_RESULTS.md — confidence threshold sweep, Feb 22 2026
    """
    if batch_id is None:
        batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # 1. Load pre-fitted TF-IDF model and vectorize input batch
    tfidf_model = load_tfidf_model(model_path)
    input_df = spark.read.table(input_table)
    vectorized_input = tfidf_model.transform(input_df)   # adds tfidf_features col

    # 2. Load FAISS index shards metadata
    index_meta = spark.read.table(index_metadata_table).collect()

    # 3. Broadcast shard paths so workers can load them locally
    shard_paths = [(row["partition_key"], row["index_path"]) for row in index_meta]
    shard_paths_bc = spark.sparkContext.broadcast(shard_paths)

    # 4. Load reference ID maps (partition_key → [company_id, ...])
    id_maps = {}
    for partition_key, index_path in shard_paths:
        id_map_path = index_path.replace("_shard.index", "_ids.pkl")
        with open(id_map_path, "rb") as f:
            id_maps[partition_key] = pickle.load(f)
    id_maps_bc = spark.sparkContext.broadcast(id_maps)

    # 5. Load reference names + vectors for BGE-M3 reranking (broadcast)
    ref_rows = spark.read.table(reference_table).select("company_id", "clean_name").collect()
    ref_names = {row["company_id"]: row["clean_name"] for row in ref_rows}
    ref_names_bc = spark.sparkContext.broadcast(ref_names)

    # 6. Load BGE-M3 model once per executor (broadcast model path, load lazily)
    dense_model_name_bc = spark.sparkContext.broadcast(dense_model_name)
    rerank_n_bc = spark.sparkContext.broadcast(rerank_n)
    min_score_bc = spark.sparkContext.broadcast(min_score)

    # 7. Batch match UDF:
    #    Stage 1 — TF-IDF retrieves top rerank_n candidates from FAISS shards
    #    Stage 2 — BGE-M3 reranks candidates using dense cosine similarity
    #    Threshold — result kept only if BGE-M3 top-1 score >= min_score (0.76)
    @pandas_udf(match_result_schema)
    def batch_search(vectors_series: pd.Series, texts_series: pd.Series) -> pd.Series:
        from sentence_transformers import SentenceTransformer
        import threading

        shard_paths = shard_paths_bc.value
        id_maps = id_maps_bc.value
        ref_names = ref_names_bc.value
        _rerank_n = rerank_n_bc.value      # 5
        _min_score = min_score_bc.value    # 0.76

        # Load BGE-M3 once per executor (thread-local)
        _tl = threading.local()
        if not hasattr(_tl, "dense_model"):
            _tl.dense_model = SentenceTransformer(dense_model_name_bc.value)

        results = []
        for vec, query_text in zip(vectors_series, texts_series):
            # ── Stage 1: TF-IDF FAISS retrieval ──────────────────────────────
            tfidf_query = np.array(vec, dtype=np.float32).reshape(1, -1)
            candidates = []

            for partition_key, index_path in shard_paths:
                index = faiss.read_index(index_path)
                k = min(_rerank_n * 2, index.ntotal)
                scores, indices = index.search(tfidf_query, k)
                ids = id_maps[partition_key]
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    company_id = ids[idx]
                    candidates.append({
                        "matched_company_id": company_id,
                        "matched_name": ref_names.get(company_id, ""),
                        "tfidf_score": float(score),
                    })

            # Keep top rerank_n candidates by TF-IDF score
            candidates.sort(key=lambda x: x["tfidf_score"], reverse=True)
            candidates = candidates[:_rerank_n]

            # ── Stage 2: BGE-M3 reranking ─────────────────────────────────────
            if not candidates:
                results.append([])
                continue

            candidate_texts = [c["matched_name"] for c in candidates]
            all_texts = [query_text] + candidate_texts
            embeddings = _tl.dense_model.encode(
                all_texts, normalize_embeddings=True, show_progress_bar=False
            )
            query_emb = embeddings[0]
            cand_embs = embeddings[1:]
            dense_scores = (cand_embs @ query_emb).tolist()  # cosine via dot on L2-normed vecs

            for c, ds in zip(candidates, dense_scores):
                c["score"] = float(ds)

            candidates.sort(key=lambda x: x["score"], reverse=True)

            # ── Confidence threshold (min_score=0.76) ─────────────────────────
            # Rows below threshold are written as abstained (empty list → NULL in results)
            top = []
            for i, c in enumerate(candidates[:top_k]):
                if i == 0 and c["score"] < _min_score:
                    # Top-1 below threshold: abstain entire row
                    break
                top.append({
                    "matched_company_id": c["matched_company_id"],
                    "matched_name": c["matched_name"],
                    "score": c["score"],
                    "rank": i + 1,
                })

            results.append(top)

        return pd.Series(results)

    # 8. Apply UDF and explode; abstained rows produce a single NULL-filled row
    from pyspark.sql.functions import explode, when, size
    from pyspark.sql.types import NullType

    matched_df = (
        vectorized_input
        .withColumn("matches", batch_search("tfidf_features", "clean_name"))
        .withColumn(
            # Keep a single NULL sentinel row for abstained queries (empty matches list)
            "match",
            when(size(col("matches")) > 0, explode(col("matches")))
            .otherwise(None)
        )
        .select(
            col("company_id").alias("input_id"),
            col("clean_name").alias("input_name"),
            col("match.matched_company_id"),   # NULL when abstained
            col("match.matched_name"),          # NULL when abstained
            col("match.score"),                 # NULL when abstained (<0.76)
            col("match.rank"),
            lit(batch_id).alias("batch_id"),
            current_timestamp().alias("matched_at"),
        )
    )

    # 9. Append results to Delta table
    matched_df.write.format("delta") \
        .mode("append") \
        .saveAsTable(results_table)

    total_inputs = input_df.count()
    abstained = matched_df.filter(col("matched_company_id").isNull()).count()
    answered = total_inputs - abstained
    print(f"Batch {batch_id}: {total_inputs} inputs → {answered} matched, "
          f"{abstained} abstained (score<{min_score}) → {results_table}")
    return matched_df
```

### 5. Performance Optimization

#### Partitioning Strategy
```python
def optimize_partitioning(df, num_partitions=None):
    """Optimize data partitioning for search performance"""

    if num_partitions is None:
        # Auto-determine based on data size
        total_companies = df.count()
        num_partitions = min(1000, max(10, total_companies // 10000))

    # Partition by company type for better locality
    partitioned_df = df.repartition(num_partitions, "company_type")

    # Add partition key
    from pyspark.sql.functions import spark_partition_id
    partitioned_df = partitioned_df.withColumn(
        "partition_key",
        spark_partition_id()
    )

    return partitioned_df
```

#### Caching Strategy
```python
# Cache frequently accessed data
vectorized_df.persist(StorageLevel.MEMORY_AND_DISK)

# Cache index metadata
index_metadata.persist(StorageLevel.MEMORY_ONLY)

# Cache preprocessing models
stopwords_broadcast = spark.sparkContext.broadcast(stop_words)
```

### 6. Monitoring & Observability

#### Batch Job Metrics
```python
import mlflow
from pyspark.sql.functions import count, avg, min as spark_min, max as spark_max

def log_batch_metrics(spark, results_table: str, batch_id: str):
    """Log per-batch matching quality and performance metrics."""

    batch_df = spark.read.table(results_table).filter(f"batch_id = '{batch_id}' AND rank = 1")

    stats = batch_df.agg(
        count("*").alias("total_matched"),
        avg("score").alias("avg_top1_score"),
        spark_min("score").alias("min_top1_score"),
        spark_max("score").alias("max_top1_score"),
    ).collect()[0]

    with mlflow.start_run(run_name=f"batch_{batch_id}"):
        mlflow.log_param("batch_id", batch_id)
        mlflow.log_metric("total_matched", stats["total_matched"])
        mlflow.log_metric("avg_top1_score", stats["avg_top1_score"])
        mlflow.log_metric("min_top1_score", stats["min_top1_score"])
        mlflow.log_metric("max_top1_score", stats["max_top1_score"])

    print(f"Batch {batch_id}: {stats['total_matched']} matches, "
          f"avg score={stats['avg_top1_score']:.4f}")


def monitor_partition_balance(spark, gold_table: str):
    """Check that reference data is evenly distributed across FAISS shards."""
    from pyspark.sql.functions import col

    partition_counts = spark.read.table(gold_table).groupBy("partition_key").count()
    avg_count = partition_counts.agg({"count": "avg"}).collect()[0][0]
    threshold = 0.5

    unbalanced = partition_counts.filter(
        (col("count") < avg_count * (1 - threshold)) |
        (col("count") > avg_count * (1 + threshold))
    )
    if unbalanced.count() > 0:
        print(f"WARNING: {unbalanced.count()} unbalanced partitions detected.")
```

## Deployment Architecture

### Job Flow

There are two distinct jobs:

1. **Corpus Refresh Job** — run when the reference data changes:
   `data_ingestion → preprocessing → index_building`

2. **Batch Matching Job** — run on schedule or triggered by new input data:
   `ingest_input → preprocess_input → batch_match → write_results`

### Databricks Job Configuration
```yaml
# jobs/corpus_refresh_job.yml
name: company-matcher-corpus-refresh
tasks:
  - task_key: data_ingestion
    job_cluster_key: shared-cluster
    python_wheel_task:
      package_name: company_matcher
      entry_point: ingestion
    libraries:
      - pypi: pyspark
      - pypi: scikit-learn

  - task_key: preprocessing
    depends_on:
      - task_key: data_ingestion
    job_cluster_key: shared-cluster
    python_wheel_task:
      package_name: company_matcher
      entry_point: preprocessing

  - task_key: index_building
    depends_on:
      - task_key: preprocessing
    job_cluster_key: high-memory-cluster
    python_wheel_task:
      package_name: company_matcher
      entry_point: build_indexes

---
# jobs/batch_matching_job.yml
name: company-matcher-batch-match
tasks:
  - task_key: ingest_input
    job_cluster_key: shared-cluster
    python_wheel_task:
      package_name: company_matcher
      entry_point: ingest_input_batch
    parameters:
      - name: input_path
        default: "{{job.parameters.input_path}}"

  - task_key: preprocess_input
    depends_on:
      - task_key: ingest_input
    job_cluster_key: shared-cluster
    python_wheel_task:
      package_name: company_matcher
      entry_point: preprocess_input

  - task_key: batch_match
    depends_on:
      - task_key: preprocess_input
    job_cluster_key: high-memory-cluster
    python_wheel_task:
      package_name: company_matcher
      entry_point: run_batch_matching
    parameters:
      - name: top_k
        default: "3"
      - name: batch_id
        default: "{{job.run_id}}"
      - name: min_score
        default: "0.76"   # best F0.5: 92.0% Precision@1, 97.5% coverage
      - name: rerank_n
        default: "5"
      - name: dense_model_name
        default: "BAAI/bge-m3"
    libraries:
      - pypi: faiss-cpu
      - pypi: sentence-transformers
```

### Cluster Configurations
```yaml
# clusters/shared-cluster.yml
cluster_name: company-matcher-shared
spark_version: 13.3.x-scala2.12
node_type_id: Standard_DS4_v2
num_workers: 4
autoscale:
  min_workers: 2
  max_workers: 8

# clusters/high-memory-cluster.yml  (used for index build + batch match)
cluster_name: company-matcher-high-mem
spark_version: 13.3.x-scala2.12
node_type_id: Standard_E8s_v3   # memory-optimised
num_workers: 4
autoscale:
  min_workers: 2
  max_workers: 16
```

## Scaling Considerations

### Horizontal Scaling
- **Data Partitioning**: Distribute reference corpus across multiple FAISS shards
- **Parallel Input Processing**: Spark partitions the input batch across workers automatically
- **Incremental Corpus Updates**: Rebuild only changed partitions via Delta Lake CDC

### Vertical Scaling
- **Memory Optimization**: Use memory-optimised node types for FAISS shard loading
- **SSD Storage**: DBFS with SSD-backed volumes for fast index shard reads
- **Batch Size Tuning**: Adjust Spark partition size (`spark.sql.shuffle.partitions`) based on input volume

### Cost Optimization
- **No always-on cluster**: Jobs spin up and terminate — no serving cluster cost
- **Spot/Preemptible Instances**: Use for preprocessing and index-build tasks
- **Index Reuse**: Corpus Refresh runs only when reference data changes, not on every batch

## Performance Benchmarks

### Production Model: `tfidf[sw=F]-rerank(n=5)+bge-m3`, `min_score=0.76`

| Metric | Value | Notes |
|--------|-------|-------|
| Top-1 accuracy (full coverage) | 90.3% | no threshold |
| **Precision@1 (answered)** | **92.0%** | **at min_score=0.76** |
| Coverage | 97.5% | ~2.5% abstained to human review |
| Top-3 accuracy | 98.0% | |
| Per-query latency (single node) | ~181 ms | dominated by BGE-M3 on CPU |

Source: MODEL_EVALUATION_RESULTS.md — evaluation on 4,019-company corpus, 1,000 test queries, seed=42 (Feb 22 2026).

### Expected Throughput (Batch Mode — distributed)
| Scenario | Input Batch Size | Estimated Runtime | Cluster |
|---|---|---|---|
| Small batch | 10K companies | ~5 min | 4-node shared |
| Medium batch | 100K companies | ~20 min | 4-node high-mem |
| Large batch | 1M companies | ~3 h | 16-node high-mem |

> Runtime is dominated by BGE-M3 encoding on CPU. Use GPU-enabled nodes (`Standard_NC6s_v3` or equivalent) to cut dense encoding time ~10×.

### Corpus Index Build Time
- **1M reference companies**: ~30 minutes on 4-node cluster
- **Incremental shard rebuild**: ~5 minutes per 100K companies changed

### Abstained Rows
Rows where BGE-M3 top-1 score < 0.76 are written with `matched_company_id = NULL`. These should be:
1. Routed to a human-review queue, **or**
2. Re-run with a fallback pipeline (e.g. `min_score=0.0` full-coverage mode for manual inspection batch)

## Migration Strategy

### Phase 1: Data Migration
1. Export current reference corpus to Delta Lake (Bronze)
2. Validate row counts and sample names against original corpus
3. Set up incremental sync for ongoing reference updates

### Phase 2: Index Build
1. Run preprocessing pipeline → Silver layer
2. Fit TF-IDF model on Silver reference data → save to DBFS
3. Build FAISS shards for Gold layer → save metadata table
4. Validate Top-1/Top-3 accuracy against evaluation dataset

### Phase 3: Batch Job Rollout
1. Deploy Batch Matching Job with sample input
2. Compare results against single-node system output on same input
3. Schedule Batch Matching Job for production input feeds

## Conclusion

This distributed architecture provides a scalable **batch-only** pipeline for Vietnamese company name matching on Databricks. New company name batches are matched against the reference corpus and results are written to Delta Lake — there is no real-time serving infrastructure.

**Production configuration** (from `MODEL_EVALUATION_RESULTS.md`):
- Model: `tfidf[sw=F]-rerank(n=5)+bge-m3`
- Threshold: `min_score=0.76` — best F0.5 balance (92.0% Precision@1, 97.5% coverage)
- ~2.5% of queries abstained (NULL output) → routed to human review

Key benefits:
- **Scalability**: Handle 1M+ reference companies and large input batches
- **Accuracy**: 92.0% Precision@1 on answered queries; 90.3% overall Top-1
- **Simplicity**: No always-on serving layer reduces cost and operational overhead
- **Reliability**: Fault-tolerant Spark batch jobs with Delta Lake checkpointing
- **Traceability**: Every batch run is identified by `batch_id`; abstained rows are flagged by NULL `matched_company_id`
- **Maintainability**: Corpus Refresh and Batch Matching are independent, separately schedulable jobs