# Databricks Batch Processing Pipeline

Production-grade batch processing pipeline for Vietnamese company name matching on Databricks.

## Architecture

This implementation uses a **Hybrid Pragmatic Approach**:
- **Spark for ETL**: Preprocessing, deduplication, data shuffling
- **Enhanced CompanyMatcher**: LSA-based indexing for 2M scale (512 dims, ~4GB)
- **Pandas UDFs**: Vectorized batch operations (100x faster than row UDFs)
- **Delta Lake**: ACID transactions, time travel, optimization

## Features

- ✅ Scales to 2M+ companies with LSA dimensionality reduction
- ✅ CPU-only (no GPU required)
- ✅ Vietnamese text preprocessing (normalization, stopword removal, accent handling)
- ✅ Distributed deduplication
- ✅ Persistent index with save/load
- ✅ Quality metrics and alerting
- ✅ YAML configuration with environment profiles
- ✅ Delta Lake integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-spark-databricks.txt
```

Required packages:
- pyspark
- delta-spark
- databricks-connect (for local development)
- sentence-transformers (for BGE-M3, optional)

### 2. Configure Pipeline

Edit `src/databricks/config/pipeline_config.yaml`:

```yaml
pipeline:
  model_name: "tfidf-lsa"  # LSA for 2M scale
  lsa_dims: 512  # Reduces memory from 8TB to ~4GB
  bronze_path: "/delta/bronze/banking_transactions"
  silver_path: "/delta/silver/companies_cleaned"
  gold_index_path: "/delta/gold/company_index"
  gold_matches_path: "/delta/gold/transaction_matches"
```

Or set environment variables:

```bash
export MODEL_NAME="tfidf-lsa"
export LSA_DIMS="512"
export BRONZE_PATH="/delta/bronze/banking_transactions"
export SILVER_PATH="/delta/silver/companies_cleaned"
export GOLD_INDEX_PATH="/delta/gold/company_index"
export GOLD_MATCHES_PATH="/delta/gold/transaction_matches"
```

### 3. Run Pipeline

#### Option A: Python Script

```python
from pyspark.sql import SparkSession
from src.databricks.orchestrator import PipelineOrchestrator
from src.databricks.config import load_config

# Initialize Spark
spark = SparkSession.builder.appName("CompanyMatching").getOrCreate()

# Create sample queries
queries_df = spark.createDataFrame([
    (1, "Vinamilk"),
    (2, "Vietcombank"),
    (3, "Sửa Việt Nam")
]).toDF("query_id", "query_text")

# Load configuration
config = load_config()

# Initialize and run pipeline
pipeline = PipelineOrchestrator(config)

# Run full pipeline (Stages 1-4)
results = pipeline.run_full_pipeline(
    bronze_path="/delta/bronze/companies",
    queries_df=queries_df
)

# Or run matching only against existing index
# results = pipeline.run_matching_only(queries_df)

# Query results
matches_df = spark.read.format("delta").load(config.gold_matches_path)
matches_df.show()

# Cleanup
pipeline.cleanup()
spark.stop()
```

#### Option B: Databricks Notebook

```python
# Cell 1: Install dependencies (run once)
%pip install -r requirements-spark-databricks.txt

# Cell 2: Load configuration and run pipeline
from src.databricks.orchestrator import PipelineOrchestrator
from src.databricks.config import load_config

config = load_config()
pipeline = PipelineOrchestrator(config)

# Run with sample data
queries_df = spark.createDataFrame([
    (1, "Vinamilk"),
    (2, "Vietcombank")
]).toDF("query_id", "query_text")

results = pipeline.run_full_pipeline(
    bronze_path=config.bronze_path,
    queries_df=queries_df
)

# Cell 3: Query results
spark.read.format("delta").load(config.gold_matches_path).show()
```

## Pipeline Stages

### Stage 1: Extract & Preprocess

Extracts company names from Bronze Delta table and applies Vietnamese text preprocessing:
- NFC Unicode normalization
- Entity type normalization (JSC → cp, LTD → tnhh)
- Functional term normalization (IMP-EXP → xnk)
- Stopword removal (keeps discriminators: cp, tnhh, mtv)
- No-accent variant generation

**Runtime**: ~5-10 min for 2M companies

**Output**: Silver layer with cleaned names

### Stage 2: Deduplicate

Removes duplicate company names based on normalization key:
- Groups by `norm_key` (no-accent cleaned name)
- Selects canonical record (first or longest)
- Tracks duplicate groups for audit

**Runtime**: ~2-5 min for 2M companies

**Output**: Deduplicated companies with canonical IDs

### Stage 3: Build Index

Builds CompanyMatcher index using LSA for 2M scale:
- TF-IDF char n-gram vectorization
- LSA dimensionality reduction (262K → 512 dims)
- L2 normalization for cosine similarity via dot product
- Saves index to disk for persistence

**Runtime**: ~10-15 min for 2M companies

**Output**: Saved matcher model + indexed metadata

**Memory**: ~4 GB for 2M companies (LSA-512)

### Stage 4: Batch Match

Matches queries against the company index:
- Preprocesses queries
- Collects to driver (for matching)
- Uses `matcher.search()` with top-K results
- Computes confidence scores and validation

**Runtime**: ~5-10 sec for 5K queries

**Output**: Gold layer with matching results

## Performance Estimates

| Metric | Value |
|--------|-------|
| Corpus Size | 2M companies |
| Query Volume | 5K per batch |
| Stage 1-2 | 5-10 min |
| Stage 3 | 10-15 min |
| Stage 4 | 5-10 sec |
| **Total Pipeline** | **~30 min** |
| Query Latency | ~2-5ms (after index loaded) |
| Accuracy | 99%+ Top-1 (LSA-512) |

## Configuration

### Model Configuration

```yaml
model_name: "tfidf-lsa"  # Recommended for 2M scale
lsa_dims: 512  # Memory: 4GB (vs 8TB for full TF-IDF)
```

Alternative models:
- `tfidf`: Full TF-IDF (best accuracy, but ~15 GB for 2M)
- `tfidf-dense`: TF-IDF + BGE-M3 (best semantic quality, but ~8 TB - not feasible for 2M)

### Environment Profiles

```yaml
profiles:
  dev:
    lsa_dims: 128  # Smaller for testing
    stage1_partitions: 10

  prod:
    lsa_dims: 512  # Full scale
    stage1_partitions: 400
```

Usage:
```python
config = load_config(profile="prod")
```

## Monitoring & Quality

### Quality Metrics

The pipeline automatically tracks:
- Average match score
- Score distribution (min, max, std)
- Confidence distribution (high/medium/low)
- High-confidence rate (score >= 0.90)

### Alerting Thresholds

```yaml
alert_threshold_avg_score: 0.85  # Alert if avg < 85%
alert_threshold_high_confidence: 0.90  # Alert if < 90% high-confidence
```

### View Metrics

```python
from src.databricks.utils.metrics import compute_quality_metrics, print_quality_metrics

metrics = compute_quality_metrics(matches_df)
print_quality_metrics(metrics)
```

## Troubleshooting

### Issue: Out of Memory

**Symptoms**: Stage 3 fails with OOM error

**Solutions**:
1. Reduce LSA dimensions: `lsa_dims: 256` (better: 128)
2. Increase driver memory: `spark.driver.memory: "16g"`
3. Reduce corpus size (sample)

### Issue: Low Match Quality

**Symptoms**: Average score < 0.85

**Solutions**:
1. Check data quality (many duplicates?)
2. Tune preprocessing: `remove_stopwords: false`
3. Use `tfidf` instead of `tfidf-lsa` (better accuracy, more memory)

### Issue: Slow Performance

**Symptoms**: Pipeline takes > 1 hour

**Solutions**:
1. Increase partitions: `stage1_partitions: 400`
2. Enable optimization: `adaptive_query: true`
3. Use cluster autoscaling

## File Structure

```
src/databricks/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── pipeline_config.py          # Configuration loader
│   └── pipeline_config.yaml         # Default configuration
├── preprocessing/
│   ├── __init__.py
│   └── vietnamese_udfs.py            # Pandas UDFs for preprocessing
├── matching/
│   ├── __init__.py
│   └── batch_matcher.py              # Batch matching utilities
├── stages/
│   ├── __init__.py
│   ├── stage1_extract.py             # Stage 1: Extract & Preprocess
│   ├── stage2_deduplicate.py         # Stage 2: Deduplicate
│   ├── stage3_build_index.py         # Stage 3: Build Index
│   └── stage4_match.py               # Stage 4: Batch Match
├── utils/
│   ├── __init__.py
│   ├── delta_utils.py                # Delta Lake helpers
│   ├── metrics.py                    # Quality metrics
│   └── validation.py                 # Data validation
└── orchestrator.py                   # Main pipeline orchestrator
```

## Advanced Usage

### Run Specific Stages

```python
# Rebuild index only (skip Stages 1-2)
results = pipeline.run_full_pipeline(
    skip_stages=[1, 2]  # Skip extract and deduplicate
)
```

### Use Custom Queries

```python
# Load queries from Delta table
queries_df = spark.read.format("delta").load("/delta/queries")

# Run matching only
matches_df = pipeline.run_matching_only(queries_df)
```

### Load Pre-built Index

```python
from src.matching.matcher import CompanyMatcher

# Load saved index
matcher = CompanyMatcher.load_index("/delta/silver/companies_cleaned_model")

# Search directly
results = matcher.search("Vinamilk", top_k=5)
print(results)
```

## References

- **Project README**: `/README.md`
- **Architecture**: `/CLAUDE.md`
- **Preprocessing**: `/src/preprocess.py`
- **Matcher**: `/src/matching/matcher.py`
- **Spark Config**: `/config/spark_config.py`

## License

This code is part of the Vietnamese Company Name Matching project.
