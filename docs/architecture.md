# System Architecture

Technical architecture and design decisions for the Vietnamese company name matching system.

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Design Decisions](#key-design-decisions)
- [Text Processing](#text-processing)
- [Matching Algorithms](#matching-algorithms)
- [Scalability](#scalability)

## Overview

The system is designed as a high-performance, real-time entity matching solution specifically optimized for Vietnamese company names. It achieves >99% accuracy with sub-3ms latency without requiring GPU acceleration.

**Tech Stack:**
- Python 3.x
- scikit-learn for TF-IDF/BM25 vectorization
- Optional: WordLlama embeddings, Spark/Databricks for distributed processing
- LLM integration (OpenAI/Zhipu GLM) for synthetic data generation

## Pipeline Architecture

The matching system follows a three-stage pipeline:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Input Query    │ -> │  Preprocessing   │ -> │  Vectorization  │
│  (raw text)     │    │  (normalize,     │    │  (TF-IDF/BM25)  │
│                 │    │   clean,         │    │                 │
│                 │    │   stopword)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Results       │ <- │ Similarity Search│ <- │  Indexed Corpus │
│   (ranked list) │    │  (cosine score)  │    │  (vectors +     │
│                 │    │                  │    │   metadata)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Stage 1: Preprocessing

Located in: `src/preprocess.py`

**Steps:**
1. **Unicode Normalization**: Ensure NFC form for all Vietnamese characters
2. **Accent Handling**: Generate both accented and no-accent variants
3. **Stopword Removal**: Remove company type terms and noise
4. **Character Filtering**: Keep meaningful symbols (&, +, -) and remove others

**Why This Matters:**
- Vietnamese text has multiple Unicode representations
- Users search with and without accents
- Legal entity types (TNHH, CP, MTV) obscure brand names

### Stage 2: Vectorization

Located in: `src/matching/matcher.py`

**Supported Models:**

| Model | Algorithm | Best For | Speed |
|-------|-----------|----------|-------|
| `tfidf` | Char N-gram TF-IDF | Typos, abbreviations | Fastest |
| `bm25` | Word-level BM25 | Exact terms | Fast |
| `tfidf-bm25` | Hybrid (combined) | General use | Balanced |
| `lsa` | TF-IDF + SVD | Large corpora | Moderate |
| `wordllama-*` | Embedding | Semantic similarity | Slow (GPU) |

### Stage 3: Similarity Search

**Dual Indexing Strategy:**
- Each company name generates **two** index entries:
  1. Accented version: "CÔNG TY TNHH SỮA VIỆT NAM"
  2. No-accent version: "CONG TY TNHH SUA VIET NAM"

**Scoring:**
- Cosine similarity for TF-IDF/LSA vectors
- BM25 relevance score
- Hybrid: Weighted combination with min-max normalization

## Key Design Decisions

### Why TF-IDF Char N-gram over Word Embeddings?

**Decision**: Use character-level TF-IDF (n-gram 2-5) instead of word embeddings like Word2Vec or BERT.

**Rationale:**

1. **Handles Abbreviations Better**
   - "TNHH" vs "Trách nhiệm hữu hạn" → both reduce to same brand name after stopwords
   - Char n-grams match character patterns, not semantic word meanings

2. **Robust to Typos**
   - "Samsung" vs "Samsng" (missing 'u') → high similarity
   - Character-level matching catches partial matches

3. **Word Reorderings**
   - "A&P TM DV" vs "TM DV A&P" → still matches well
   - Char n-grams are position-independent

4. **Performance**
   - Sub-millisecond latency without GPU
   - Scales to millions of companies
   - Lower memory footprint than embeddings

**Trade-off**: Less effective for pure semantic similarity (e.g., "Sữa" vs "Dairy products"), but this is acceptable for company name matching where brand names are proper nouns.

### Why Dual Variant Indexing?

**Problem**: Users search for "Vinamilk" but the corpus has "CÔNG TY TNHH SỮA VIỆT NAM".

**Solution**: Index both accented and no-accent versions.

**Benefits:**
- Query: "Vinamilk" (no accent) matches both corpus variants
- Query: "Việt" (with accent) matches both corpus variants
- No need to generate query variants at search time

**Memory Cost**: 2x index size, still manageable (tens of MB for 100K companies).

### Why Comprehensive Stopword Removal?

**Removed Terms:**
- Legal entity types: CTY, CONG TY, TNHH, CP, MTV, etc.
- Business activities: THUONG MAI, DICH VU, XAY DUNG, etc.
- Location markers: CHI NHANH, VAN PHONG DAI DIEN, etc.
- In both accented and unaccented forms

**Rationale:**
- Focus matching on **brand name**, not legal structure
- "CÔNG TY TNHH ABC" and "ABC" should match perfectly
- Reduces vector space dimensionality

## Text Processing

### Vietnamese Text Normalization

Located in: `src/preprocess.py`

**Key Functions:**

```python
from src.preprocess import normalize_vietnamese_text, clean_company_name

# Step 1: Normalize Unicode
normalized = normalize_vietnamese_text(raw_input)

# Step 2: Clean and remove stopwords
cleaned = clean_company_name(normalized, remove_stopwords=True)
```

**Features:**
- NFC normalization (canonical composition)
- Custom accent removal (preserves Vietnamese character mappings)
- Regex-based stopword removal
- Special character handling

### Accent Handling

**Custom Implementation:**
- Preserves character-by-character mapping
- Handles Vietnamese-specific characters (ă, â, đ, ê, ô, ơ, ư)
- Works for both text generation and matching

**Example:**
```
Input:  "CÔNG TY TNHH SỮA VIỆT NAM"
No-accent: "CONG TY TNHH SUA VIET NAM"
```

## Matching Algorithms

### TF-IDF Char N-gram

**Configuration:**
- n-gram range: (2, 5) - captures character sequences of 2-5 chars
- Sublinear_tf: True - applies logarithmic scaling
- Min_df: 1 - include all terms (useful for small corpora)

**Strengths:**
- Excellent for typos and partial matches
- Fast query time
- No training required

**Weaknesses:**
- Less effective for semantic similarity
- Sensitive to rare character patterns

### BM25

**Configuration:**
- Word-level tokenization
- k1: 1.5 (term frequency saturation)
- b: 0.75 (length normalization)

**Strengths:**
- Better for exact term matching
- Handles document length differences
- Good for keyword-focused queries

**Weaknesses:**
- Less robust to typos and character-level variations
- Requires proper word boundaries

### Hybrid (TF-IDF + BM25)

**Scoring Formula:**
```
hybrid_score = (tfidf_weight * normalized_tfidf) +
               (bm25_weight * normalized_bm25)
```

**Default Weights:**
- tfidf_weight: 0.7
- bm25_weight: 0.3

**Normalization:** Min-max scaling to ensure scores are in [0, 1]

**Advantages:**
- Combines semantic (TF-IDF) and relevance (BM25) signals
- Tunable for different use cases
- Best overall performance

### LSA (Latent Semantic Analysis)

**Purpose:** Dimensionality reduction for large corpora (>100K companies)

**Configuration:**
- n_components: 100 (reduced dimensions)
- Truncated SVD for efficiency

**When to Use:**
- Very large corpora where TF-IDF vectors become high-dimensional
- Memory-constrained environments
- Need for faster similarity computation

**Trade-offs:**
- Loss of some interpretability
- Requires training (SVD fit)
- Slight accuracy reduction (~0.5%)

## Scalability

### Local Processing

**Single Machine:**
- Up to 1M companies: Comfortable (RAM: ~2GB)
- Up to 10M companies: Feasible (RAM: ~20GB)
- Latency: <10ms per query

**Recommendations:**
- Use LSA for >100K companies
- Enable query caching for repeated searches
- Consider approximate nearest neighbors for >1M companies

### Distributed Processing

**Spark/Databricks:**

Located in: `examples/spark_local_example.py`, `examples/databricks_connect_example.py`

**Architecture:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Driver Node    │ <-> │  Worker Nodes   │ <-> │  Shared Storage │
│  (coordinates)  │     |  (process data) │     │  (S3, DBFS, etc)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Use Cases:**
- Processing >10M companies
- Batch evaluation jobs
- Distributed feature extraction
- Real-time serving with Databricks

**Setup:**
```bash
# Local Spark
python examples/spark_local_example.py

# Databricks Connect
python examples/databricks_connect_example.py
```

**Performance:**
- Linear scaling with number of nodes
- Suitable for offline batch processing
- Not recommended for <100K companies (overhead too high)

### Performance Optimization

**Index Building:**
- Batch processing for large corpora
- Parallel preprocessing (multiprocessing)
- Incremental updates for dynamic corpora

**Query Optimization:**
- Dual indexing (accented + no-accent)
- Result caching (LRU cache)
- Top-k optimization (early termination)

**Memory Optimization:**
- Sparse matrix representation (scipy.sparse)
- LSA dimensionality reduction
- Quantization for large-scale deployment

## Error Analysis

Common failure patterns and mitigations:

| Error Type | Example | Mitigation |
|------------|---------|------------|
| Very short queries | "ABC" | Min-length threshold |
| Common brand names | "Samsung" | Return multiple high-scoring matches |
| OCR errors | "Vina mīľk" | Fuzzy matching fallback |
| English variants | "Vietnam Dairy" | Translation dictionary |

## Future Improvements

1. **Active Learning**: Continuously improve from user feedback
2. **Deep Learning**: Explore transformer models for Vietnamese
3. **Entity Linking**: Connect to business registries
4. **Real-time Updates**: Streaming pipeline for new companies

## References

- [scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Vietnamese NLP Resources](https://github.com/magizbox/underthesea)

---

For implementation details, see the [matching guide](matching-guide.md).
For Spark/Databricks setup, see the [Spark guide](spark-databricks.md).
