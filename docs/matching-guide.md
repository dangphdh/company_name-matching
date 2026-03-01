# Matching Guide

Complete guide to using the company name matching algorithms.

## Table of Contents

- [Quick Start](#quick-start)
- [Model Selection](#model-selection)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Performance Tuning](#performance-tuning)
- [Examples](#examples)

## Quick Start

### Basic Usage

```python
from src.matching.matcher import CompanyMatcher

# Initialize with default hybrid model
matcher = CompanyMatcher()

# Build index from company names
companies = [
    "CÔNG TY TNHH SỮA VIỆT NAM",
    "Ngân hàng TMCP Ngoại thương Việt Nam",
    "Tập đoàn Hòa Phát"
]
matcher.build_index(companies)

# Search for matches
results = matcher.search("Vinamilk", top_k=5)
for result in results:
    print(f"{result['company']}: {result['score']:.3f}")
```

**Output:**
```
CÔNG TY TNHH SỮA VIỆT NAM: 0.923
```

## Model Selection

### Available Models

| Model | Description | Use Case | Speed |
|-------|-------------|----------|-------|
| `tfidf` | Character n-gram TF-IDF | Typos, abbreviations | Fastest |
| `bm25` | Word-level BM25 | Exact terms | Fast |
| `tfidf-bm25` | Hybrid (recommended) | General purpose | Balanced |
| `lsa` | TF-IDF + SVD | Large corpora | Moderate |
| `wordllama-l2` | WordLlama L2 embeddings | Semantic similarity | Slow |
| `wordllama-l3` | WordLlama L3 embeddings | Semantic similarity | Slow |

### Choosing a Model

**Use `tfidf` (TF-IDF) when:**
- You need fastest possible queries
- Handling typos and character-level variations is important
- Memory is constrained

**Use `bm25` when:**
- Users search with exact terms
- Word boundaries are clear and correct
- You want keyword-focused matching

**Use `tfidf-bm25` (Hybrid) when:**
- You want the best overall performance (recommended)
- You have diverse query patterns
- You need to balance semantic and term-based matching

**Use `lsa` when:**
- Corpus size >100K companies
- Memory is constrained
- You can accept slight accuracy reduction

**Use `wordllama-*` when:**
- Semantic similarity is critical
- You have GPU available
- Latency requirements are relaxed

### Model Initialization

```python
# TF-IDF only
matcher = CompanyMatcher(model_name='tfidf')

# BM25 only
matcher = CompanyMatcher(model_name='bm25')

# Hybrid (default)
matcher = CompanyMatcher(model_name='tfidf-bm25')

# Hybrid with custom weights
matcher = CompanyMatcher(
    model_name='tfidf-bm25',
    tfidf_weight=0.7,
    bm25_weight=0.3
)

# LSA
matcher = CompanyMatcher(
    model_name='lsa',
    lsa_components=100
)

# WordLlama (requires GPU)
matcher = CompanyMatcher(
    model_name='wordllama-l2',
    use_gpu=True
)
```

## API Reference

### CompanyMatcher Class

#### Constructor

```python
CompanyMatcher(
    model_name: str = 'tfidf-bm25',
    remove_stopwords: bool = True,
    use_gpu: bool = False,
    **kwargs
)
```

**Parameters:**
- `model_name`: Model to use (`tfidf`, `bm25`, `tfidf-bm25`, `lsa`, `wordllama-l2`, `wordllama-l3`)
- `remove_stopwords`: Whether to remove Vietnamese company type stopwords
- `use_gpu`: Whether to use GPU (for WordLlama models)
- `**kwargs`: Model-specific parameters
  - `tfidf_weight`: Weight for TF-IDF in hybrid (default: 0.7)
  - `bm25_weight`: Weight for BM25 in hybrid (default: 0.3)
  - `lsa_components`: Number of LSA components (default: 100)
  - `ngram_range`: TF-IDF n-gram range (default: (2, 5))

#### Methods

##### build_index()

```python
build_index(company_names: List[str]) -> None
```

Build the search index from a list of company names.

**Parameters:**
- `company_names`: List of company name strings

**Example:**
```python
companies = ["CÔNG TY TNHH ABC", "Tập đoàn XYZ"]
matcher.build_index(companies)
```

##### search()

```python
search(
    query: str,
    top_k: int = 10,
    return_similar: bool = True
) -> List[Dict[str, Any]]
```

Search for company names matching the query.

**Parameters:**
- `query`: Search query string
- `top_k`: Number of results to return (default: 10)
- `return_similar`: Whether to include similarity scores (default: True)

**Returns:**
- List of dictionaries with keys:
  - `company`: Matched company name
  - `score`: Similarity score (0-1)
  - `id`: Original index (optional)

**Example:**
```python
results = matcher.search("Vinamilk", top_k=5)
# [{'company': 'CÔNG TY TNHH SỮA VIỆT NAM', 'score': 0.923}, ...]
```

##### batch_search()

```python
batch_search(
    queries: List[str],
    top_k: int = 10
) -> List[List[Dict[str, Any]]]
```

Search multiple queries in batch.

**Parameters:**
- `queries`: List of query strings
- `top_k`: Number of results per query

**Returns:**
- List of result lists

**Example:**
```python
queries = ["Vinamilk", "Vietcombank", "Hoa Phat"]
results = matcher.batch_search(queries, top_k=3)
```

## Advanced Usage

### Custom Preprocessing

```python
from src.matching.matcher import CompanyMatcher
from src.preprocess import clean_company_name

# Preprocess companies
companies = ["CÔNG TY TNHH SỮA VIỆT NAM"]
cleaned = [clean_company_name(c, remove_stopwords=True) for c in companies]

# Build index with cleaned names
matcher = CompanyMatcher(remove_stopwords=False)  # Don't clean again
matcher.build_index(cleaned)

# Search with same preprocessing
query = clean_company_name("Vinamilk", remove_stopwords=True)
results = matcher.search(query)
```

### Working with Metadata

```python
from src.matching.matcher import CompanyMatcher

# Companies with metadata
companies_metadata = [
    {"name": "CÔNG TY TNHH SỮA VIỆT NAM", "id": "001", "tax_id": "123456"},
    {"name": "Ngân hàng TMCP Ngoại thương", "id": "002", "tax_id": "789012"},
]

# Extract names for indexing
names = [c["name"] for c in companies_metadata]
matcher.build_index(names)

# Search and retrieve metadata
query = "Vinamilk"
results = matcher.search(query, top_k=1)

if results:
    matched_id = results[0]['id']  # Index in original list
    metadata = companies_metadata[matched_id]
    print(f"Found: {metadata['name']}, Tax ID: {metadata['tax_id']}")
```

### Threshold Filtering

```python
# Search with minimum score threshold
results = matcher.search("Vinamilk", top_k=10)

# Filter by threshold
MIN_SCORE = 0.7
filtered_results = [r for r in results if r['score'] >= MIN_SCORE]

if not filtered_results:
    print("No matches found above threshold")
```

### Fuzzy Matching Fallback

```python
from src.matching.matcher import CompanyMatcher
from rapidfuzz import process, fuzz

matcher = CompanyMatcher()
matcher.build_index(companies)

query = "Vinaamilk"  # Typo

# Try TF-IDF first
results = matcher.search(query, top_k=1)

# If score is low, fallback to fuzzy matching
if not results or results[0]['score'] < 0.5:
    print("TF-IDF score low, trying fuzzy matching...")
    fuzzy_result = process.extractOne(query, companies, scorer=fuzz.WRatio)
    if fuzzy_result and fuzzy_result[1] > 80:
        print(f"Fuzzy match: {fuzzy_result[0]} (score: {fuzzy_result[1]})")
```

## Performance Tuning

### Index Building

**For small corpora (<10K companies):**
```python
matcher = CompanyMatcher(model_name='tfidf-bm25')
matcher.build_index(companies)  # Fast (<1 second)
```

**For medium corpora (10K-100K):**
```python
matcher = CompanyMatcher(model_name='tfidf-bm25')
matcher.build_index(companies)  # Moderate (1-10 seconds)
```

**For large corpora (>100K):**
```python
# Use LSA for dimensionality reduction
matcher = CompanyMatcher(
    model_name='lsa',
    lsa_components=100
)
matcher.build_index(companies)  # Slower (10-60 seconds)
```

### Query Performance

**Optimize for speed:**
```python
# Use TF-IDF only (fastest)
matcher = CompanyMatcher(model_name='tfidf')
```

**Optimize for accuracy:**
```python
# Use hybrid with balanced weights
matcher = CompanyMatcher(
    model_name='tfidf-bm25',
    tfidf_weight=0.6,
    bm25_weight=0.4
)
```

**Optimize for memory:**
```python
# Use LSA to reduce memory footprint
matcher = CompanyMatcher(
    model_name='lsa',
    lsa_components=50  # Fewer components = less memory
)
```

### Caching

```python
from functools import lru_cache

matcher = CompanyMatcher()
matcher.build_index(companies)

# Cache frequent queries
@lru_cache(maxsize=1000)
def cached_search(query: str):
    return matcher.search(query, top_k=5)

# Use cached search
results = cached_search("Vinamilk")
```

## Examples

### Example 1: Simple Name Matching

```python
from src.matching.matcher import CompanyMatcher

# Setup
companies = [
    "CÔNG TY TNHH SỮA VIỆT NAM",
    "CÔNG TY CỔ PHẦN SUA VIET NAM",
    "VIETNAM DAIRY PRODUCTS JSC"
]

matcher = CompanyMatcher()
matcher.build_index(companies)

# Search
results = matcher.search("Vinamilk", top_k=3)
for r in results:
    print(f"{r['company']}: {r['score']:.3f}")
```

### Example 2: Handling Variants

```python
# Query variants all match the same company
queries = [
    "Vinamilk",           # Brand name
    "Sữa Việt Nam",       # Vietnamese translation
    "Vietnam Dairy",      # English translation
    "Cty Sữa VN",         # Abbreviation
]

matcher = CompanyMatcher()
matcher.build_index(["CÔNG TY TNHH SỮA VIỆT NAM"])

for query in queries:
    results = matcher.search(query, top_k=1)
    if results:
        print(f"'{query}' -> {results[0]['score']:.3f}")
```

### Example 3: Batch Processing

```python
from src.matching.matcher import CompanyMatcher

# Load companies from file
with open('data/sample_system_names.txt', 'r') as f:
    companies = [line.strip() for line in f]

# Build index
matcher = CompanyMatcher()
matcher.build_index(companies)

# Process multiple queries
queries = ["Vinamilk", "Vietcombank", "Hoa Phat"]
all_results = matcher.batch_search(queries, top_k=3)

# Display results
for query, results in zip(queries, all_results):
    print(f"\nQuery: {query}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['company']} ({r['score']:.3f})")
```

### Example 4: Error Handling

```python
from src.matching.matcher import CompanyMatcher

matcher = CompanyMatcher()

# Handle empty query
try:
    results = matcher.search("", top_k=5)
except ValueError as e:
    print(f"Error: {e}")

# Handle query before building index
try:
    results = matcher.search("Vinamilk")
except RuntimeError as e:
    print(f"Error: {e}")
    # Build index first
    matcher.build_index(companies)
    results = matcher.search("Vinamilk")
```

### Example 5: Custom Scoring

```python
from src.matching.matcher import CompanyMatcher

matcher = CompanyMatcher()
matcher.build_index(companies)

# Get raw scores
results = matcher.search("Vinamilk", top_k=10)

# Apply custom business logic
def rerank(results, query):
    """Rerank results based on business rules."""
    reranked = []

    for r in results:
        score = r['score']
        company = r['company']

        # Boost exact matches
        if query.lower() in company.lower():
            score *= 1.2

        # Boost companies with certain keywords
        if "TẬP ĐOÀN" in company:
            score *= 1.1

        reranked.append({**r, 'score': min(score, 1.0)})

    # Re-sort by modified score
    reranked.sort(key=lambda x: x['score'], reverse=True)
    return reranked

# Apply custom reranking
final_results = rerank(results, "Vinamilk")
```

## Troubleshooting

### Low Match Scores

**Problem:** All scores are low (<0.5)

**Solutions:**
1. Check if stopwords are being removed correctly
2. Verify Unicode normalization is applied
3. Try different models (e.g., `tfidf-bm25` instead of `tfidf`)
4. Ensure both accented and no-accent variants are indexed

### Slow Index Building

**Problem:** Index building takes too long

**Solutions:**
1. Use LSA for large corpora: `CompanyMatcher(model_name='lsa')`
2. Reduce n-gram range: `ngram_range=(2, 4)`
3. Batch process in chunks
4. Use Spark/Databricks for distributed processing

### Memory Issues

**Problem:** Out of memory errors with large corpora

**Solutions:**
1. Reduce LSA components: `lsa_components=50`
2. Use sparse matrices (default with TF-IDF)
3. Process in batches
4. Use approximate nearest neighbor algorithms

---

For architecture details, see [architecture.md](architecture.md).
For Spark/Databricks setup, see [spark-databricks.md](spark-databricks.md).
