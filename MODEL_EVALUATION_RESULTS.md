# Vietnamese Company Name Matching - Model Evaluation Results

## Overview
Comprehensive evaluation of multiple models for Vietnamese company name matching using a dataset of 4,019 companies and 1,000 test queries. The evaluation measures Top-1 and Top-3 accuracy, along with average latency per query.

## Dataset Information
- **Corpus Size**: 4,019 companies (with dual indexing: accented + unaccented variants)
- **Test Queries**: 1,000 synthetic queries (combinatorial variations)
- **Evaluation Metrics**: Top-1 Accuracy, Top-3 Accuracy, Average Latency

## Model Performance Summary

### Complete Results Table

| Model | Stopwords | Top-1 Accuracy | Top-3 Accuracy | Avg Latency | Notes |
|-------|-----------|----------------|----------------|-------------|---------|
| **TF-IDF** | **False** | **76.10%** | **93.10%** | **12.34 ms** | **Best overall performance** |
| **TF-IDF** | **True** | **66.20%** | **92.70%** | **7.21 ms** | Worse with stopwords removed |
| **BM25** | **False** | **67.40%** | **85.60%** | **7.39 ms** | **Strong speed-accuracy balance** |
| **BM25** | **True** | **65.50%** | **92.20%** | **3.70 ms** | Fastest, good Top-3 |
| **BGE-M3** | **False** | **72.00%** | **89.10%** | **216.15 ms** | Best semantic understanding |
| **BGE-M3** | **True** | **64.00%** | **89.80%** | **195.62 ms** | Worse with stopwords removed |
| **Vietnamese SBERT** | **False** | **61.70%** | **77.80%** | **148.23 ms** | Poor performance |
| **Vietnamese SBERT** | **True** | **64.10%** | **89.40%** | **151.05 ms** | Improved with stopwords removed |
| **WordLlama L2** | **False** | **61.10%** | **79.80%** | **7.97 ms** | Poor performance |
| **WordLlama L2** | **True** | **66.70%** | **92.60%** | **9.06 ms** | Good baseline performance |

## Detailed Analysis

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
**Best performing model overall**

**Configuration:**
- Char-level n-grams (2-5)
- Sublinear TF scaling
- Cosine similarity

**Key Findings:**
- **Stopwords=False**: 76.1% Top-1, 93.1% Top-3
- **Stopwords=True**: 66.2% Top-1, 92.7% Top-3
- **Insight**: Keeping stopwords (company type terms) significantly improves accuracy
- **Latency**: 7-12ms, very fast for production use

**Error Patterns:**
- Struggles with highly abbreviated combinatorial variants
- Company type confusion (TNHH vs CP)
- Similar brand names

### 2. BM25 (Best Matching 25)
**Excellent speed-accuracy balance**

**Configuration:**
- BM25Okapi implementation
- Word-level tokenization
- Probabilistic ranking

**Key Findings:**
- **Stopwords=False**: 67.4% Top-1, 85.6% Top-3
- **Stopwords=True**: 65.5% Top-1, 92.2% Top-3
- **Latency**: 3.7-7.4ms, extremely fast
- **Advantage**: More robust to term frequency variations than TF-IDF

### 3. BGE-M3 (BAAI General Embedding)
**Best semantic understanding**

**Configuration:**
- Multilingual embedding model
- 2.27GB model size
- Cosine similarity on embeddings

**Key Findings:**
- **Stopwords=False**: 72.0% Top-1, 89.1% Top-3
- **Stopwords=True**: 64.0% Top-1, 89.8% Top-3
- **Latency**: ~200ms, slowest but most semantically aware
- **Advantage**: Better understanding of contextual relationships

### 4. Vietnamese SBERT (Sentence-BERT)
**Vietnamese-optimized embeddings**

**Configuration:**
- keepitreal/vietnamese-sbert model
- 540MB model size
- Cosine similarity

**Key Findings:**
- **Stopwords=False**: 61.7% Top-1, 77.8% Top-3 (poor)
- **Stopwords=True**: 64.1% Top-1, 89.4% Top-3
- **Latency**: ~150ms
- **Advantage**: Optimized for Vietnamese language patterns

### 5. WordLlama L2
**Fast baseline model**

**Configuration:**
- Word-level embeddings
- Lightweight model
- Cosine similarity

**Key Findings:**
- **Stopwords=False**: 61.1% Top-1, 79.8% Top-3 (poor)
- **Stopwords=True**: 66.7% Top-1, 92.6% Top-3
- **Latency**: 8-9ms, very fast
- **Use case**: Good for quick prototyping

## Performance Rankings

### By Top-1 Accuracy
1. **TF-IDF (stopwords=False)**: 76.10% ⭐ **BEST**
2. **BGE-M3 (stopwords=False)**: 72.00%
3. **BM25 (stopwords=False)**: 67.40%
4. **WordLlama (stopwords=True)**: 66.70%
5. **TF-IDF (stopwords=True)**: 66.20%
6. **BM25 (stopwords=True)**: 65.50%
7. **Vietnamese SBERT (stopwords=True)**: 64.10%
8. **BGE-M3 (stopwords=True)**: 64.00%
9. **Vietnamese SBERT (stopwords=False)**: 61.70%
10. **WordLlama (stopwords=False)**: 61.10%

### By Speed (Latency)
1. **BM25 (stopwords=True)**: 3.70 ms ⚡ **FASTEST**
2. **TF-IDF (stopwords=True)**: 7.21 ms
3. **WordLlama (stopwords=False)**: 7.97 ms
4. **BM25 (stopwords=False)**: 7.39 ms
5. **WordLlama (stopwords=True)**: 9.06 ms
6. **TF-IDF (stopwords=False)**: 12.34 ms
7. **Vietnamese SBERT**: ~150 ms
8. **BGE-M3**: ~200 ms

## Key Insights

### Stopword Impact
- **TF-IDF**: Keeping stopwords improves accuracy by ~10%
- **BM25**: Minimal impact, slight preference for keeping stopwords
- **Embedding models**: Mixed results, generally better with stopwords removed

### Error Analysis
**Common failure patterns across all models:**
1. **Highly abbreviated variants**: "TNHH MTV SX DV" → confused with similar companies
2. **Company type confusion**: TNHH vs CP vs MTV misclassification
3. **Similar brand names**: Companies with identical core names but different types
4. **Branch/office confusion**: Main company vs branch vs representative office

### Model Characteristics

| Aspect | TF-IDF | BM25 | BGE-M3 | Vietnamese SBERT | WordLlama |
|--------|--------|------|--------|------------------|-----------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Semantic Understanding** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Memory Usage** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Complexity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Recommendations

### For Production Use
1. **Primary Choice**: **TF-IDF with stopwords=False**
   - Best accuracy (76.1%)
   - Reasonable speed (12.34ms)
   - Simple deployment

2. **Alternative**: **BM25 with stopwords=False**
   - Good accuracy (67.4%)
   - Very fast (7.39ms)
   - Robust to term variations

### For Development/Research
- **BGE-M3**: When semantic understanding is critical
- **Vietnamese SBERT**: For Vietnamese-specific optimizations
- **WordLlama**: Fast baseline for experimentation

### For High-Volume Applications
- **BM25**: Best speed-accuracy trade-off
- **TF-IDF**: If accuracy is paramount

## Technical Implementation

### Dependencies Added
```
sentence-transformers
rank-bm25
wordllama
```

### Model Configurations
- **TF-IDF**: Char n-grams (2-5), sublinear TF, min_df=1
- **BM25**: BM25Okapi with word tokenization
- **Embeddings**: Cosine similarity on dense vectors
- **Dual Indexing**: All models use accented + unaccented variants

### Evaluation Setup
- 4,019 company corpus
- 1,000 synthetic test queries
- Top-1 and Top-3 accuracy metrics
- Average latency per query

## Conclusion

The evaluation demonstrates that **TF-IDF with preserved stopwords** achieves the best overall performance for Vietnamese company name matching. However, **BM25** offers an excellent alternative with better speed characteristics. The choice depends on specific use case requirements for accuracy vs. speed trade-offs.

**Final Recommendation**: Use **TF-IDF (stopwords=False)** for production systems requiring maximum accuracy, and **BM25 (stopwords=False)** for high-throughput applications.

---
*Evaluation conducted on February 5, 2026*
*Dataset: 4,019 companies, 1,000 test queries*
*Models tested: TF-IDF, BM25, BGE-M3, Vietnamese SBERT, WordLlama*</content>
<parameter name="filePath">/media/Mydisk/Dang/Project/company_name-matching/MODEL_EVALUATION_RESULTS.md