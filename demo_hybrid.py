"""
Simple demo of hybrid TF-IDF + BM25 matcher
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching.matcher import CompanyMatcher

# Test data
companies = [
    "Công ty Cổ phần Sữa Việt Nam",
    "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam",  
    "Tập đoàn Viễn thông Quân đội",
    "Công ty TNHH Tư vấn Giáo dục",
    "Công ty Cổ phần Công nghệ Thông tin ABC",
]

# Demo queries
queries = [
    "Vinamilk",
    "Vietcombank",
    "Viettel",
    "ABC Consulting",
    "ABC Technology"
]

print("=" * 70)
print("HYBRID TF-IDF + BM25 MATCHER DEMO")
print("=" * 70)

# Test TF-IDF baseline
print("\n1. TF-IDF Matcher (Baseline)")
print("-" * 70)
tfidf_matcher = CompanyMatcher(model_name='tfidf')
tfidf_matcher.build_index(companies)

for query in queries:
    results = tfidf_matcher.search(query, top_k=2)
    print(f"Query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['company']} (score: {r['score']:.4f})")

# Test BM25 baseline  
print("\n2. BM25 Matcher (Baseline)")
print("-" * 70)
bm25_matcher = CompanyMatcher(model_name='bm25')
bm25_matcher.build_index(companies)

for query in queries:
    results = bm25_matcher.search(query, top_k=2)
    print(f"Query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['company']} (score: {r['score']:.4f})")

# Test Hybrid (equal weights)
print("\n3. Hybrid TF-IDF + BM25 (50/50 weights)")
print("-" * 70)
hybrid_matcher = CompanyMatcher(model_name='tfidf-bm25', tfidf_weight=0.5, bm25_weight=0.5)
hybrid_matcher.build_index(companies)

for query in queries:
    results = hybrid_matcher.search(query, top_k=2)
    print(f"Query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['company']} (score: {r['score']:.4f})")

# Test Hybrid with TF-IDF emphasis
print("\n4. Hybrid TF-IDF + BM25 (70/30 weights - TF-IDF emphasis)")
print("-" * 70)
hybrid_tfidf_matcher = CompanyMatcher(model_name='tfidf-bm25', tfidf_weight=0.7, bm25_weight=0.3)
hybrid_tfidf_matcher.build_index(companies)

for query in queries:
    results = hybrid_tfidf_matcher.search(query, top_k=2)
    print(f"Query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['company']} (score: {r['score']:.4f})")

# Test Hybrid with BM25 emphasis
print("\n5. Hybrid TF-IDF + BM25 (30/70 weights - BM25 emphasis)")
print("-" * 70)
hybrid_bm25_matcher = CompanyMatcher(model_name='tfidf-bm25', tfidf_weight=0.3, bm25_weight=0.7)
hybrid_bm25_matcher.build_index(companies)

for query in queries:
    results = hybrid_bm25_matcher.search(query, top_k=2)
    print(f"Query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['company']} (score: {r['score']:.4f})")

print("\n" + "=" * 70)
print("HYBRID MODEL BENEFITS:")
print("-" * 70)
print("✓ TF-IDF: Good at semantic similarity via character n-grams")
print("✓ BM25: Good at term importance and frequency weighting")
print("✓ Hybrid: Combines both strengths for better overall matching")
print("✓ Tunable weights: Adjust (tfidf_weight, bm25_weight) for your use case")
print("=" * 70)
