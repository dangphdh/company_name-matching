"""
Test hybrid TF-IDF + BM25 matcher and compare performance
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from src.matching.matcher import CompanyMatcher

def load_eval_data():
    """Load evaluation dataset"""
    corpus = []
    with open('data/eval/corpus.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line)['name'])
    
    queries = []
    with open('data/eval/queries.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    
    return corpus, queries

def evaluate_matcher(model_name, corpus, queries, **kwargs):
    """Evaluate a matcher model"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    matcher = CompanyMatcher(model_name=model_name, **kwargs)
    matcher.build_index(corpus)
    
    top1_correct = 0
    top3_correct = 0
    total_queries = len(queries)
    latencies = []
    
    for query in queries:
        query_text = query['text']
        target_id = query['target_id']
        
        start = time.time()
        results = matcher.search(query_text, top_k=3)
        elapsed = time.time() - start
        latencies.append(elapsed)
        
        # Check if correct company is in results
        for rank, result in enumerate(results, 1):
            result_id = None
            # Try to find matching company in corpus to get ID
            for idx, corp_name in enumerate(corpus):
                if corp_name == result['company']:
                    result_id = idx
                    break
            
            if result_id == target_id:
                if rank == 1:
                    top1_correct += 1
                if rank <= 3:
                    top3_correct += 1
                break
    
    top1_accuracy = (top1_correct / total_queries * 100) if total_queries > 0 else 0
    top3_accuracy = (top3_correct / total_queries * 100) if total_queries > 0 else 0
    avg_latency = np.mean(latencies) * 1000  # Convert to ms
    
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total_queries})")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}% ({top3_correct}/{total_queries})")
    print(f"Average Latency: {avg_latency:.2f}ms")
    
    return {
        'model': model_name,
        'top1_accuracy': top1_accuracy,
        'top3_accuracy': top3_accuracy,
        'avg_latency': avg_latency
    }

if __name__ == '__main__':
    try:
        corpus, queries = load_eval_data()
        print(f"Loaded {len(corpus)} companies and {len(queries)} queries")
        
        results = []
        
        # Test TF-IDF baseline
        results.append(evaluate_matcher('tfidf', corpus, queries))
        
        # Test BM25 baseline
        results.append(evaluate_matcher('bm25', corpus, queries))
        
        # Test Hybrid with equal weights
        results.append(evaluate_matcher('tfidf-bm25', corpus, queries, tfidf_weight=0.5, bm25_weight=0.5))
        
        # Test Hybrid with TF-IDF emphasis
        results.append(evaluate_matcher('tfidf-bm25', corpus, queries, tfidf_weight=0.7, bm25_weight=0.3))
        
        # Test Hybrid with BM25 emphasis
        results.append(evaluate_matcher('tfidf-bm25', corpus, queries, tfidf_weight=0.3, bm25_weight=0.7))
        
        # Print comparison table
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<25} {'Top-1 Acc':<12} {'Top-3 Acc':<12} {'Avg Latency (ms)':<16}")
        print(f"{'-'*80}")
        
        for r in results:
            print(f"{r['model']:<25} {r['top1_accuracy']:<11.2f}% {r['top3_accuracy']:<11.2f}% {r['avg_latency']:<15.2f}")
        
        # Find best model
        best = max(results, key=lambda x: x['top1_accuracy'])
        print(f"\nBest Top-1 Accuracy: {best['model']} ({best['top1_accuracy']:.2f}%)")
        
    except FileNotFoundError:
        print("Error: Could not find evaluation data.")
        print("Please run: python scripts/generate_eval_dataset.py")
