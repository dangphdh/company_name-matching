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

def load_eval_data(max_queries=1000):
    """Load evaluation dataset"""
    corpus = []
    corp_id_to_name = {}
    with open('data/eval/corpus.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item['name'])
            corp_id_to_name[item['id']] = item['name']
    
    queries = []
    with open('data/eval/queries.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    
    if len(queries) > max_queries:
        import random
        random.seed(42)
        queries = random.sample(queries, max_queries)
    
    return corpus, queries, corp_id_to_name

def evaluate_matcher(model_name, corpus, queries, corp_id_to_name, label=None, **kwargs):
    """Evaluate a matcher model"""
    display_name = label or model_name
    print(f"\n{'='*60}")
    print(f"Evaluating: {display_name}")
    print(f"{'='*60}")
    
    matcher = CompanyMatcher(model_name=model_name, **kwargs)
    matcher.build_index(corpus)
    
    top1_correct = 0
    top3_correct = 0
    total_queries = len(queries)
    latencies = []
    
    for query in queries:
        query_text = query['text']
        target_name = corp_id_to_name[query['target_id']]
        
        start = time.time()
        results = matcher.search(query_text, top_k=3)
        elapsed = time.time() - start
        latencies.append(elapsed)
        
        # Check if correct company is in results
        predicted_names = [r['company'] for r in results]
        
        if predicted_names and predicted_names[0] == target_name:
            top1_correct += 1
        if target_name in predicted_names[:3]:
            top3_correct += 1
    
    top1_accuracy = (top1_correct / total_queries * 100) if total_queries > 0 else 0
    top3_accuracy = (top3_correct / total_queries * 100) if total_queries > 0 else 0
    avg_latency = np.mean(latencies) * 1000  # Convert to ms
    
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total_queries})")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}% ({top3_correct}/{total_queries})")
    print(f"Average Latency: {avg_latency:.2f}ms")
    
    return {
        'model': display_name,
        'top1_accuracy': top1_accuracy,
        'top3_accuracy': top3_accuracy,
        'avg_latency': avg_latency
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense-model', default='BAAI/bge-m3',
                        help='SentenceTransformer model for sparse-dense hybrids')
    parser.add_argument('--skip-dense', action='store_true',
                        help='Skip sparse-dense hybrid tests (faster)')
    args = parser.parse_args()

    try:
        corpus, queries, corp_id_to_name = load_eval_data()
        print(f"Loaded {len(corpus)} companies and {len(queries)} queries")
        
        results = []
        
        # ── Sparse baselines ─────────────────────────────────────────────────
        # New preprocessing: entity-type normalization + cp/tnhh/mtv kept as signal
        results.append(evaluate_matcher('tfidf', corpus, queries, corp_id_to_name,
                                        label='tfidf[entity-norm,sw=T]'))
        # Without stopwords: preserves all tokens (previous best)
        results.append(evaluate_matcher('tfidf', corpus, queries, corp_id_to_name,
                                        label='tfidf[sw=F]',
                                        remove_stopwords=False))
        results.append(evaluate_matcher('bm25',  corpus, queries, corp_id_to_name,
                                        label='bm25[entity-norm,sw=T]'))

        # ── Sparse-sparse hybrid (TF-IDF + BM25) ─────────────────────────────
        results.append(evaluate_matcher('tfidf-bm25', corpus, queries, corp_id_to_name,
                                        label='tfidf+bm25(0.5/0.5)',
                                        tfidf_weight=0.5, bm25_weight=0.5))
        results.append(evaluate_matcher('tfidf-bm25', corpus, queries, corp_id_to_name,
                                        label='tfidf+bm25(0.7/0.3)',
                                        tfidf_weight=0.7, bm25_weight=0.3))

        # ── Sparse-dense hybrids ──────────────────────────────────────────────
        if not args.skip_dense:
            dm = args.dense_model
            short = dm.split('/')[-1]  # e.g. bge-m3

            # TF-IDF + dense, weighted fusion
            results.append(evaluate_matcher('tfidf-dense', corpus, queries, corp_id_to_name,
                                            label=f'tfidf+{short}(w,0.5/0.5)',
                                            dense_model_name=dm,
                                            sparse_weight=0.5, dense_weight=0.5,
                                            fusion='weighted'))
            results.append(evaluate_matcher('tfidf-dense', corpus, queries, corp_id_to_name,
                                            label=f'tfidf+{short}(w,0.7/0.3)',
                                            dense_model_name=dm,
                                            sparse_weight=0.7, dense_weight=0.3,
                                            fusion='weighted'))
            # TF-IDF + dense, RRF fusion
            results.append(evaluate_matcher('tfidf-dense', corpus, queries, corp_id_to_name,
                                            label=f'tfidf+{short}(rrf)',
                                            dense_model_name=dm,
                                            fusion='rrf'))

            # ── Reranking: TF-IDF retrieves N candidates, dense reranks ──────
            for n in [5, 10, 20]:
                results.append(evaluate_matcher('tfidf-dense', corpus, queries, corp_id_to_name,
                                                label=f'tfidf-rerank(n={n})+{short}',
                                                dense_model_name=dm,
                                                fusion='tfidf-rerank',
                                                rerank_n=n))

            # ── Adaptive rerank: dense only when TF-IDF score gap is ambiguous ─
            for thresh in [0.02, 0.05, 0.10]:
                results.append(evaluate_matcher('tfidf-dense', corpus, queries, corp_id_to_name,
                                                label=f'adaptive-rerank(t={thresh})+{short}',
                                                dense_model_name=dm,
                                                fusion='adaptive-rerank',
                                                rerank_n=5,
                                                rerank_threshold=thresh))

            # ── Combined: tfidf[sw=F]+entity-norm as retriever → BGE-M3 rerank ─
            # sw=False keeps all tokens (88.8% standalone) → denser candidate pool for reranker
            for n in [5, 10]:
                results.append(evaluate_matcher('tfidf-dense', corpus, queries, corp_id_to_name,
                                                label=f'tfidf[sw=F]-rerank(n={n})+{short}',
                                                dense_model_name=dm,
                                                fusion='tfidf-rerank',
                                                rerank_n=n,
                                                remove_stopwords=False))
            results.append(evaluate_matcher('tfidf-dense', corpus, queries, corp_id_to_name,
                                            label=f'tfidf[sw=F]-adaptive(t=0.10)+{short}',
                                            dense_model_name=dm,
                                            fusion='adaptive-rerank',
                                            rerank_n=5,
                                            rerank_threshold=0.10,
                                            remove_stopwords=False))

            # BM25 + dense, weighted fusion
            results.append(evaluate_matcher('bm25-dense', corpus, queries, corp_id_to_name,
                                            label=f'bm25+{short}(w,0.5/0.5)',
                                            dense_model_name=dm,
                                            sparse_weight=0.5, dense_weight=0.5,
                                            fusion='weighted'))
            # BM25 + dense, RRF fusion
            results.append(evaluate_matcher('bm25-dense', corpus, queries, corp_id_to_name,
                                            label=f'bm25+{short}(rrf)',
                                            dense_model_name=dm,
                                            fusion='rrf'))
        
        # ── Print comparison table ────────────────────────────────────────────
        print(f"\n{'='*90}")
        print("COMPARISON SUMMARY")
        print(f"{'='*90}")
        print(f"{'Model':<35} {'Top-1 Acc':>10} {'Top-3 Acc':>10} {'Avg Latency':>14}")
        print(f"{'-'*90}")
        
        for r in results:
            print(f"{r['model']:<35} {r['top1_accuracy']:>9.2f}% {r['top3_accuracy']:>9.2f}% {r['avg_latency']:>12.2f} ms")
        
        best = max(results, key=lambda x: x['top1_accuracy'])
        print(f"\nBest Top-1: {best['model']}  →  {best['top1_accuracy']:.2f}%")
        
    except FileNotFoundError:
        print("Error: Could not find evaluation data.")
        print("Please run: python scripts/generate_eval_dataset.py")

