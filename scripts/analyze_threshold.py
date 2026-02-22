"""
Analyze confidence threshold trade-off for tfidf[sw=F]-rerank(n=5)+bge-m3.

For each query we record the top-1 reranked score and whether the answer was
correct.  We then sweep a threshold t ∈ [0, 1]:
  - Coverage  = fraction of queries where top-1 score >= t  (we answered)
  - Precision = Top-1 accuracy AMONG answered queries
  - F-score   = harmonic-mean of Precision and Coverage (β=0.5: precision-weighted)

The goal: find the smallest t that gives ≥ 95 % / 98 % / 99 % precision at
reasonable coverage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import numpy as np
from src.matching.matcher import CompanyMatcher


def load_eval_data(max_queries: int = 1000):
    corpus, id2name = [], {}
    with open('data/eval/corpus.jsonl', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item['name'])
            id2name[item['id']] = item['name']
    queries = []
    with open('data/eval/queries.jsonl', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    if len(queries) > max_queries:
        import random; random.seed(42)
        queries = random.sample(queries, max_queries)
    return corpus, queries, id2name


def collect_scores(matcher, queries, id2name):
    """Return list of (top1_score, is_correct) for every query."""
    records = []
    for q in queries:
        target = id2name[q['target_id']]
        results = matcher.search(q['text'], top_k=3)   # no min_score here
        if results:
            records.append((results[0]['score'], results[0]['company'] == target))
        else:
            records.append((0.0, False))
    return records


def threshold_table(records, thresholds):
    rows = []
    total = len(records)
    for t in thresholds:
        answered   = [(s, c) for s, c in records if s >= t]
        coverage   = len(answered) / total * 100
        precision  = (sum(c for _, c in answered) / len(answered) * 100) if answered else 0.0
        correct_all = sum(c for _, c in answered)
        # F0.5: weights precision twice as heavily as coverage
        if precision + coverage > 0:
            f05 = (1 + 0.5**2) * precision * coverage / (0.5**2 * precision + coverage)
        else:
            f05 = 0.0
        rows.append((t, coverage, precision, correct_all, len(answered), f05))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense-model', default='BAAI/bge-m3')
    parser.add_argument('--max-queries', type=int, default=1000)
    parser.add_argument('--rerank-n', type=int, default=5)
    args = parser.parse_args()

    print(f"Loading eval data ({args.max_queries} queries)…")
    corpus, queries, id2name = load_eval_data(args.max_queries)
    print(f"  Corpus: {len(corpus)} companies  |  Queries: {len(queries)}")

    print(f"\nBuilding tfidf[sw=F]-rerank(n={args.rerank_n})+{args.dense_model.split('/')[-1]} …")
    matcher = CompanyMatcher(
        model_name='tfidf-dense',
        remove_stopwords=False,
        dense_model_name=args.dense_model,
        fusion='tfidf-rerank',
        rerank_n=args.rerank_n,
    )
    matcher.build_index(corpus)

    print("Collecting scores…")
    t0 = time.time()
    records = collect_scores(matcher, queries, id2name)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/len(queries)*1000:.1f} ms/query avg)")

    # Overall baseline (no threshold)
    overall_top1 = sum(c for _, c in records) / len(records) * 100
    print(f"\nBaseline (no threshold): {overall_top1:.2f}% Top-1  over {len(records)} queries")

    # Score distribution
    scores = [s for s, _ in records]
    print(f"\nScore distribution of top-1 reranked result:")
    for pct in [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{pct:3d} = {np.percentile(scores, pct):.4f}")

    # Threshold sweep
    thresholds = [round(t, 2) for t in np.arange(0.00, 1.01, 0.02)]
    rows = threshold_table(records, thresholds)

    print(f"\n{'Threshold':>10} {'Coverage':>10} {'Precision':>10} {'Correct':>9} {'Answered':>10} {'F0.5':>8}")
    print("-" * 65)
    prev_prec = -1
    highlight_marks = {95.0: "← 95% precision", 98.0: "← 98% precision", 99.0: "← 99% precision"}
    last_mark = {}
    for t, cov, prec, correct, answered, f05 in rows:
        mark = ""
        for target_p, label in highlight_marks.items():
            if prec >= target_p and target_p not in last_mark:
                last_mark[target_p] = True
                mark += f"  {label}"
        print(f"{t:>10.2f} {cov:>9.1f}% {prec:>9.1f}% {correct:>9} {answered:>10} {f05:>8.2f}{mark}")

    # Find recommended thresholds
    print("\n── Recommended thresholds ──────────────────────────────────────")
    for target_prec in [95.0, 98.0, 99.0]:
        for t, cov, prec, correct, answered, f05 in rows:
            if prec >= target_prec:
                print(f"  ≥{target_prec:.0f}% precision: threshold={t:.2f}  →  "
                      f"coverage={cov:.1f}%  ({answered}/{len(records)} queries answered)")
                break
        else:
            print(f"  ≥{target_prec:.0f}% precision: not achievable with any threshold")

    # Best F0.5
    best = max(rows, key=lambda r: r[5])
    print(f"\n  Best F0.5 trade-off: threshold={best[0]:.2f}  "
          f"precision={best[2]:.1f}%  coverage={best[1]:.1f}%  F0.5={best[5]:.2f}")


if __name__ == '__main__':
    main()
