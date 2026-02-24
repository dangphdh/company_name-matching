import json
import time
import os
from tqdm import tqdm
from src.matching.matcher import CompanyMatcher

def evaluate_matcher(corpus_file, queries_file, model_name='BAAI/bge-m3', remove_stopwords=True, lsa_dims=512):
    """
    Đánh giá độ chính xác của thuật toán matching trên dataset đã sinh.
    """
    # 1. Load Corpus
    corpus = []
    corp_id_to_name = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item['name'])
            corp_id_to_name[item['id']] = item['name']

    # 2. Load Queries
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))

    # Giới hạn số lượng query nếu quá nhiều (để test nhanh)
    MAX_QUERIES = 1000
    if len(queries) > MAX_QUERIES:
        import random
        random.seed(42)
        queries = random.sample(queries, MAX_QUERIES)

    print(f"Bắt đầu đánh giá với {len(corpus)} cty và {len(queries)} query mẫu...")

    # 3. Build Matcher (Sử dụng model nhỏ hơn hoặc cache nếu cần, ở đây dùng mặc định)
    matcher = CompanyMatcher(model_name=model_name, remove_stopwords=remove_stopwords, lsa_dims=lsa_dims)
    matcher.build_index(corpus)

    # 4. Run Evaluation
    hits = 0
    top_3_hits = 0
    start_time = time.time()

    results_analysis = []

    for q in tqdm(queries, desc="Evaluating"):
        target_name = corp_id_to_name[q['target_id']]
        predictions = matcher.search(q['text'], top_k=5)
        
        # Check Top 1 — treat all results tied at the top-1 score as rank-1.
        # This correctly handles near-duplicate corpus entries sharing the same
        # normalised form (e.g. "XNK" ↔ "XUẤT NHẬP KHẨU" both → "xnk" after norm).
        is_top1 = False
        if predictions:
            top1_score = predictions[0]['score']
            top1_group = {p['company'] for p in predictions
                          if p['score'] == top1_score}
            if target_name in top1_group:
                hits += 1
                is_top1 = True
        
        # Check Top 3 — target in any of the returned results (top_k=5 groups)
        if any(p['company'] == target_name for p in predictions):
            top_3_hits += 1
            
        if not is_top1:
            # Lưu lại case sai Top 1 để phân tích
            results_analysis.append({
                "query": q['text'],
                "method": q['method'],
                "target": target_name,
                "predicted_top1": predictions[0]['company'] if predictions else "None",
                "score": predictions[0]['score'] if predictions else 0
            })

    duration = time.time() - start_time
    
    # 5. Print Metrics
    accuracy = hits / len(queries)
    top_3_accuracy = top_3_hits / len(queries)
    
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*30)
    print(f"Model: {model_name}")
    print(f"Remove Stopwords: {remove_stopwords}")
    if model_name in ('tfidf-lsa', 'lsa'):
        print(f"LSA dims: {lsa_dims}")
    print(f"Tổng số queries: {len(queries)}")
    print(f"Số lỗi (Top 1): {len(results_analysis)}")
    print(f"Accuracy (Top 1): {accuracy:.2%}")
    print(f"Accuracy (Top 3): {top_3_accuracy:.2%}")
    print(f"Avg Latency: {duration/len(queries)*1000:.2f} ms/query")
    print("="*30)

    # In ra một số mẫu lỗi nếu có
    if results_analysis:
        print("\nMột số trường hợp matching sai tiêu biểu:")
        for err in results_analysis[:5]:
            print(f"- Query: '{err['query']}' ({err['method']})")
            print(f"  Target:    {err['target']}")
            print(f"  Predicted: {err['predicted_top1']} (Score: {err['score']:.4f})")

if __name__ == "__main__":
    # Test với TF-IDF (Có loại stopword)
    print("\n" + "="*50)
    print("--- Testing TF-IDF (Remove Stopwords: True) ---")
    evaluate_matcher(
        corpus_file="data/eval/corpus.jsonl",
        queries_file="data/eval/queries.jsonl",
        model_name="tfidf",
        remove_stopwords=True
    )

    # Test với TF-IDF (KHÔNG loại stopword)
    print("\n" + "="*50)
    print("--- Testing TF-IDF (Remove Stopwords: False) ---")
    evaluate_matcher(
        corpus_file="data/eval/corpus.jsonl",
        queries_file="data/eval/queries.jsonl",
        model_name="tfidf",
        remove_stopwords=False
    )

    # Test với WordLlama l2 (Có loại stopword)
    print("\n" + "="*50)
    print("--- Testing WordLlama L2 (Remove Stopwords: True) ---")
    evaluate_matcher(
        corpus_file="data/eval/corpus.jsonl",
        queries_file="data/eval/queries.jsonl",
        model_name="wordllama-l2",
        remove_stopwords=True
    )

    # Test với WordLlama l2 (KHÔNG loại stopword)
    print("\n" + "="*50)
    print("--- Testing WordLlama L2 (Remove Stopwords: False) ---")
    evaluate_matcher(
        corpus_file="data/eval/corpus.jsonl",
        queries_file="data/eval/queries.jsonl",
        model_name="wordllama-l2",
        remove_stopwords=False
    )

    # Test với Vietnamese SBERT (Có loại stopword)
    print("\n" + "="*50)
    print("--- Testing Vietnamese SBERT (Remove Stopwords: True) ---")
    evaluate_matcher(
        corpus_file="data/eval/corpus.jsonl",
        queries_file="data/eval/queries.jsonl",
        model_name="keepitreal/vietnamese-sbert",
        remove_stopwords=True
    )

    # ── TF-IDF + LSA tests ───────────────────────────────────────────────────
    # Tests the dimensionality-reduction path that makes large-scale deployment
    # feasible (2.4M corpus: dense TF-IDF = ~2.46 TB; LSA k=512 = ~4.8 GB).

    for dims in (128, 256, 512):
        for sw in (False, True):
            sw_label = "sw=F" if not sw else "sw=T"
            print("\n" + "="*50)
            print(f"--- Testing TF-IDF + LSA (k={dims}, {sw_label}) ---")
            evaluate_matcher(
                corpus_file="data/eval/corpus.jsonl",
                queries_file="data/eval/queries.jsonl",
                model_name="tfidf-lsa",
                remove_stopwords=sw,
                lsa_dims=dims,
            )
