import json
import time
import os
from tqdm import tqdm
from src.matching.matcher import CompanyMatcher

def evaluate_matcher(corpus_file, queries_file, model_name='BAAI/bge-m3'):
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
    matcher = CompanyMatcher(model_name=model_name)
    matcher.build_index(corpus)

    # 4. Run Evaluation
    hits = 0
    top_3_hits = 0
    start_time = time.time()

    results_analysis = []

    for q in tqdm(queries, desc="Evaluating"):
        target_name = corp_id_to_name[q['target_id']]
        predictions = matcher.search(q['text'], top_k=5)
        
        # Check Top 1
        is_top1 = False
        if predictions and predictions[0]['company'] == target_name:
            hits += 1
            is_top1 = True
        
        # Check Top 3
        if any(p['company'] == target_name for p in predictions[:3]):
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
    evaluate_matcher(
        corpus_file="data/eval/corpus.jsonl",
        queries_file="data/eval/queries.jsonl",
        model_name="tfidf-char-ngram" 
    )
