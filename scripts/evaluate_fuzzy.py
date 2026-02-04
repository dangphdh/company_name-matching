import json
import time
from tqdm import tqdm
from thefuzz import fuzz, process
from src.preprocess import clean_company_name

class FuzzyMatcher:
    def __init__(self):
        self.corpus = []
        
    def build_index(self, names):
        self.corpus = [clean_company_name(n) for n in names]
        self.original_names = names

    def search(self, query, top_k=5):
        query_cleaned = clean_company_name(query)
        # kết quả trả về: [(text, score, index), ...]
        results = process.extract(query_cleaned, self.corpus, scorer=fuzz.token_set_ratio, limit=top_k)
        
        final_results = []
        for result in results:
            # Xử lý trường hợp format kết quả khác nhau tùy version
            if len(result) == 3:
                match_text, score, index = result
            else:
                match_text, score = result
                # Nếu không có index, ta phải tìm index bằng cách khác (không khuyến khích)
                index = self.corpus.index(match_text)
                
            final_results.append({
                "company": self.original_names[index],
                "score": score / 100.0
            })
        return final_results

def evaluate_matcher(corpus_file, queries_file):
    corpus_data = []
    corp_id_to_name = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus_data.append(item['name'])
            corp_id_to_name[item['id']] = item['name']

    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))

    # Test nhanh với 200 mẫu
    import random
    random.seed(42)
    sample_queries = random.sample(queries, min(len(queries), 200))

    print(f"Bắt đầu đánh giá (Fuzzy Logic) với {len(corpus_data)} cty và {len(sample_queries)} query mẫu...")

    matcher = FuzzyMatcher()
    matcher.build_index(corpus_data)

    hits = 0
    start_time = time.time()
    results_analysis = []

    for q in tqdm(sample_queries, desc="Evaluating"):
        target_name = corp_id_to_name[q['target_id']]
        predictions = matcher.search(q['text'], top_k=3)
        
        if predictions and predictions[0]['company'] == target_name:
            hits += 1
        else:
            results_analysis.append({
                "query": q['text'],
                "method": q['method'],
                "target": target_name,
                "predicted": predictions[0]['company'] if predictions else "None",
                "score": predictions[0]['score'] if predictions else 0
            })

    duration = time.time() - start_time
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ (FUZZY MATCHING)")
    print("="*30)
    print(f"Accuracy (Top 1): {hits / len(sample_queries):.2%}")
    print(f"Avg Latency: {duration / len(sample_queries) * 1000:.2f} ms/query")
    print("="*30)

    if results_analysis:
        print("\nMẫu lỗi:")
        for err in results_analysis[:3]:
            print(f"- Query: '{err['query']}' -> Target: '{err['target']}' (Pred: '{err['predicted']}')")

if __name__ == "__main__":
    evaluate_matcher("data/eval/corpus.jsonl", "data/eval/queries.jsonl")
