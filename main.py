import os
from src.matching.matcher import CompanyMatcher

def main():
    # 1. Load danh sách công ty từ file sample_system_names.txt
    corpus_path = "data/sample_system_names.txt"
    if not os.path.exists(corpus_path):
        print(f"Lỗi: Không tìm thấy file {corpus_path}")
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        system_companies = [line.strip() for line in f if line.strip()]

    # 2. Khởi tạo Matcher - chọn modeltype
    # Option 1: TF-IDF Char N-gram (classic, fast)
    # matcher = CompanyMatcher(model_name='tfidf')
    
    # Option 2: BM25 (term relevance)
    # matcher = CompanyMatcher(model_name='bm25')
    
    # Option 3: Hybrid TF-IDF + BM25 (best overall performance) ⭐ RECOMMENDED
    matcher = CompanyMatcher(model_name='tfidf-bm25', tfidf_weight=0.5, bm25_weight=0.5)
    
    # 3. Xây dựng index
    matcher.build_index(system_companies)

    # 4. Giả lập một số query thực tế
    test_queries = [
        "BAO BI DUY TIN",                   # Tên sạch, không dấu
        "TNHH THƯƠNG MẠI DỊCH VỤ XNK A&P",  # Tên đầy đủ
        "IMPORT EXPORT A&P",                # Tên tiếng Anh / Loại hình tiếng Anh
        "CÔNG TY TNHH TM DV WU GIA",        # Viết tắt
        "văn phòng đại diện power điện",     # Đảo chữ + không dấu
        "cty cp hdt",                       # Viết tắt + Tên ngắn
    ]

    print("\n--- KẾT QUẢ SO KHỚP (Top 1) ---")
    print(f"Model: {matcher.model_name.upper()}\n")
    for query in test_queries:
        results = matcher.search(query, top_k=1)
        if results:
            print(f"Query: '{query:30}' -> Match: {results[0]['company']} ({results[0]['score']:.4f})")
        else:
            print(f"Query: '{query:30}' -> Không tìm thấy")
    
    # 5. Demo comparing different models
    print("\n\n--- DEMO: SO SÁNH CÁC MÔ HÌNH ---")
    demo_query = "Vinamilk"
    print(f"\nQuery: '{demo_query}'")
    print("=" * 80)
    
    models = [
        ('tfidf', {}),
        ('bm25', {}),
        ('tfidf-bm25', {'tfidf_weight': 0.5, 'bm25_weight': 0.5}),
        ('tfidf-bm25', {'tfidf_weight': 0.7, 'bm25_weight': 0.3}),
    ]
    
    for model_name, kwargs in models:
        temp_matcher = CompanyMatcher(model_name=model_name, **kwargs)
        temp_matcher.build_index(system_companies)
        results = temp_matcher.search(demo_query, top_k=1)
        if results:
            label = model_name.upper()
            if kwargs:
                label += f" ({kwargs['tfidf_weight']}/{kwargs['bm25_weight']})"
            print(f"{label:30} -> {results[0]['company']:40} (score: {results[0]['score']:.4f})")

if __name__ == "__main__":
    main()

