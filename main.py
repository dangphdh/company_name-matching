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

    # 2. Khởi tạo Matcher 
    # Mặc định dùng TF-IDF Char N-gram vì độ chính xác cao và chạy ổn định trên Windows
    matcher = CompanyMatcher()
    
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
    for query in test_queries:
        results = matcher.search(query, top_k=1)
        if results:
            print(f"Query: '{query:30}' -> Match: {results[0]['company']} ({results[0]['score']:.4f})")
        else:
            print(f"Query: '{query:30}' -> Không tìm thấy")

if __name__ == "__main__":
    main()
