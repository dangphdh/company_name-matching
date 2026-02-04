import os
from src.matching.matcher import CompanyMatcher

def main():
    # 1. Danh sách công ty "Chuẩn" từ hệ thống
    system_companies = [
        "Công ty Cổ phần Sữa Việt Nam",
        "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam",
        "Tập đoàn Viễn thông Quân đội",
        "Công ty TNHH Samsung Electronics Việt Nam",
        "Ngân hàng Thương mại Cổ phần Công thương Việt Nam",
        "Tổng Công ty Hàng không Việt Nam - CTCP",
        "Tập đoàn Xăng dầu Việt Nam",
        "Công ty Cổ phần FPT",
        "Tập đoàn Vingroup - Công ty CP",
        "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam"
    ]

    # 2. Khởi tạo Matcher (Mặc định dùng BGE-M3)
    matcher = CompanyMatcher(model_name='BAAI/bge-m3')
    
    # 3. Xây dựng index
    matcher.build_index(system_companies)

    # 4. Giả lập một số query "nhiễu" (không dấu, viết tắt, sai lệch)
    test_queries = [
        "VINAMILK",                        # Tên thương hiệu
        "CTY CO PHAN SUA VIET NAM",        # Viết tắt + Không dấu
        "ngan hang vietcombank",          # Tên tắt phổ biến
        "Samsung Electronics VN",          # Viết tắt tiếng Anh
        "VIETTEL",                         # Thương hiệu
        "BIDV",                            # Viết tắt ngân hàng
        "Tập đoàn Vingroup",               # Thiếu hậu tố
    ]

    print("\n--- KẾT QUẢ SO KHỚP ---")
    for query in test_queries:
        print(f"\nTruy vấn: '{query}'")
        results = matcher.search(query, top_k=3)
        for res in results:
            print(f" -> {res['company']} (Score: {res['score']:.4f})")

if __name__ == "__main__":
    main()
