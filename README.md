# Vietnamese Company Name Matching

Dự án này cung cấp bộ công cụ để sinh dữ liệu mô phỏng và so khớp tên công ty Việt Nam với độ chính xác cao. Hệ thống được thiết kế để xử lý các biến thể phổ biến trong đời thực như viết tắt, không dấu, sai lỗi chính tả và hoán đổi vị trí từ.

## 🚀 Tính năng chính

- **Scraper Dữ liệu:** Thu thập tên doanh nghiệp thực tế từ các nguồn uy tín (VNR500, Infocom).
- **Synthetic Data Generator:**
    - **Combinatorial:** Sinh hàng chục nghìn mẫu test dựa trên quy tắc (viết tắt, tiếng Anh, không dấu).
    - **LLM-based:** Sử dụng GLM-4 để sinh các lỗi gõ phím và biến thể tự nhiên.
- **High-performance Matching:** Sử dụng TF-IDF Char N-gram tối ưu cho tiếng Việt, đạt độ chính xác **>99%** với độ trễ **<3ms**.
- **Tiền xử lý thông minh:** Tự động chuẩn hóa Unicode, loại bỏ nhiễu loại hình doanh nghiệp (TNHH, CP, MTV, ...) để tập trung vào tên thương hiệu.

## 📁 Cấu trúc dự án

```text
├── main.py                 # File chạy demo nhanh
├── requirements.txt        # Danh sách thư viện cần thiết
├── src/
│   ├── preprocess.py       # Xử lý văn bản & Stop words tiếng Việt
│   ├── matching/
│   │   └── matcher.py      # Thuật toán so khớp TF-IDF
│   └── synthetic/
│       ├── combinatorial.py # Sinh dữ liệu theo quy tắc
│       └── generator.py     # Sinh dữ liệu qua LLM
├── scripts/
│   ├── scrape_infocom.py    # Tool thu thập dữ liệu doanh nghiệp
│   ├── generate_eval_dataset.py # Tạo tập dataset đánh giá (Corpus & Queries)
│   └── evaluate_matching.py     # Script đánh giá Accuracy & Latency
└── data/
    ├── sample_system_names.txt  # Danh sách 1000+ tên công ty chuẩn
    └── eval/                    # Chứa tập dữ liệu đánh giá dạng JSONL
```

## 🛠 Cài đặt

1. Tạo môi trường ảo và cài đặt dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt
```

2. Cấu hình LLM (tùy chọn - chỉ khi dùng `SyntheticGenerator`):
Cập nhật API Key trong [config/llm_config.yaml](config/llm_config.yaml).

## 📖 Hướng dẫn sử dụng

### 1. Thu thập dữ liệu
Nếu muốn mở rộng danh sách công ty:
```bash
python scripts/scrape_infocom.py
```

### 2. Sinh tập dữ liệu đánh giá
Tạo ra file `corpus.jsonl` và `queries.jsonl` từ danh sách tên công ty có sẵn:
```bash
python scripts/generate_eval_dataset.py
```

### 3. Đánh giá thuật toán
Chạy script để đo lường độ chính xác Top-1, Top-3 và thời gian xử lý:
```bash
python scripts/evaluate_matching.py
```

### 4. Chạy Demo thực tế
Sử dụng Matcher trong code của bạn:
```python
from src.matching.matcher import CompanyMatcher

matcher = CompanyMatcher()
matcher.build_index(["CÔNG TY TNHH SỮA VIỆT NAM", ...])
results = matcher.search("Vinamilk")
print(results)
```

## 📊 Kết quả thực nghiệm (trên 1,000 công ty / 50,000 queries)

- **Accuracy (Top 1):** ~99.8%
- **Accuracy (Top 3):** 100%
- **Avg Latency:** 2.1 ms / query

## ⚖️ Giấy phép
Dự án được phát triển cho mục đích học tập và nghiên cứu so khớp thực thể.
