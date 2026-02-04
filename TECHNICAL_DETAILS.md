# Chi tiết Kỹ thuật (Technical Details)

Tài liệu này giải thích các quyết định kỹ thuật và logic xử lý đằng sau hệ thống Matching.

## 1. Chiến lược Tiền xử lý (Preprocessing)

Hệ thống xử lý tên công ty qua các bước sau trong `src/preprocess.py`:

1.  **Chuẩn hóa NFC:** Đảm bảo tất cả ký tự tiếng Việt đồng nhất về mã Unicode.
2.  **Loại bỏ Stop Words nâng cao:** 
    *   Không chỉ xóa "Công ty TNHH", mà còn xóa các cụm từ bổ trợ gây nhiễu cho thuật toán vector như "thương mại", "dịch vụ", "một thành viên", "chi nhánh", "VPĐD",...
    *   Xử lý stop words trên cả văn bản có dấu và không dấu để tối đa hóa khả năng khớp tên thương hiệu (Brand name).
3.  **Lọc ký tự đặc biệt:** Giữ lại các ký tự mang ý nghĩa trong tên riêng như `&`, `+`, `-` và loại bỏ các ký tự nhiễu khác.

## 2. Tìm kiếm và So khớp (Matching)

Thuật toán hiện tại sử dụng **TF-IDF Char N-gram** (thay vì Vector Embedding truyền thống) vì các lý do:

-   **Xử lý viết tắt:** `TNHH` và `Trách nhiệm hữu hạn` sau khi qua stop words sẽ bị loại bỏ, chỉ còn lại Brand name. Char N-gram giúp khớp `Samsung` với `Samsng` (lỗi chính tả) hoặc `SSVN` (viết tắt) hiệu quả hơn.
-   **Tốc độ:** TF-IDF trên tập 1,000 thực thể cho tốc độ phản hồi tính bằng miligiây, phù hợp cho các hệ thống thời gian thực mà không cần GPU.
-   **Độ chính xác:** Qua đánh giá, TF-IDF Char (ngram 2-5) xử lý hoán đổi từ ("A&P TM DV" vs "TM DV A&P") tốt hơn so với các phương pháp so khớp chuỗi thông thường (Levenshtein).

## 3. Sinh dữ liệu mô phỏng (Synthetic Generation)

Hệ thống kết hợp 2 phương pháp:

1.  **Combinatorial (Quy tắc):**
    *   Thay thế loại hình DN bằng các từ đồng nghĩa/viết tắt.
    *   Chuyển sang không dấu.
    *   Dịch các thuật ngữ kinh doanh phổ biến sang tiếng Anh (Import/Export, Construction, ...).
2.  **LLM (Hành vi):**
    *   Mô phỏng các lỗi gõ phím của người dùng Việt (ví dụ gõ nhầm Telex).
    *   Mô phỏng cách con người viết tắt tên riêng một cách ngẫu hứng.

## 4. Đánh giá (Evaluation)

Hệ thống sử dụng metric:
-   **Top-1 Accuracy:** Tỉ lệ kết quả đúng đầu tiên trùng khớp với Target ID.
-   **Top-3 Accuracy:** Tỉ lệ Target ID nằm trong 3 kết quả trả về đầu tiên.
-   **Latency:** Thời gian xử lý trung bình cho mỗi yêu cầu.
