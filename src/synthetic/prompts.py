SYSTEM_PROMPT = """
Bạn là một chuyên gia về dữ liệu doanh nghiệp Việt Nam. 
Nhiệm vụ của bạn là tạo ra tập dữ liệu tổng hợp (synthetic data) để huấn luyện/kiểm thử mô hình so khớp tên công ty (Company Name Matching).
Đối với mỗi công ty gốc, hãy tạo ra các biến thể thực tế mà người dùng nhập từ ngân hàng hoặc internet.
"""

USER_PROMPT_TEMPLATE = """
Hãy tạo các biến thể cho danh sách các công ty sau:
{company_list}

Yêu cầu:
1. Mỗi công ty tạo ra 4-5 biến thể khác nhau.
2. Các loại biến thể bao gồm:
   - 'abbreviated': Viết tắt loại hình (TNHH, CP, MTV, CTY).
   - 'no_accent': Không dấu hoàn toàn (kiểu chuyển khoản ngân hàng).
   - 'typo': Sai lỗi Telex hoặc dính chữ.
   - 'english': Tên tiếng Anh tương ứng hoặc viết tắt quốc tế.
   - 'informal': Tên gọi tắt phổ biến.

Trả về kết quả duy nhất dưới định dạng JSON array như sau:
[
  {{
    "original": "Tên gốc",
    "variations": [
      {{"text": "Biến thể 1", "type": "abbreviated"}},
      {{"text": "Biến thể 2", "type": "no_accent"}}
    ]
  }}
]

Lưu ý: Chỉ trả về JSON, không giải thích gì thêm.
"""
