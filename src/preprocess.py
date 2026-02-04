import re
import unicodedata

def normalize_vietnamese_text(text):
    """Chuẩn hóa tiếng Việt sang NFC."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    return text.lower().strip()

def remove_accents(input_str):
    """Loại bỏ dấu tiếng Việt."""
    if not input_str:
        return ""
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ""
    for char in input_str:
        if char in s1:
            s += s0[s1.index(char)]
        else:
            s += char
    return s

def clean_company_name(name):
    """
    Làm sạch tên công ty: chuẩn hóa Unicode, xóa loại hình DN phổ biến để so khớp tốt hơn.
    """
    if not name:
        return ""
    
    # 1. Chuẩn hóa cơ bản
    name = normalize_vietnamese_text(name)
    
    # 2. Xóa các ký tự đặc biệt
    name = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', name)
    
    # 3. Loại bỏ nhiễu loại hình doanh nghiệp phổ biến (có thể mở rộng)
    stop_words = [
        r'\bcty\b', r'\bcông ty\b', r'\btnhh\b', r'\bcp\b', r'\bcổ phần\b', 
        r'\bmtv\b', r'\btrách nhiệm hữu hạn\b', r'\btm\b', r'\bdv\b', 
        r'\bthương mại\b', r'\bdịch vụ\b', r'\bjsc\b', r'\bltd\b', r'\bco\b'
    ]
    
    for word in stop_words:
        name = re.sub(word, '', name)
        
    # 4. Gom nhóm khoảng trắng
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

if __name__ == "__main__":
    test_name = "Công ty TNHH MTV Thương mại và Dịch vụ ABC"
    print(f"Original: {test_name}")
    print(f"Cleaned: {clean_company_name(test_name)}")
    print(f"No Accents: {remove_accents(clean_company_name(test_name))}")
