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

def clean_company_name(name, remove_stopwords=True):
    """
    Làm sạch tên công ty: chuẩn hóa Unicode, tùy chọn xóa loại hình DN phổ biến.
    """
    if not name:
        return ""
    
    # 1. Chuẩn hóa cơ bản
    name = normalize_vietnamese_text(name)
    
    # 2. Xóa các ký tự đặc biệt (giữ lại & và +)
    name = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ&+\-]', ' ', name)
    
    if remove_stopwords:
        # 3. Loại bỏ nhiễu loại hình doanh nghiệp phổ biến (có thể mở rộng)
        stop_words = [
            r'\bcty\b', r'\bcông ty\b', r'\btnhh\b', r'\bcp\b', r'\bcổ phần\b', 
            r'\bmtv\b', r'\btrách nhiệm hữu hạn\b', r'\btm\b', r'\bdv\b', 
            r'\bthương mại\b', r'\bdịch vụ\b', r'\bjsc\b', r'\bltd\b', r'\bco\b',
            r'\bxnk\b', r'\bxuất nhập khẩu\b', r'\bimport\b', r'\bexport\b',
            r'\bva\b', r'\bvà\b', r'\bmt\b', r'\bdt\b', r'\bđt\b', r'\bđầu tư\b',
            r'\bxd\b', r'\bxây dựng\b', r'\bsx\b', r'\bsản xuất\b', r'\bpt\b',
            r'\bphát triển\b', r'\bvận tải\b', r'\bvt\b', r'\blogistics\b',
            r'\bchi nhánh\b', r'\bcn\b', r'\bvpđd\b', r'\bvăn phòng đại diện\b',
            r'\bmtv\b', r'\bmột thành viên\b', r'\b1 thành viên\b'
        ]
        
        for word in stop_words:
            name = re.sub(word, '', name)
            
        # Xử lý lại stop words cho bản không dấu để triệt để hơn
        name_no_accent = remove_accents(name)
        stop_words_no_accent = [
            r'\bcong ty\b', r'\bco phan\b', r'\btrach nhiem huu han\b',
            r'\bthuong mai\b', r'\bdich vu\b', r'\bxuat nhap khau\b',
            r'\bdau tu\b', r'\bxay dung\b', r'\bsan xuat\b', r'\bphat trien\b',
            r'\bvan tai\b', r'\bchi nhanh\b', r'\bmot thanh vien\b'
        ]
        for word in stop_words_no_accent:
            name_no_accent = re.sub(word, '', name_no_accent)
    else:
        name_no_accent = remove_accents(name)
    
    # 4. Gom nhóm khoảng trắng
    name = re.sub(r'\s+', ' ', name_no_accent).strip()
    
    return name

if __name__ == "__main__":
    test_name = "Công ty TNHH MTV Thương mại và Dịch vụ ABC"
    print(f"Original: {test_name}")
    print(f"Cleaned: {clean_company_name(test_name)}")
    print(f"No Accents: {remove_accents(clean_company_name(test_name))}")
