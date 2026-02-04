import itertools
from src.preprocess import remove_accents

class CombinatorialGenerator:
    def __init__(self):
        # Từ điển các biến thể của loại hình doanh nghiệp và các cụm từ phổ biến
        self.type_variants = {
            "công ty tnhh mtv": ["tnhh mtv", "tnhh một thành viên", "tnhh 1tv", "tnhh 1 thành viên"],
            "công ty tnhh": ["tnhh", "cty tnhh", "co. ltd", "co.,ltd", "trách nhiệm hữu hạn"],
            "công ty cổ phần": ["cp", "ctcp", "cty cp", "jsc", "cổ phần"],
            "tập đoàn": ["group", "tđ", "tap doan"],
            "thương mại dịch vụ": ["tm dv", "tm&dv", "tmdv", "thương mại và dịch vụ", "tm-dv"],
            "sản xuất thương mại": ["sx tm", "sxtm", "sx & tm", "sản xuất tm", "sx-tm"],
            "xuất nhập khẩu": ["xnk", "xn-kh", "import export", "imp-exp", "xnk"],
            "đầu tư xây dựng": ["đt xd", "dtxd", "đt & xd", "đầu tư và xây dựng", "đtxd"],
            "văn phòng đại diện": ["vpđd", "vpdd", "rep office", "văn phòng đd"],
            "chi nhánh": ["cn", "br", "chi nhanh"],
            "một thành viên": ["mtv", "1tv", "1 thành viên"]
        }

    def generate(self, original_name):
        """
        Tạo ra các tổ hợp từ keywords để sinh ra hàng loạt mẫu.
        """
        name_lower = original_name.lower()
        variants = set()
        
        # 1. Tìm và thay thế các từ khóa loại hình
        current_variations = [name_lower]
        
        for key, alt_list in self.type_variants.items():
            if key in name_lower:
                new_variations = []
                for var in current_variations:
                    for alt in [key] + alt_list:
                        new_variations.append(var.replace(key, alt))
                current_variations = list(set(new_variations))

        # 2. Thêm các biến thể: Không dấu, Viết hoa
        final_set = set()
        for v in current_variations:
            # Nguyên bản (nhưng đã thay type)
            final_set.add(v.strip())
            # Không dấu
            v_no_accent = remove_accents(v)
            final_set.add(v_no_accent.strip())
            # Viết hoa toàn bộ
            final_set.add(v.upper().strip())
            final_set.add(v_no_accent.upper().strip())

        # 3. Loại bỏ bản gốc nếu có
        if original_name.lower() in final_set:
            final_set.remove(original_name.lower())

        return list(final_set)

if __name__ == "__main__":
    gen = CombinatorialGenerator()
    test_name = "Công ty TNHH Thương mại Dịch vụ ABC"
    results = gen.generate(test_name)
    print(f"Sinh ra {len(results)} biến thể cho '{test_name}':")
    for r in list(results)[:10]:
        print(f" - {r}")
