"""
Script to augment company names by generating variants with different company types and industries.
This creates more training data and noise to make the matching model more robust.
"""
import re
import random
from typing import List, Dict, Set

# Company type mappings for substitution
COMPANY_TYPE_REPLACEMENTS = {
    # TNHH variants
    r'\bCÔNG TY TNHH\b': ['CÔNG TY CỔ PHẦN', 'CÔNG TY HỢP DANH', 'DOANH NGHIỆP TƯ NHÂN', 'CÔNG TY'],
    r'\bCông ty TNHH\b': ['Công ty Cổ phần', 'Công ty Hợp danh', 'Doanh nghiệp Tư nhân', 'Công ty'],
    r'\bCTY TNHH\b': ['CTY CỔ PHẦN', 'CTY HỢP DANH', 'DOANH NGHIỆP TN', 'CTY'],
    r'\bTNHH\b': ['CỔ PHẦN', 'HỢP DANH', 'TƯ NHÂN', 'CP'],
    r'\bCo., Ltd\b': ['JSC', 'Corp', 'LLC', 'Ltd'],
    r'\bCO\. LTD\b': ['JSC', 'CORP', 'LLC', 'LTD'],

    # Cổ phần variants
    r'\bCÔNG TY CỔ PHẦN\b': ['CÔNG TY TNHH', 'CÔNG TY HỢP DANH', 'TẬP ĐOÀN', 'CÔNG TY'],
    r'\bCông ty Cổ phần\b': ['Công ty TNHH', 'Công ty Hợp danh', 'Tập đoàn', 'Công ty'],
    r'\bCTCP\b': ['TNHH', 'HĐ', 'TĐ', 'CTY'],
    r'\bCỔ PHẦN\b': ['TNHH', 'HỢP DANH', 'CP', 'TẬP ĐOÀN'],
    r'\bJSC\b': ['LTD', 'LLC', 'CORP', 'INC'],

    # Tập đoàn variants
    r'\bTẬP ĐOÀN\b': ['CÔNG TY CỔ PHẦN', 'CÔNG TY TNHH', 'TỔNG CÔNG TY', 'GROUP'],
    r'\bTập đoàn\b': ['Công ty Cổ phần', 'Công ty TNHH', 'Tổng công ty', 'Group'],

    # Doanh nghiệp tư nhân
    r'\bDOANH NGHIỆP TƯ NHÂN\b': ['CÔNG TY TNHH', 'CÔNG TY CỔ PHẦN', 'DNTN'],
    r'\bDoanh nghiệp Tư nhân\b': ['Công ty TNHH', 'Công ty Cổ phần', 'DNTN'],
    r'\bDNTN\b': ['TNHH', 'CP', 'CTY'],
}

# Industry term mappings for substitution
INDUSTRY_REPLACEMENTS = {
    # Construction
    r'\bXÂY DỰNG\b': ['SẢN XUẤT', 'THƯƠNG MẠI', 'DỊCH VỤ', 'CÔNG NGHIỆP', 'XD'],
    r'\bXây dựng\b': ['Sản xuất', 'Thương mại', 'Dịch vụ', 'Công nghiệp', 'XD'],
    r'\bXD\b': ['SX', 'TM', 'DV', 'CN'],

    # Trade/Commerce
    r'\bTHƯƠNG MẠI\b': ['SẢN XUẤT', 'DỊCH VỤ', 'XÂY DỰNG', 'TM', 'KINH DOANH'],
    r'\bThương mại\b': ['Sản xuất', 'Dịch vụ', 'Xây dựng', 'TM', 'Kinh doanh'],
    r'\bTM\b': ['SX', 'DV', 'XD', 'KD'],

    # Services
    r'\bDỊCH VỤ\b': ['THƯƠNG MẠI', 'SẢN XUẤT', 'TƯ VẤN', 'GIẢI PHÁP', 'DV'],
    r'\bDịch vụ\b': ['Thương mại', 'Sản xuất', 'Tư vấn', 'Giải pháp', 'DV'],
    r'\bDV\b': ['TM', 'SX', 'TV'],

    # Manufacturing/Production
    r'\bSẢN XUẤT\b': ['THƯƠNG MẠI', 'DỊCH VỤ', 'CHẾ BIẾN', 'SX'],
    r'\bSản xuất\b': ['Thương mại', 'Dịch vụ', 'Chế biến', 'SX'],
    r'\bSX\b': ['TM', 'DV', 'CB'],

    # Import/Export
    r'\bXUẤT NHẬP KHẨU\b': ['THƯƠNG MẠI', 'QUỐC TẾ', 'LOGISTICS', 'XNK'],
    r'\bXuất nhập khẩu\b': ['Thương mại', 'Quốc tế', 'Logistics', 'XNK'],
    r'\bXNK\b': ['TM', 'QT', 'LOGISTICS', 'IMP-EXP'],

    # Investment
    r'\bĐẦU TƯ\b': ['PHÁT TRIỂN', 'QUẢN LÝ', 'TÀI CHÍNH', 'ĐT'],
    r'\bĐầu tư\b': ['Phát triển', 'Quản lý', 'Tài chính', 'ĐT'],
    r'\bĐT\b': ['PT', 'QL', 'TC'],

    # Transport/Logistics
    r'\bVẬN TẢI\b': ['LOGISTICS', 'GIAO NHẬN', 'VẬN CHUYỂN', 'VT'],
    r'\bVận tải\b': ['Logistics', 'Giao nhận', 'Vận chuyển', 'VT'],
    r'\bLOGISTICS\b': ['VẬN TẢI', 'GIAO NHẬN', 'PHÂN PHỐI'],

    # Technology
    r'\bCÔNG NGHỆ\b': ['THÔNG TIN', 'KỸ THUẬT', 'GIẢI PHÁP', 'CN'],
    r'\bCông nghệ\b': ['Thông tin', 'Kỹ thuật', 'Giải pháp', 'CN'],
    r'\bCN\b': ['IT', 'KT', 'GP'],

    # Food/Foodstuffs
    r'\bTHỰC PHẨM\b': ['ĐỒ UỐNG', 'TIÊU DÙNG', 'HÀNG TIÊU DÙNG', 'TP'],
    r'\bThực phẩm\b': ['Đồ uống', 'Tiêu dùng', 'Hàng tiêu dùng', 'TP'],
}


def generate_company_variants(company_name: str, num_variants: int = 3) -> List[str]:
    """
    Generate variants of a company name by changing company type and industry terms.

    Args:
        company_name: Original company name
        num_variants: Number of variants to generate per name

    Returns:
        List of variant company names
    """
    variants = set()

    # Generate company type variants
    for pattern, replacements in COMPANY_TYPE_REPLACEMENTS.items():
        if re.search(pattern, company_name, re.IGNORECASE):
            for replacement in random.sample(replacements, min(len(replacements), num_variants)):
                variant = re.sub(pattern, replacement, company_name, flags=re.IGNORECASE)
                # Preserve original case pattern roughly
                if company_name.isupper():
                    variant = variant.upper()
                variants.add(variant)
            break

    # Generate industry term variants
    for pattern, replacements in INDUSTRY_REPLACEMENTS.items():
        if re.search(pattern, company_name, re.IGNORECASE):
            for replacement in random.sample(replacements, min(len(replacements), num_variants)):
                variant = re.sub(pattern, replacement, company_name, flags=re.IGNORECASE)
                if company_name.isupper():
                    variant = variant.upper()
                variants.add(variant)
            break

    # If no patterns matched, try adding common prefixes/suffixes
    if not variants:
        common_additions = [
            ('', ' VIỆT NAM'),
            ('', ' HỒ CHÍ MINH'),
            ('', ' HÀ NỘI'),
            ('CÔNG TY ', ''),
            ('', ' QUỐC TẾ'),
        ]
        for prefix, suffix in random.sample(common_additions, min(len(common_additions), num_variants)):
            variant = prefix + company_name + suffix
            variants.add(variant)

    return list(variants)


def augment_corpus(
    input_file: str = 'data/sample_system_names.txt',
    output_file: str = 'data/sample_system_names.txt',
    variants_per_company: int = 2,
    max_total_variants: int = 2000
) -> None:
    """
    Augment the company name corpus by generating variants.

    Args:
        input_file: Path to input file with company names
        output_file: Path to output file (will be overwritten)
        variants_per_company: Number of variants to generate per company
        max_total_variants: Maximum number of variants to add (to avoid too much noise)
    """
    # Read existing company names
    print(f"Reading companies from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f if line.strip()]

    print(f"Found {len(companies)} companies")

    # Generate variants
    all_variants: Set[str] = set()
    print(f"Generating variants (max {max_total_variants} total)...")

    # Shuffle companies to get diverse variants
    random.shuffle(companies)

    for i, company in enumerate(companies):
        if len(all_variants) >= max_total_variants:
            break

        variants = generate_company_variants(company, num_variants=variants_per_company)

        for variant in variants:
            # Don't add the original name or duplicates
            if variant != company and variant not in companies and variant not in all_variants:
                all_variants.add(variant)

            if len(all_variants) >= max_total_variants:
                break

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} companies, generated {len(all_variants)} variants...")

    # Merge original companies with variants
    augmented_companies = sorted(list(set(companies) | all_variants))

    # Write to output file
    print(f"\nWriting {len(augmented_companies)} total companies to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for company in augmented_companies:
            f.write(f"{company}\n")

    print(f"\n=== Summary ===")
    print(f"Original companies: {len(companies)}")
    print(f"New variants added: {len(all_variants)}")
    print(f"Total companies: {len(augmented_companies)}")
    print(f"Increase: {len(augmented_companies) - len(companies)} companies ({100 * (len(augmented_companies) - len(companies)) / len(companies):.1f}%)")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Augment the corpus
    augment_corpus(
        input_file='data/sample_system_names.txt',
        output_file='data/sample_system_names.txt',
        variants_per_company=2,
        max_total_variants=2000
    )
