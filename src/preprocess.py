import re
import unicodedata

def normalize_vietnamese_text(text):
    """Chuẩn hóa tiếng Việt sang NFC."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    return text.lower().strip()


# Ordered list of (pattern, replacement) pairs.
# Applied after lowercase to canonicalize all entity-type variants → a single
# abbreviated token that is then KEPT in the cleaned name as a discriminating signal.
_ENTITY_NORMS = [
    # ── English / mixed-script forms ─────────────────────────────────────────
    (r'\bjsc\.?\b',              'cp'),     # JSC   → cp  (joint-stock = cổ phần)
    (r'\bco\.,?\s*ltd\.?\b',     'tnhh'),   # CO.,LTD / CO. LTD → tnhh
    (r'\bco\.\s*ltd\.?\b',       'tnhh'),
    (r'\bllc\.?\b',              'tnhh'),   # LLC   → tnhh
    (r'\bltd\.?\b',              'tnhh'),   # LTD   → tnhh
    (r'\bcorp\.?\b',             'cp'),     # CORP  → cp
    # ── English branch / representative office ───────────────────────────────
    (r'\brep(?:resentative)?\s+office\b', 'vpdd'),  # REP OFFICE → vpdd
    (r'\bbranch\b',              'cn'),     # BRANCH → cn  (chi nhánh)
    # ── Vietnamese abbreviation aliases ──────────────────────────────────────
    (r'\bctcp\b',                'cp'),     # CTCP  → cp
    (r'\bcty\s+cp\b',            'cp'),     # CTY CP → cp  (strips 'cty' later)
    (r'\bcty\s+tnhh\b',          'tnhh'),   # CTY TNHH → tnhh
    # ── TẬP ĐOÀN / GROUP → canonical token 'td' ──────────────────────────────
    (r'\btập đoàn\b',            'td'),     # accented long-form
    (r'\btap\s+doan\b',          'td'),     # no-accent long-form
    (r'\bgroup\b',               'td'),     # English GROUP (used as "tập đoàn")
    (r'\btd\b',                  'td'),     # abbreviation (already canonical)
    # ── Vietnamese long-form → canonical abbrev ───────────────────────────────
    (r'\bcổ phần\b',             'cp'),
    (r'\btrách nhiệm hữu hạn\b', 'tnhh'),
    (r'\bmột thành viên\b',      'mtv'),
    (r'\b1 thành viên\b',        'mtv'),
    (r'\bvăn phòng đại diện\b',  'vpdd'),
    (r'\bvpđd\b',                'vpdd'),
    # ── No-accent variants (queries already stripped of diacritics) ───────────
    (r'\bco\s+phan\b',           'cp'),
    (r'\btrach\s+nhiem\s+huu\s+han\b', 'tnhh'),
    (r'\bmot\s+thanh\s+vien\b',  'mtv'),
    (r'\bvan\s+phong\s+dai\s+dien\b', 'vpdd'),
    (r'\bhop\s+tac\s+xa\b',      'htx'),
    # ── Vietnamese long-form (accented) ──────────────────────────────────────
    (r'\bhợp tác xã\b',          'htx'),
    (r'\bhợp danh\b',            'hd'),
    (r'\bhop\s+danh\b',          'hd'),
]

def normalize_entity_types(name: str) -> str:
    """
    Map all entity-type surface forms to a single canonical abbreviation token.
    Must be called AFTER normalize_vietnamese_text() (lowercase NFC) and
    BEFORE special character removal (so that CO.,LTD etc. are still intact).

    Examples:
        'jsc thuong mai abc'         → 'cp thuong mai abc'
        'co.,ltd dich vu xyz'        → 'tnhh dich vu xyz'
        'ctcp tm dv abc'             → 'cp tm dv abc'
        'cổ phần a b c'              → 'cp a b c'
        'trách nhiệm hữu hạn a b'   → 'tnhh a b'
    """
    for pattern, replacement in _ENTITY_NORMS:
        name = re.sub(pattern, replacement, name)
    return name


# Normalise common functional-word abbreviations so that a query using an
# abbreviated form (e.g. "IMP-EXP", "DTXD", "TM&DV") and an index entry using
# the expanded form (e.g. "XUẤT NHẬP KHẨU", "ĐẦU TƯ XÂY DỰNG", "THƯƠNG MẠI
# DỊCH VỤ") both collapse to the same canonical token.
# Applied BEFORE special-char stripping so hyphens/ampersands are still intact.
_FUNCTIONAL_NORMS = [
    # ── IMP-EXP / IMPORT EXPORT ↔ XUẤT NHẬP KHẨU ────────────────────────────
    (r'\bimp[\-\s]?exp\b',                        'xnk'),
    (r'\bimport[\-\s]+export\b',                  'xnk'),
    (r'\bxuất nhập khẩu\b',                       'xnk'),
    (r'\bxuat\s+nhap\s+khau\b',                   'xnk'),
    # ── SX-TM / SXTM ↔ SẢN XUẤT THƯƠNG MẠI ─────────────────────────────────
    # NOTE: sxtm patterns must come BEFORE tmdv patterns so that in the string
    # "sản xuất thương mại dịch vụ":
    #   1) "sản xuất thương mại" → sxtm   (sxtm fires first, consumes "thương mại")
    #   2) "dịch vụ" remains → tmdv patterns have no "thương mại" left to match
    # This ensures "sxtm dịch vụ" for both index and query forms.
    (r'\bsx(?:\s*[&\-]\s*|\s+)tm\b',             'sxtm'),  # SX-TM / SX TM / SX & TM
    (r'\bsxtm\b',                                 'sxtm'),  # already canonical
    (r'\bsản xuất thương mại\b',                  'sxtm'),  # accented long-form
    (r'\bsan\s+xuat\s+(?:thuong\s+mai|tm)\b',     'sxtm'),  # no-accent: full or abbrev
    # ── TM&DV / TMDV / TM-DV ↔ THƯƠNG MẠI DỊCH VỤ ───────────────────────────
    # Applied AFTER sxtm so that "sản xuất thương mại" is already consumed.
    (r'\btm\s*[&\-]\s*dv\b',                      'tmdv'),  # TM&DV / TM-DV
    (r'\bthương mại\s+(?:và\s+)?dịch vụ\b',       'tmdv'),  # with or without 'và'
    (r'\bthuong\s+mai\s+(?:va\s+)?dich\s+vu\b',   'tmdv'),  # no-accent
    (r'\btm\s+(?:và\s+)?dịch vụ\b',               'tmdv'),  # TM (và) dịch vụ
    (r'\btm\s+(?:va\s+)?dich\s+vu\b',             'tmdv'),  # no-accent mixed
    # ── DTXD / DT XD ↔ ĐẦU TƯ (VÀ) XÂY DỰNG ────────────────────────────────
    (r'\bdtxd\b',                                  'dtxd'),
    (r'\bđt\s*xd\b',                              'dtxd'),
    (r'\bdt\s*xd\b',                              'dtxd'),
    (r'\bđầu tư\s+(?:và\s+)?xây dựng\b',          'dtxd'),  # with or without 'và'
    (r'\bdau\s+tu\s+(?:va\s+)?xay\s+dung\b',      'dtxd'),  # no-accent
    # ── 1TV / 1 TV (abbreviation for "một thành viên") ───────────────────────
    (r'\b1\s*tv\b',                               'mtv'),
    # ── BR (branch prefix, e.g. "BR Công ty…" → "cn Công ty…") ──────────────
    (r'(?:^|\s)br\s+(?=(?:co|cty|công|tnhh|cp|chi|cn)\b)', 'cn '),
]


def normalize_functional_terms(name: str) -> str:
    """
    Collapse common functional-word synonyms/abbreviations to canonical tokens
    so that TF-IDF char n-grams align regardless of writing style.

    Must be called AFTER normalize_entity_types() and BEFORE special-char removal.

    Examples:
        'tnhh imp-exp nam son'               → 'tnhh xnk nam son'
        'tnhh xuất nhập khẩu nam sơn'       → 'tnhh xnk nam son'  (via dual-index)
        'cp tư vấn dtxd tmdv trường thịnh'  → 'cp tư vấn dtxd tmdv trường thịnh'
        'cp tư vấn đầu tư xây dựng tm&dv…' → 'cp tư vấn dtxd tmdv …'
        'tnhh mtv sxtm imp-exp nam son'      → 'tnhh mtv sxtm xnk nam son'
    """
    for pattern, replacement in _FUNCTIONAL_NORMS:
        name = re.sub(pattern, replacement, name)
    return name


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
    Làm sạch tên công ty: chuẩn hóa Unicode, chuẩn hoá loại hình DN,
    tùy chọn xóa từ nhiễu (giữ lại các ký hiệu phân biệt như cp/tnhh/mtv).
    """
    if not name:
        return ""

    # 1. Chuẩn hóa cơ bản (NFC + lowercase)
    name = normalize_vietnamese_text(name)

    # 1b. Chuẩn hoá loại hình DN: JSC→cp, CO.,LTD→tnhh, ctcp→cp, etc.
    #     Phải chạy TRƯỚC khi xóa ký tự đặc biệt để bắt CO.,LTD v.v.
    name = normalize_entity_types(name)

    # 1c. Chuẩn hoá cụm từ chức năng: IMP-EXP→xnk, DTXD→dtxd, TM&DV→tmdv, v.v.
    #     Giúp query dạng viết tắt và index dạng đầy đủ khớp char n-gram.
    name = normalize_functional_terms(name)

    # 2. Xóa các ký tự đặc biệt (giữ lại & và +)
    name = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ&+\-]', ' ', name)

    if remove_stopwords:
        # 3. Loại bỏ tiếng ồn — CHỈ xóa các từ KHÔNG phân biệt.
        #    Giữ lại cp / tnhh / mtv / vpdd / htx / hd vì đây là tín hiệu phân biệt
        #    giữa các công ty cùng tên thương hiệu nhưng khác loại hình.
        stop_words = [
            # Generic company-noun prefix
            r'\bcông ty\b', r'\bcty\b',
            # Generic activity words
            r'\bthương mại\b', r'\bdịch vụ\b',
            r'\bxuất nhập khẩu\b', r'\bxnk\b',
            r'\bđầu tư\b', r'\bxây dựng\b', r'\bsản xuất\b',
            r'\bphát triển\b', r'\bvận tải\b', r'\blogistics\b',
            # Generic connectors
            r'\bvà\b',
            # Abbreviations for the above (do NOT include cp/tnhh/mtv/cn)
            r'\btm\b', r'\bdv\b', r'\bdt\b', r'\bđt\b',
            r'\bxd\b', r'\bsx\b', r'\bpt\b', r'\bvt\b',
            r'\bmt\b',
            # Canonical functional-phrase tokens (created by normalize_functional_terms)
            r'\btmdv\b', r'\bdtxd\b', r'\bsxtm\b',
            # Now-redundant long-form expansions (already normalized to abbrevs above)
            r'\btrách nhiệm hữu hạn\b', r'\bcổ phần\b',
            r'\bmột thành viên\b', r'\b1 thành viên\b',
            r'\bvăn phòng đại diện\b', r'\bvpđd\b',
        ]

        for word in stop_words:
            name = re.sub(word, '', name)

        # Xử lý lại stop words cho bản không dấu để triệt để hơn
        name_no_accent = remove_accents(name)
        stop_words_no_accent = [
            r'\bcong ty\b',
            r'\bthuong mai\b', r'\bdich vu\b',
            r'\bxuat nhap khau\b',
            r'\bdau tu\b', r'\bxay dung\b', r'\bsan xuat\b',
            r'\bphat trien\b', r'\bvan tai\b',
            # Long-form no-accent (already normalized but guard again)
            r'\btrach nhiem huu han\b', r'\bco phan\b',
            r'\bmot thanh vien\b', r'\bvan phong dai dien\b',
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
