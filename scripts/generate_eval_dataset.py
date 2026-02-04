import json
import os
from tqdm import tqdm
from src.synthetic.generator import SyntheticGenerator
from src.synthetic.combinatorial import CombinatorialGenerator

def run_evaluation_dataset_generation(input_file, output_queries_file, output_corpus_file):
    """
    Đọc danh sách tên công ty hệ thống và tạo dataset đánh giá.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        system_names = [line.strip() for line in f if line.strip()]

    llm_gen = SyntheticGenerator()
    comb_gen = CombinatorialGenerator()
    
    queries = []
    corpus = []
    relevance = []

    print(f"Bắt đầu sinh dữ liệu cho {len(system_names)} công ty...")

    for idx, name in enumerate(tqdm(system_names)):
        corp_id = f"CORP_{idx:04d}"
        corpus.append({"id": corp_id, "name": name})

        # 1. Sinh bằng Quy tắc (Combinatorial) - Nhanh, nhiều
        rule_variants = comb_gen.generate(name)
        for v in rule_variants:
            queries.append({
                "id": f"Q_RULE_{len(queries)}",
                "text": v,
                "target_id": corp_id,
                "method": "combinatorial"
            })

        # 2. Sinh bằng LLM (Chọn lọc, thông minh hơn - ví dụ 1 batch 10 cty)
        # Để tiết kiệm, ở bản demo này ta chỉ sinh LLM cho 2 công ty đầu tiên
        if idx < 2:
            llm_results = llm_gen.generate_variants([name])
            # Giả định llm_results trả về đúng format [ { "original": ..., "variations": [ { "text": ..., "type": ... } ] } ]
            if isinstance(llm_results, list) and len(llm_results) > 0:
                for var in llm_results[0].get("variations", []):
                    queries.append({
                        "id": f"Q_LLM_{len(queries)}",
                        "text": var["text"],
                        "target_id": corp_id,
                        "method": f"llm_{var['type']}"
                    })

    # Lưu Corpus
    with open(output_corpus_file, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Lưu Queries (Dataset đánh giá)
    with open(output_queries_file, 'w', encoding='utf-8') as f:
        for item in queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nHoàn thành!")
    print(f"- Tổng số công ty (Corpus): {len(corpus)}")
    print(f"- Tổng số mẫu test sinh ra (Queries): {len(queries)}")

if __name__ == "__main__":
    # Đảm bảo thư mục data/eval tồn tại
    os.makedirs("data/eval", exist_ok=True)
    
    run_evaluation_dataset_generation(
        input_file="data/sample_system_names.txt",
        output_queries_file="data/eval/queries.jsonl",
        output_corpus_file="data/eval/corpus.jsonl"
    )
