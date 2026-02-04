import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.synthetic.generator import SyntheticGenerator
from src.synthetic.combinatorial import CombinatorialGenerator

def run_evaluation_dataset_generation(
    input_file,
    output_queries_file,
    output_corpus_file,
    llm_limit=None,
    llm_batch_size=20,
    api_provider=None,
    max_workers=5,
    use_parallel=True
):
    """
    Đọc danh sách tên công ty hệ thống và tạo dataset đánh giá.

    Args:
        input_file: Path to input company names file
        output_queries_file: Path to output queries file
        output_corpus_file: Path to output corpus file
        llm_limit: Maximum number of companies to generate LLM variants for (None = all)
        llm_batch_size: Number of companies per LLM batch
        api_provider: API provider to use ("openai", "zai", "openrouter")
        max_workers: Maximum number of parallel workers for LLM generation
        use_parallel: Whether to use parallel processing for LLM calls
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        system_names = [line.strip() for line in f if line.strip()]

    print(f"Using API provider: {api_provider or 'config default'}")
    llm_gen = SyntheticGenerator(api_provider=api_provider)
    comb_gen = CombinatorialGenerator()

    queries = []
    corpus = []

    print(f"Bắt đầu sinh dữ liệu cho {len(system_names)} công ty...")

    # Phase 1: Generate corpus and combinatorial variants (fast)
    print("\n=== Phase 1: Generating corpus and combinatorial variants ===")
    for idx, name in enumerate(tqdm(system_names, desc="Combinatorial")):
        corp_id = f"CORP_{idx:04d}"
        corpus.append({"id": corp_id, "name": name})

        # Generate combinatorial variants
        rule_variants = comb_gen.generate(name)
        for v in rule_variants:
            queries.append({
                "id": f"Q_RULE_{len(queries)}",
                "text": v,
                "target_id": corp_id,
                "method": "combinatorial"
            })

    print(f"Generated {len(corpus)} corpus entries and {len(queries)} combinatorial queries")

    # Phase 2: Generate LLM variants (slower, batched)
    companies_for_llm = system_names[:llm_limit] if llm_limit else system_names

    if companies_for_llm:
        print(f"\n=== Phase 2: Generating LLM variants ===")
        print(f"Processing {len(companies_for_llm)} companies with batch size {llm_batch_size}")
        print(f"Using {max_workers} parallel workers" if use_parallel else "Using sequential processing")

        llm_queries = process_llm_batches(
            companies_for_llm,
            corpus,
            llm_gen,
            llm_batch_size,
            max_workers if use_parallel else 1
        )
        queries.extend(llm_queries)

    # Save results
    print(f"\n=== Saving results ===")
    with open(output_corpus_file, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_queries_file, 'w', encoding='utf-8') as f:
        for item in queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n=== Summary ===")
    print(f"- Total companies (Corpus): {len(corpus)}")
    print(f"- Total test queries generated: {len(queries)}")
    print(f"  - Combinatorial: {sum(1 for q in queries if q['method'] == 'combinatorial')}")
    print(f"  - LLM-based: {sum(1 for q in queries if q['method'].startswith('llm_'))}")

def process_llm_batch(batch, corpus, llm_gen):
    """Process a single batch of companies through LLM."""
    try:
        llm_results = llm_gen.generate_variants(batch)

        if not isinstance(llm_results, list):
            return []

        batch_queries = []
        for result in llm_results:
            if not isinstance(result, dict):
                continue

            # Find the corpus ID for this company
            original_name = result.get("original", "")
            corp_id = None
            for corp in corpus:
                if corp["name"] == original_name:
                    corp_id = corp["id"]
                    break

            if not corp_id:
                continue

            # Add variants
            for var in result.get("variations", []):
                if isinstance(var, dict) and "text" in var and "type" in var:
                    batch_queries.append({
                        "id": f"Q_LLM_{len(batch_queries)}",
                        "text": var["text"],
                        "target_id": corp_id,
                        "method": f"llm_{var['type']}"
                    })

        return batch_queries

    except Exception as e:
        print(f"\nError processing batch: {e}")
        return []

def process_llm_batches(companies, corpus, llm_gen, batch_size, max_workers=1):
    """Process multiple batches of companies through LLM, optionally in parallel."""
    # Create batches
    batches = [companies[i:i + batch_size] for i in range(0, len(companies), batch_size)]
    all_queries = []

    if max_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_llm_batch, batch, corpus, llm_gen): i
                for i, batch in enumerate(batches)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM batches"):
                batch_queries = future.result()
                all_queries.extend(batch_queries)
    else:
        # Sequential processing
        for batch in tqdm(batches, desc="LLM batches"):
            batch_queries = process_llm_batch(batch, corpus, llm_gen)
            all_queries.extend(batch_queries)

    return all_queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation dataset for company name matching")
    parser.add_argument("--input", default="data/sample_system_names.txt", help="Input file with company names")
    parser.add_argument("--output-queries", default="data/eval/queries.jsonl", help="Output queries file")
    parser.add_argument("--output-corpus", default="data/eval/corpus.jsonl", help="Output corpus file")
    parser.add_argument("--llm-limit", type=int, default=None, help="Limit LLM generation to N companies")
    parser.add_argument("--llm-batch-size", type=int, default=20, help="Number of companies per LLM batch")
    parser.add_argument("--api-provider", choices=["openai", "zai", "openrouter"], default=None,
                        help="API provider to use (default: from config)")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum parallel workers for LLM calls")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("data/eval", exist_ok=True)

    run_evaluation_dataset_generation(
        input_file=args.input,
        output_queries_file=args.output_queries,
        output_corpus_file=args.output_corpus,
        llm_limit=args.llm_limit,
        llm_batch_size=args.llm_batch_size,
        api_provider=args.api_provider,
        max_workers=args.max_workers,
        use_parallel=not args.no_parallel
    )
