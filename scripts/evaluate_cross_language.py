#!/usr/bin/env python3
"""
Cross-Language Company Name Matching Evaluation

Evaluates the performance of different matching models on English queries
targeting Vietnamese company names.

Usage:
    python scripts/evaluate_cross_language.py --model tfidf-wordllama
    python scripts/evaluate_cross_language.py --model tfidf-wordllama --profile-latency
"""

import argparse
import json
import time
from typing import List, Dict, Tuple
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.matcher import CompanyMatcher


# Test dataset: English queries → Vietnamese company names
CROSS_LANGUAGE_TEST_SET = [
    # Dairy/Food companies
    {
        "query": "Vietnam Dairy Products JSC",
        "target": "CÔNG TY CỔ PHẦN SỮA VIỆT NAM",
        "category": "dairy"
    },
    {
        "query": "Vinamilk",
        "target": "CÔNG TY CỔ PHẦN SỮA VIỆT NAM",
        "category": "dairy"
    },
    {
        "query": "Vietnam Milk Company",
        "target": "CÔNG TY CỔ PHẦN SỮA VIỆT NAM",
        "category": "dairy"
    },

    # Banking
    {
        "query": "Vietcombank",
        "target": "Ngân hàng TMCP Ngoại thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "Vietnam Commercial Bank for Foreign Trade",
        "target": "Ngân hàng TMCP Ngoại thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "VCB Bank",
        "target": "Ngân hàng TMCP Ngoại thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "Vietnam Joint Stock Commercial Bank for Industry and Trade",
        "target": "Ngân hàng TMCP Công Thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "Vietinbank",
        "target": "Ngân hàng TMCP Công Thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "Vietnam Bank for Industry and Trade",
        "target": "Ngân hàng TMCP Công Thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "Vietnam Technological and Commercial Joint Stock Bank",
        "target": "Ngân hàng TMCP Kỹ Thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "Techcombank",
        "target": "Ngân hàng TMCP Kỹ Thương Việt Nam",
        "category": "banking"
    },
    {
        "query": "Vietnam Prosperity Joint Stock Commercial Bank",
        "target": "Ngân hàng TMCP Phát Hành Việt Nam",
        "category": "banking"
    },
    {
        "query": "VPBank",
        "target": "Ngân hàng TMCP Phát Hành Việt Nam",
        "category": "banking"
    },
    {
        "query": "Asia Commercial Joint Stock Bank",
        "target": "Ngân hàng TMCP Á Châu",
        "category": "banking"
    },
    {
        "query": "ACB Bank",
        "target": "Ngân hàng TMCP Á Châu",
        "category": "banking"
    },
    {
        "query": "Vietnam Export Import Commercial Joint Stock Bank",
        "target": "Ngân hàng TMCP Xuất Nhập Khẩu Việt Nam",
        "category": "banking"
    },
    {
        "query": "Eximbank",
        "target": "Ngân hàng TMCP Xuất Nhập Khẩu Việt Nam",
        "category": "banking"
    },
    {
        "query": "Joint Stock Commercial Bank for Investment and Development of Vietnam",
        "target": "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam",
        "category": "banking"
    },
    {
        "query": "BIDV",
        "target": "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam",
        "category": "banking"
    },
    {
        "query": "Vietnam International Commercial Joint Stock Bank",
        "target": "Ngân hàng TMCP Quốc Tế Việt Nam",
        "category": "banking"
    },
    {
        "query": "VIB",
        "target": "Ngân hàng TMCP Quốc Tế Việt Nam",
        "category": "banking"
    },
    {
        "query": "Saigon Thuong Tin Commercial Joint Stock Bank",
        "target": "Ngân hàng TMCP Sài Gòn Thương Tín",
        "category": "banking"
    },
    {
        "query": "Sacombank",
        "target": "Ngân hàng TMCP Sài Gòn Thương Tín",
        "category": "banking"
    },

    # Telecom
    {
        "query": "Viettel Group",
        "target": "Tập đoàn Công nghiệp - Viễn thông Quân đội",
        "category": "telecom"
    },
    {
        "query": "Military Industry and Telecoms Group",
        "target": "Tập đoàn Công nghiệp - Viễn thông Quân đội",
        "category": "telecom"
    },
    {
        "query": "Vietnam Posts and Telecommunications Group",
        "target": "Tập đoàn Bưu chính Viễn thông Việt Nam",
        "category": "telecom"
    },
    {
        "query": "VNPT",
        "target": "Tập đoàn Bưu chính Viễn thông Việt Nam",
        "category": "telecom"
    },
    {
        "query": "Mobifone Corporation",
        "target": "Tổng công ty Viễn thông MobiFone",
        "category": "telecom"
    },
    {
        "query": "Vietnam Mobile Telecom Services Company",
        "target": "Tổng công ty Viễn thông MobiFone",
        "category": "telecom"
    },

    # Aviation
    {
        "query": "Vietnam Airlines",
        "target": "Tổng công ty Hàng không Việt Nam",
        "category": "aviation"
    },
    {
        "query": "Vietnam Airline Company",
        "target": "Tổng công ty Hàng không Việt Nam",
        "category": "aviation"
    },
    {
        "query": "Vietjet Aviation Joint Stock Company",
        "target": "Công ty Cổ phần Hàng không Vietjet",
        "category": "aviation"
    },
    {
        "query": "Vietjet Air",
        "target": "Công ty Cổ phần Hàng không Vietjet",
        "category": "aviation"
    },
    {
        "query": "Bamboo Airways",
        "target": "Công ty TNHH Hàng không Tre Việt",
        "category": "aviation"
    },
    {
        "query": "Bamboo Airline",
        "target": "Công ty TNHH Hàng không Tre Việt",
        "category": "aviation"
    },

    # Energy/Oil & Gas
    {
        "query": "PetroVietnam",
        "target": "Tập đoàn Dầu khí Việt Nam",
        "category": "energy"
    },
    {
        "query": "Vietnam Oil and Gas Group",
        "target": "Tập đoàn Dầu khí Việt Nam",
        "category": "energy"
    },
    {
        "query": "Vietnam Electricity",
        "target": "Tập đoàn Điện lực Việt Nam",
        "category": "energy"
    },
    {
        "query": "EVN",
        "target": "Tập đoàn Điện lực Việt Nam",
        "category": "energy"
    },
    {
        "query": "Vietnam National Oil and Gas Group",
        "target": "Tập đoàn Dầu khí Việt Nam",
        "category": "energy"
    },

    # Real Estate/Construction
    {
        "query": "Vingroup",
        "target": "Tập đoàn Vingroup",
        "category": "real_estate"
    },
    {
        "query": "VinHomes",
        "target": "Công ty Cổ phần Vinhomes",
        "category": "real_estate"
    },
    {
        "query": "Vietnam Real Estate Group",
        "target": "Tập đoàn Vingroup",
        "category": "real_estate"
    },
    {
        "query": "Novaland Group",
        "target": "Công ty Cổ phần Tập đoàn Đầu tư Địa ốc No Va",
        "category": "real_estate"
    },
    {
        "query": "Nam Long Investment Joint Stock Company",
        "target": "Công ty Cổ phần Đầu tư Xây dựng Nam Long",
        "category": "real_estate"
    },

    # Retail/Consumer Goods
    {
        "query": "Mobile World Investment Corporation",
        "target": "Công ty Cổ phần Đầu tư Thế Giới Di Động",
        "category": "retail"
    },
    {
        "query": "MWG",
        "target": "Công ty Cổ phần Đầu tư Thế Giới Di Động",
        "category": "retail"
    },
    {
        "query": "The Gioi Di Dong",
        "target": "Công ty Cổ phần Đầu tư Thế Giới Di Động",
        "category": "retail"
    },
    {
        "query": "Dien May Xanh",
        "target": "Công ty Cổ phần Đầu tư Thế Giới Di Động",
        "category": "retail"
    },
    {
        "query": "Blue Electronics",
        "target": "Công ty Cổ phần Đầu tư Thế Giới Di Động",
        "category": "retail"
    },
    {
        "query": "Phu Nhuan Jewelry Joint Stock Company",
        "target": "Công ty Cổ phần Vàng bạc Đá quý Phú Nhuận",
        "category": "retail"
    },
    {
        "query": "PNJ",
        "target": "Công ty Cổ phần Vàng bạc Đá quý Phú Nhuận",
        "category": "retail"
    },
    {
        "query": "Phu Nhuan Jewelry",
        "target": "Công ty Cổ phần Vàng bạc Đá quý Phú Nhuận",
        "category": "retail"
    },

    # Technology
    {
        "query": "FPT Corporation",
        "target": "Công ty Cổ phần FPT",
        "category": "technology"
    },
    {
        "query": "FPT",
        "target": "Công ty Cổ phần FPT",
        "category": "technology"
    },
    {
        "query": "Finetek Company",
        "target": "Công ty Cổ phần FPT",
        "category": "technology"
    },
    {
        "query": "CMC Technology Group",
        "target": "Tập đoàn Công nghệ CMC",
        "category": "technology"
    },
    {
        "query": "CMC",
        "target": "Tập đoàn Công nghệ CMC",
        "category": "technology"
    },

    # Manufacturing/Industrial
    {
        "query": "Hoa Phat Group",
        "target": "Tập đoàn Hòa Phát",
        "category": "manufacturing"
    },
    {
        "query": "Hoa Phat Steel",
        "target": "Tập đoàn Hòa Phát",
        "category": "manufacturing"
    },
    {
        "query": "Thailand Iron and Steel",
        "target": "Tập đoàn Hòa Phát",
        "category": "manufacturing"
    },
    {
        "query": "Tien Phong Plastic JSC",
        "target": "Công ty Cổ phần Nhựa Tiền Phong",
        "category": "manufacturing"
    },
    {
        "query": "Tien Phong Plastic",
        "target": "Công ty Cổ phần Nhựa Tiền Phong",
        "category": "manufacturing"
    },

    # Transportation/Logistics
    {
        "query": "Vietnam Railway",
        "target": "Tổng công ty Đường sắt Việt Nam",
        "category": "transportation"
    },
    {
        "query": "VNR",
        "target": "Tổng công ty Đường sắt Việt Nam",
        "category": "transportation"
    },
    {
        "query": "Vietnam Shipping",
        "target": "Tổng công ty Hàng hải Việt Nam",
        "category": "transportation"
    },
    {
        "query": "Vosco",
        "target": "Tổng công ty Hàng hải Việt Nam",
        "category": "transportation"
    },

    # Insurance
    {
        "query": "Baoviet Insurance",
        "target": "Tổng công ty Bảo hiểm Việt Nam",
        "category": "insurance"
    },
    {
        "query": "Vietnam Insurance",
        "target": "Tổng công ty Bảo hiểm Việt Nam",
        "category": "insurance"
    },
    {
        "query": "Bao Minh Insurance",
        "target": "Công ty Cổ phần Bảo Minh",
        "category": "insurance"
    },
    {
        "query": "Bao Minh",
        "target": "Công ty Cổ phần Bảo Minh",
        "category": "insurance"
    },

    # Agriculture
    {
        "query": "Loc Troi Group",
        "target": "Công ty Cổ phần Lộc Trời",
        "category": "agriculture"
    },
    {
        "query": "An Giang Plant Protection Joint Stock Company",
        "target": "Công ty Cổ phần Bảo vệ Thực vật An Giang",
        "category": "agriculture"
    },
    {
        "query": "Plant Protection Company",
        "target": "Công ty Cổ phần Bảo vệ Thực vật An Giang",
        "category": "agriculture"
    },

    # Other major companies
    {
        "query": "Masan Group",
        "target": "Tập đoàn Masan",
        "category": "other"
    },
    {
        "query": "Masan Consumer",
        "target": "Công ty Cổ phần Hàng tiêu dùng Masan",
        "category": "other"
    },
    {
        "query": "Sabeco",
        "target": "Tổng công ty Cổ phần Bia - Rượu - Nước giải khát Việt Nam",
        "category": "other"
    },
    {
        "query": "Saigon Beer",
        "target": "Tổng công ty Cổ phần Bia - Rượu - Nước giải khát Việt Nam",
        "category": "other"
    },
    {
        "query": "Vinacapital",
        "target": "Công ty Cổ phần VinaCapital",
        "category": "other"
    },
]


def build_corpus_from_test_set(test_set: List[Dict]) -> List[str]:
    """Build a unique corpus from the test set targets."""
    corpus = list(set(item["target"] for item in test_set))
    return corpus


def evaluate_cross_language(
    test_set: List[Dict],
    model_name: str = 'tfidf-wordllama',
    rerank_threshold: float = 0.10,
    profile_latency: bool = False
) -> Dict:
    """
    Evaluate cross-language matching performance.

    Args:
        test_set: List of test cases with 'query', 'target', 'category'
        model_name: Model to use for matching
        rerank_threshold: Threshold for adaptive reranking
        profile_latency: If True, collect detailed latency statistics

    Returns:
        Dictionary with evaluation metrics
    """
    # Build corpus from test set
    corpus = build_corpus_from_test_set(test_set)
    print(f"Building corpus with {len(corpus)} companies...")

    # Initialize matcher
    print(f"Initializing matcher: {model_name}")
    matcher = CompanyMatcher(
        model_name=model_name,
        rerank_threshold=rerank_threshold,
        remove_stopwords=True
    )

    # Build index
    matcher.build_index(corpus)

    # Run evaluation
    results = {
        "top1_hits": 0,
        "top3_hits": 0,
        "top5_hits": 0,
        "total": len(test_set),
        "errors": [],
        "category_stats": {},
        "latencies_ms": [] if profile_latency else None,
        "rerank_count": 0
    }

    print(f"\nEvaluating {len(test_set)} queries...\n")

    for i, test_case in enumerate(test_set, 1):
        query = test_case["query"]
        target = test_case["target"]
        category = test_case["category"]

        # Time the search if profiling
        start_time = time.time()

        # Search
        predictions = matcher.search(query, top_k=5)

        # Record latency
        if profile_latency:
            latency_ms = (time.time() - start_time) * 1000
            results["latencies_ms"].append(latency_ms)

        # Check if target appears in results
        target_idx = None
        for idx, pred in enumerate(predictions):
            if pred["company"] == target:
                target_idx = idx
                break

        # Update metrics
        if target_idx is not None:
            if target_idx == 0:
                results["top1_hits"] += 1
            if target_idx < 3:
                results["top3_hits"] += 1
            if target_idx < 5:
                results["top5_hits"] += 1
        else:
            results["errors"].append({
                "query": query,
                "target": target,
                "category": category,
                "predicted": predictions[0]["company"] if predictions else None,
                "score": predictions[0]["score"] if predictions else 0.0
            })

        # Update category stats
        if category not in results["category_stats"]:
            results["category_stats"][category] = {"hits": 0, "total": 0}
        results["category_stats"][category]["total"] += 1
        if target_idx == 0:
            results["category_stats"][category]["hits"] += 1

        # Progress indicator
        if i % 20 == 0:
            current_acc = results["top1_hits"] / i
            print(f"  Processed {i}/{len(test_set)} queries | Current Top-1: {current_acc:.1%}")

    return results, matcher


def print_evaluation_report(results: Dict, model_name: str):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("CROSS-LANGUAGE COMPANY NAME MATCHING EVALUATION")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Total Queries: {results['total']}")

    # Overall metrics
    top1_acc = results["top1_hits"] / results["total"]
    top3_acc = results["top3_hits"] / results["total"]
    top5_acc = results["top5_hits"] / results["total"]

    print(f"\nOVERALL ACCURACY:")
    print(f"  Top-1:  {top1_acc:.2%} ({results['top1_hits']}/{results['total']})")
    print(f"  Top-3:  {top3_acc:.2%} ({results['top3_hits']}/{results['total']})")
    print(f"  Top-5:  {top5_acc:.2%} ({results['top5_hits']}/{results['total']})")

    # Latency statistics
    if results["latencies_ms"]:
        latencies = sorted(results["latencies_ms"])
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = sum(latencies) / len(latencies)

        print(f"\nLATENCY STATISTICS:")
        print(f"  Average: {avg:.2f} ms")
        print(f"  P50:     {p50:.2f} ms")
        print(f"  P95:     {p95:.2f} ms")
        print(f"  P99:     {p99:.2f} ms")

    # Category breakdown
    print(f"\nACCURACY BY CATEGORY:")
    for category, stats in sorted(results["category_stats"].items()):
        cat_acc = stats["hits"] / stats["total"]
        print(f"  {category:20s}: {cat_acc:.2%} ({stats['hits']}/{stats['total']})")

    # Errors
    if results["errors"]:
        print(f"\nMISMATCHES ({len(results['errors'])} total):")
        for err in results["errors"][:10]:  # Show first 10
            print(f"  - Query: '{err['query']}'")
            print(f"    Expected: {err['target']}")
            print(f"    Got:      {err['predicted']} (score: {err['score']:.3f})")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cross-language company name matching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tfidf-wordllama",
        choices=["tfidf", "wordllama-m2v-multilingual", "tfidf-wordllama", "hybrid-cross-lang"],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--rerank-threshold",
        type=float,
        default=0.10,
        help="Confidence threshold for adaptive reranking (default: 0.10)"
    )
    parser.add_argument(
        "--profile-latency",
        action="store_true",
        help="Collect detailed latency statistics"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for evaluation results (JSON)"
    )

    args = parser.parse_args()

    # Run evaluation
    results, matcher = evaluate_cross_language(
        test_set=CROSS_LANGUAGE_TEST_SET,
        model_name=args.model,
        rerank_threshold=args.rerank_threshold,
        profile_latency=args.profile_latency
    )

    # Print report
    print_evaluation_report(results, args.model)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare output data (remove latency list for JSON readability)
        output_data = {
            "model": args.model,
            "rerank_threshold": args.rerank_threshold,
            "total_queries": results["total"],
            "top1_accuracy": results["top1_hits"] / results["total"],
            "top3_accuracy": results["top3_hits"] / results["total"],
            "top5_accuracy": results["top5_hits"] / results["total"],
            "category_stats": results["category_stats"],
            "errors": results["errors"]
        }

        if results["latencies_ms"]:
            latencies = sorted(results["latencies_ms"])
            output_data["latency"] = {
                "avg_ms": sum(latencies) / len(latencies),
                "p50_ms": latencies[len(latencies) // 2],
                "p95_ms": latencies[int(len(latencies) * 0.95)],
                "p99_ms": latencies[int(len(latencies) * 0.99)]
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
