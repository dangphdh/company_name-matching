"""
Analyze failure cases of TF-IDF matcher to identify error patterns
and propose improvement directions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, re, math, unicodedata, collections
from difflib import SequenceMatcher
import numpy as np

from src.matching.matcher import CompanyMatcher
from src.preprocess import clean_company_name, remove_accents

# ─────────────────────────── helpers ───────────────────────────

def load_eval_data(max_queries=5000, seed=42):
    corpus, corp_id_to_name = [], {}
    with open('data/eval/corpus.jsonl', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item['name'])
            corp_id_to_name[item['id']] = item['name']

    queries = []
    with open('data/eval/queries.jsonl', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))

    import random
    random.seed(seed)
    if len(queries) > max_queries:
        queries = random.sample(queries, max_queries)

    return corpus, queries, corp_id_to_name


def is_no_accent(text):
    """Check if text contains no Vietnamese diacritics."""
    return text == remove_accents(text)


def is_all_upper(text):
    alpha = [c for c in text if c.isalpha()]
    return bool(alpha) and all(c.isupper() for c in alpha)


def is_abbreviated(target_name, query_text):
    """
    Detect if query looks like an abbreviation of the target.
    Heuristic: query is much shorter and shares initials.
    """
    q_words = query_text.strip().split()
    t_words = target_name.strip().split()
    if len(q_words) == 0 or len(t_words) == 0:
        return False
    # Single token that looks like initials: e.g. "VCB" for "Vietcombank"
    if len(q_words) == 1 and len(q_words[0]) <= 6 and len(t_words) >= 2:
        return True
    # Query ≤ half the words of target and query is all-upper
    if len(q_words) <= len(t_words) / 2 and is_all_upper(query_text):
        return True
    return False


def token_overlap(a, b):
    """Jaccard token overlap ratio."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def char_sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def categorize_query(query_text, target_name, method):
    """
    Return a list of applicable error category tags for a failed query.
    """
    tags = []
    q = query_text.strip()

    # ── method tag ──────────────────────────────────────────────
    tags.append(f"method:{method}")

    # ── accent ──────────────────────────────────────────────────
    if is_no_accent(q):
        tags.append("no_accent")
    else:
        tags.append("accented")

    # ── case ────────────────────────────────────────────────────
    if is_all_upper(q):
        tags.append("all_upper")
    elif q == q.lower():
        tags.append("all_lower")
    else:
        tags.append("mixed_case")

    # ── length ──────────────────────────────────────────────────
    words = q.split()
    if len(words) == 1:
        tags.append("single_token")
    elif len(words) <= 3:
        tags.append("short_query")
    else:
        tags.append("long_query")

    # ── abbreviation ────────────────────────────────────────────
    if is_abbreviated(target_name, q):
        tags.append("abbreviated")

    # ── word-order swap ─────────────────────────────────────────
    q_clean = remove_accents(q).lower()
    t_clean = remove_accents(target_name).lower()
    q_tokens = set(q_clean.split())
    t_tokens = set(t_clean.split())
    if (len(q_tokens) == len(t_tokens) and q_tokens == t_tokens
            and q_clean != t_clean):
        tags.append("word_reorder")

    # ── token overlap with target ────────────────────────────────
    overlap = token_overlap(q, target_name)
    if overlap == 0.0:
        tags.append("no_token_overlap")
    elif overlap < 0.3:
        tags.append("low_token_overlap")

    return tags


# ─────────────────────────── collect failures ───────────────────

def collect_failures(matcher, queries, corp_id_to_name, top_k=5):
    failures, successes = [], []
    for query in queries:
        q_text = query['text']
        target_name = corp_id_to_name[query['target_id']]
        results = matcher.search(q_text, top_k=top_k)
        predicted = [r['company'] for r in results]
        scores = [r['score'] for r in results]

        entry = {
            'query': q_text,
            'method': query.get('method', 'unknown'),
            'target': target_name,
            'predicted': predicted,
            'scores': scores,
        }
        if predicted and predicted[0] == target_name:
            successes.append(entry)
        else:
            entry['top1_wrong'] = predicted[0] if predicted else None
            entry['top1_score'] = scores[0] if scores else 0.0
            # score the correct answer if it appears in top_k
            entry['target_in_topk'] = target_name in predicted
            entry['target_rank'] = (predicted.index(target_name) + 1
                                    if target_name in predicted else None)
            failures.append(entry)

    return failures, successes


# ─────────────────────────── analysis ───────────────────────────

def analyze_failures(failures, successes):
    total = len(failures) + len(successes)

    # ── 1. Tag distribution ──────────────────────────────────────
    tag_counter = collections.Counter()
    for f in failures:
        for tag in categorize_query(f['query'], f['target'], f['method']):
            tag_counter[tag] += 1

    # ── 2. Method breakdown ───────────────────────────────────────
    method_fail = collections.Counter(f['method'] for f in failures)
    method_all = collections.Counter()
    for e in failures + successes:
        method_all[e['method']] += 1
    method_error_rate = {
        m: method_fail[m] / method_all[m] for m in method_all
    }

    # ── 3. Score gap for failures ─────────────────────────────────
    top1_scores = [f['top1_score'] for f in failures]
    topk_hits = [f for f in failures if f['target_in_topk']]

    # ── 4. Longest common prefix / suffix analysis ────────────────
    confusable_pairs = []  # (target, wrong_pred, query)
    for f in failures:
        if f['top1_wrong']:
            confusable_pairs.append((f['target'], f['top1_wrong'], f['query']))

    # ── 5. Clean-token overlap for failures ──────────────────────
    clean_overlaps = []
    for f in failures:
        q_clean = clean_company_name(f['query'], remove_stopwords=True)
        t_clean = clean_company_name(f['target'], remove_stopwords=True)
        if t_clean:
            overlap = token_overlap(q_clean, t_clean)
            clean_overlaps.append(overlap)

    # ── 6. Very-short target names ────────────────────────────────
    short_target_fails = [f for f in failures
                          if len(f['target'].split()) <= 3]

    # ── 7. Confusable company prefix (many companies share prefix) ─
    target_clean_map = {}
    for f in failures:
        t = clean_company_name(f['target'], remove_stopwords=True)
        target_clean_map[t] = target_clean_map.get(t, 0) + 1

    return {
        'total': total, 'n_fail': len(failures), 'n_succ': len(successes),
        'tag_counter': tag_counter,
        'method_error_rate': method_error_rate, 'method_fail': method_fail,
        'top1_scores': top1_scores,
        'topk_hits_n': len(topk_hits),
        'confusable_pairs': confusable_pairs,
        'clean_overlaps': clean_overlaps,
        'short_target_fails': short_target_fails,
    }


def print_report(stats, failures, top_examples=8):
    n, nf, ns = stats['total'], stats['n_fail'], stats['n_succ']
    print("\n" + "=" * 70)
    print(f"ERROR ANALYSIS REPORT  ({n} queries, {nf} failures, {ns} correct)")
    print(f"Overall Top-1 accuracy: {ns/n*100:.2f}%")
    print("=" * 70)

    # ── Per-method ───────────────────────────────────────────────
    print("\n── Error rate by query method ──────────────────────────────────")
    for m, rate in sorted(stats['method_error_rate'].items(),
                          key=lambda x: -x[1]):
        total_m = stats['method_fail'][m] + \
                  sum(1 for e in [{'method': m}]
                      if stats['method_fail'].get(e['method'], 0) == 0)
        print(f"  {m:25s}  error={rate*100:5.1f}%  (failed {stats['method_fail'][m]})")

    # ── Tag distribution ─────────────────────────────────────────
    print("\n── Failure tags (sorted by frequency) ─────────────────────────")
    for tag, cnt in stats['tag_counter'].most_common(20):
        pct = cnt / nf * 100
        print(f"  {tag:30s}  {cnt:4d}  ({pct:.1f}% of failures)")

    # ── Score distribution ───────────────────────────────────────
    s = np.array(stats['top1_scores'])
    print(f"\n── Top-1 score distribution for failures ───────────────────────")
    print(f"  mean={s.mean():.3f}  median={np.median(s):.3f}  "
          f"min={s.min():.3f}  max={s.max():.3f}")
    buckets = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    for lo, hi in buckets:
        cnt = np.sum((s >= lo) & (s < hi))
        print(f"  [{lo:.1f}, {hi:.2f})  {cnt:4d}  ({cnt/nf*100:.1f}%)")

    print(f"\n  Target in top-5 (but not top-1): {stats['topk_hits_n']} "
          f"({stats['topk_hits_n']/nf*100:.1f}% of failures)")

    # ── Clean overlap ────────────────────────────────────────────
    ov = np.array(stats['clean_overlaps'])
    print(f"\n── Token overlap (clean query vs clean target) for failures ────")
    print(f"  mean={ov.mean():.3f}  zero_overlap={np.sum(ov==0)/nf*100:.1f}%")

    # ── Short target names ───────────────────────────────────────
    short = stats['short_target_fails']
    print(f"\n── Failures on short targets (≤3 words): {len(short)} "
          f"({len(short)/nf*100:.1f}%)")

    # ── Confusable pair analysis: do wrong pred & target share prefix?─
    cp = stats['confusable_pairs']
    shared_prefix_3 = sum(1 for t, w, q in cp
                          if remove_accents(t).lower()[:3] ==
                          remove_accents(w).lower()[:3])
    print(f"\n── Wrong prediction shares 3-char prefix with target: "
          f"{shared_prefix_3}/{len(cp)} ({shared_prefix_3/len(cp)*100:.1f}%)")

    # ── Sample failures ──────────────────────────────────────────
    print(f"\n── Sample failure cases ────────────────────────────────────────")

    # Group by interesting sub-categories
    categories = {
        "no_accent": [f for f in failures if is_no_accent(f['query'])],
        "all_upper": [f for f in failures if is_all_upper(f['query'])],
        "abbreviated": [f for f in failures
                        if is_abbreviated(f['target'], f['query'])],
        "llm_typo": [f for f in failures if f['method'] == 'llm_typo'],
        "llm_english": [f for f in failures if f['method'] == 'llm_english'],
        "llm_abbreviated": [f for f in failures
                            if f['method'] == 'llm_abbreviated'],
        "no_token_overlap": [f for f in failures
                             if token_overlap(f['query'], f['target']) == 0.0],
        "target_in_top5": [f for f in failures if f['target_in_topk']],
    }

    for cat, cases in categories.items():
        if not cases:
            continue
        print(f"\n  [{cat}] — {len(cases)} failures")
        for f in cases[:top_examples]:
            rank_str = (f"rank={f['target_rank']}" if f['target_rank']
                        else "not in top-5")
            print(f"    query   : {f['query']}")
            print(f"    target  : {f['target']}")
            print(f"    top1    : {f['top1_wrong']}  (score={f['top1_score']:.3f})")
            print(f"    ({rank_str})")
            print()

    # ── Diagnosis + proposed fixes ───────────────────────────────
    _print_diagnosis(stats, nf)


def _print_diagnosis(stats, nf):
    tc = stats['tag_counter']
    er = stats['method_error_rate']

    print("\n" + "=" * 70)
    print("DIAGNOSIS & PROPOSED IMPROVEMENTS")
    print("=" * 70)

    issues = []

    # Issue 1: no_accent dominates
    no_acc_pct = tc.get('no_accent', 0) / nf * 100
    if no_acc_pct > 20:
        issues.append((
            no_acc_pct,
            "HIGH NO-ACCENT FAILURE RATE",
            f"{no_acc_pct:.1f}% of failures are no-accent queries.",
            [
                "Add a no-accent → accent normalization step before search "
                "(map common patterns like 'a'->'ă'/'â', 'o'->'ô'/'ơ', 'u'->'ư').",
                "Expand no-accent index variants: store multiple accent "
                "reconstructions for ambiguous tokens.",
                "Add n-gram range (1,4) word-level TF-IDF alongside char n-gram "
                "to capture whole-word unaccented matches.",
            ]
        ))

    # Issue 2: abbreviated queries
    abbr_pct = tc.get('abbreviated', 0) / nf * 100
    if abbr_pct > 5:
        issues.append((
            abbr_pct,
            "ABBREVIATION FAILURES",
            f"{abbr_pct:.1f}% of failures involve abbreviated queries.",
            [
                "Generate abbreviation expansion at index time: add "
                "'TNHH ABC' → 'tnhh', 'abc', 'ta', 'ta dv' char-level variants.",
                "Include initials index: 'Ban Quản Lý Dự Án' → 'bqlda' "
                "as an extra indexed token.",
                "Fine-tune TF-IDF sublinear_tf and min_df for short queries.",
            ]
        ))

    # Issue 3: top-5 contains answer (ranking problem)
    topk_pct = stats['topk_hits_n'] / nf * 100
    if topk_pct > 30:
        issues.append((
            topk_pct,
            "RANKING PROBLEM (correct answer in top-5 but not top-1)",
            f"{topk_pct:.1f}% of failures have the correct answer in top-5.",
            [
                "Use tfidf-rerank(n=5) + BGE-M3: already proven (+0-1% Top-1, "
                "+0.1% Top-3 at same latency).",
                "Tune char n-gram range: try (1,4) or (2,4) to reduce dimensionality noise.",
                "Add BM25 word-level re-scorer for candidate set expansion.",
                "Use cross-encoder (e.g., BAAI/bge-reranker-base) for top-5 reranking.",
            ]
        ))

    # Issue 4: typo failures
    typo_rate = er.get('llm_typo', 0)
    if typo_rate > 0.15:
        issues.append((
            typo_rate * 100,
            "TYPO ROBUSTNESS",
            f"llm_typo error rate = {typo_rate*100:.1f}%",
            [
                "Add fuzzy pre-processing: correct obvious single-char substitutions "
                "before vectorization.",
                "Add augmented training examples with common OCR/keyboard typos to "
                "the index (e.g., 'l'→'1', 'o'→'0').",
                "Use lower n in char n-gram (e.g., start at 1 instead of 2) to "
                "capture single-char errors.",
            ]
        ))

    # Issue 5: English queries
    eng_rate = er.get('llm_english', 0)
    if eng_rate > 0.15:
        issues.append((
            eng_rate * 100,
            "ENGLISH/TRANSLATED QUERY FAILURES",
            f"llm_english error rate = {eng_rate*100:.1f}%",
            [
                "Maintain an English alias field in the corpus and add it to index.",
                "Use multilingual BGE-M3 for English queries — already supported "
                "by 'tfidf-rerank' fusion.",
                "Add an optional machine-translation step for detected English queries.",
            ]
        ))

    # Issue 6: zero token overlap
    zero_ov_pct = tc.get('no_token_overlap', 0) / nf * 100
    if zero_ov_pct > 10:
        issues.append((
            zero_ov_pct,
            "ZERO TOKEN OVERLAP (semantic gap)",
            f"{zero_ov_pct:.1f}% of failures share NO tokens with the target.",
            [
                "These require semantic/embedding matching. Use BGE-M3 dense search "
                "as primary when TF-IDF score < threshold.",
                "Build separate brand alias dictionary (e.g. 'Vinamilk' → "
                "'Công ty TNHH Sữa Việt Nam').",
                "Expand stopword removal to also strip province/city names that "
                "flood the vocabulary.",
            ]
        ))

    # Sort by severity
    issues.sort(key=lambda x: -x[0])

    for rank, (pct, title, desc, fixes) in enumerate(issues, 1):
        print(f"\n{'─'*70}")
        print(f"Issue #{rank}: {title}")
        print(f"  Severity: {pct:.1f}%")
        print(f"  Description: {desc}")
        print(f"  Proposed fixes:")
        for i, fix in enumerate(fixes, 1):
            print(f"    {i}. {fix}")

    print("\n" + "=" * 70)
    print("QUICK WIN PRIORITY LIST")
    print("=" * 70)
    print("""
  1. [IMMEDIATE] run stopwords=False evaluation — prior results show
     76.1% vs 66.2% Top-1. Most 'no_accent/all_upper' failures disappear
     because stop words like 'ban', 'quan', 'ly' are signal words here.

  2. [IMMEDIATE] n-gram range tuning — try (1,5) or (2,4) in TfidfVectorizer
     to capture full-word matches and reduce char-level noise.

  3. [SHORT TERM] Build acronym/initial index — for each company generate
     the initials token (e.g. "BQLDA") and add it as a searchable variant.

  4. [SHORT TERM] Score-based fallback — if TF-IDF top-1 score < 0.5,
     automatically route to BGE-M3 dense search.

  5. [MEDIUM TERM] Cross-encoder reranking — replace BGE-M3 dot-product
     reranker with a cross-encoder (BAAI/bge-reranker-base) for top-5
     candidates; expected +2-4% Top-1 from the literature.

  6. [MEDIUM TERM] English alias table — maintain a JSON mapping of
     common English brand names → Vietnamese legal names and do exact
     lookup before TF-IDF.
""")


# ─────────────────────────── main ───────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-queries', type=int, default=5000)
    parser.add_argument('--remove-stopwords', type=lambda x: x.lower() != 'false',
                        default=True)
    args = parser.parse_args()

    print(f"Loading eval data (max {args.max_queries} queries)...")
    corpus, queries, corp_id_to_name = load_eval_data(
        max_queries=args.max_queries)
    print(f"  Corpus: {len(corpus)} companies | Queries: {len(queries)}")

    print(f"\nBuilding TF-IDF index (remove_stopwords={args.remove_stopwords})...")
    matcher = CompanyMatcher(model_name='tfidf',
                             remove_stopwords=args.remove_stopwords)
    matcher.build_index(corpus)

    print("Running inference...")
    failures, successes = collect_failures(matcher, queries, corp_id_to_name,
                                           top_k=5)

    stats = analyze_failures(failures, successes)
    print_report(stats, failures)
