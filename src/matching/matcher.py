import re
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_company_name, remove_accents


def _sigmoid(x):
    """Sigmoid to map cross-encoder logits → (0, 1) probability."""
    return 1.0 / (1.0 + math.exp(-float(x)))

# Entity-type tokens that are meaningful discriminators between sibling companies.
# Ordered by specificity so the first match wins.
_ENTITY_TYPE_TOKENS = ('vpdd', 'cn', 'td', 'htx', 'hd', 'tnhh', 'cp', 'mtv')

def _extract_entity_type(cleaned_name: str):
    """Return the first entity-type token found in a cleaned company name, or None."""
    tokens = set(cleaned_name.split())
    for et in _ENTITY_TYPE_TOKENS:
        if et in tokens:
            return et
    return None


def _has_repeated_tokens(name: str) -> bool:
    """Return True if the cleaned name contains any consecutive repeated token
    or consecutive repeated bigram (e.g. 'san xuat san xuat')."""
    tokens = name.split()
    # Check repeated unigrams
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1] and len(tokens[i]) > 1:
            return True
    # Check repeated bigrams
    for i in range(len(tokens) - 3):
        if tokens[i] == tokens[i + 2] and tokens[i + 1] == tokens[i + 3]:
            return True
    return False


def _rrf_fuse(score_lists, k=60):
    """
    Reciprocal Rank Fusion over a list of score arrays.
    Each score array is sorted descending to get ranks, then fused.
    Returns a combined score array of the same length.
    """
    n = len(score_lists[0])
    fused = np.zeros(n)
    for scores in score_lists:
        # rank[i] = position (0-based) of element i when sorted descending
        order = np.argsort(scores)[::-1]  # indices sorted by score desc
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        fused += 1.0 / (k + rank + 1)
    return fused


class CompanyMatcher:
    def __init__(self, model_name='tfidf', use_gpu=False, remove_stopwords=True,
                 tfidf_weight=0.5, bm25_weight=0.5,
                 dense_model_name='BAAI/bge-m3',
                 sparse_weight=0.5, dense_weight=0.5,
                 fusion='weighted',
                 rerank_n=10,
                 rerank_threshold=0.05,
                 lsa_dims=512,
                 max_features=100000):
        """
        Khởi tạo mô hình matching.

        Args:
            model_name: 'tfidf', 'bm25', 'tfidf-bm25' (sparse);
                        'tfidf-dense', 'bm25-dense' (sparse+dense);
                        'wordllama-*', '<hf-path>' (dense-only)
            use_gpu: Use GPU for dense encoders
            remove_stopwords: Remove stopwords during preprocessing
            tfidf_weight / bm25_weight: Weights for tfidf-bm25 hybrid
            dense_model_name: SentenceTransformer model for sparse-dense hybrids
            sparse_weight / dense_weight: Weights for sparse-dense hybrid
            fusion: Score fusion strategy for tfidf-dense / bm25-dense:
                'weighted'        — weighted sum of sparse + dense scores
                'rrf'             — Reciprocal Rank Fusion
                'tfidf-rerank'    — TF-IDF retrieves rerank_n candidates, dense reranks
                'union-rerank'    — TF-IDF top-N ∪ dense top-N, reranked by dense
                'adaptive-rerank' — TF-IDF first; if gap(top1, top2) < rerank_threshold
                                    rerank top-rerank_n candidates with dense
                'cross-rerank'    — TF-IDF retrieves rerank_n candidates, CrossEncoder
                                    (e.g. BAAI/bge-reranker-base) rescores each pair
            rerank_n: Candidates retrieved per retriever for rerank strategies
            rerank_threshold: Score gap below which adaptive-rerank triggers (default 0.05)
            lsa_dims: Number of LSA (TruncatedSVD) dimensions for 'tfidf-lsa' model.
                      512 is the recommended value for 2.4M-scale corpora where
                      storing dense TF-IDF vectors (~2.5 TB) is not feasible.
                      Ignored for all other model_name values.
            max_features: Maximum vocabulary size for TF-IDF vectorization.
                         Reduces memory usage for LSA models by limiting sparse matrix size.
                         Default 100000 (reduces memory from ~20GB to ~11GB for 2.4M companies).
                         Set to None for unlimited vocabulary (may cause OOM on large corpora).
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.remove_stopwords = remove_stopwords
        self.tfidf_weight = tfidf_weight
        self.max_features = max_features
        self.bm25_weight = bm25_weight
        self.dense_model_name = dense_model_name
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.fusion = fusion
        self.rerank_n = rerank_n
        self.rerank_threshold = rerank_threshold
        self.lsa_dims = lsa_dims
        self.index = None
        self.corpus_names = []
        self.corpus_vectors = None
        
        if model_name in ('tfidf-lsa', 'lsa'):
            print(f"Using TF-IDF + LSA (TruncatedSVD k={lsa_dims}) Matcher...")
            self.svd = None  # fitted in build_index
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1,
                max_features=self.max_features
            )
        elif model_name == 'tfidf' or model_name == 'tfidf-char-ngram':
            print(f"Using Pure Python/Sklearn TF-IDF Matcher...")
            self.vectorizer = TfidfVectorizer(
                analyzer='char', 
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1
            )
        elif model_name == 'tfidf-bm25' or model_name == 'hybrid':
            print(f"Using Hybrid TF-IDF + BM25 Matcher (TF-IDF: {tfidf_weight}, BM25: {bm25_weight})...")
            self.vectorizer = TfidfVectorizer(
                analyzer='char', 
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1
            )
            self.bm25_model = None  # Will be initialized in build_index
        elif model_name == 'tfidf-dense':
            print(f"Using Hybrid TF-IDF + Dense ({dense_model_name}) Matcher "
                  f"[fusion={fusion}"
                  + (f", rerank_n={rerank_n}]..." if 'rerank' in fusion
                     else f", sparse={sparse_weight}, dense={dense_weight}]..."))
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1
            )
            if fusion == 'cross-rerank':
                from sentence_transformers import CrossEncoder
                print(f"  Loading CrossEncoder: {dense_model_name}...")
                self.cross_encoder = CrossEncoder(
                    dense_model_name, device='cuda' if use_gpu else 'cpu'
                )
            else:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(
                    dense_model_name, device='cuda' if use_gpu else 'cpu'
                )
        elif model_name == 'bm25-dense':
            print(f"Using Hybrid BM25 + Dense ({dense_model_name}) Matcher "
                  f"[fusion={fusion}, sparse={sparse_weight}, dense={dense_weight}]...")
            self.bm25_model = None  # Will be initialized in build_index
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(
                dense_model_name, device='cuda' if use_gpu else 'cpu'
            )
        elif model_name == 'tfidf-wordllama' or model_name == 'hybrid-cross-lang':
            print(f"Using Hybrid TF-IDF + WordLlama Matcher (config=m2v_multilingual)...")
            # TF-IDF vectorizer (for fast retrieval)
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1
            )
            # WordLlama will be loaded in build_index()
            self.wl = None
        elif 'wordllama' in model_name.lower():
            from wordllama import WordLlama
            print(f"Using WordLlama Matcher ({model_name})...")
            # Trích xuất config name nếu có (vd: wordllama-l2 -> l2_supercat)
            config = "l2_supercat" # default
            if 'l3' in model_name.lower():
                config = "l3_supercat"
            elif 'l2' in model_name.lower():
                config = "l2_supercat"
            elif 'm2v' in model_name.lower():
                # Handle model2vec configs
                for m2v_config in ['m2v_multilingual', 'm2v_glove', 'potion_base_8m']:
                    if m2v_config in model_name.lower() or m2v_config.replace('_', '-') in model_name.lower():
                        self.wl = WordLlama.load_m2v(config=m2v_config)
                        break
                else:
                    self.wl = WordLlama.load_m2v(config="m2v_multilingual")
                return

            self.wl = WordLlama.load(config=config)
        elif 'sbert' in model_name.lower() or ('/' in model_name and 'bm25' not in model_name.lower()):
            # Support for SentenceTransformer models (HuggingFace)
            from sentence_transformers import SentenceTransformer
            print(f"Using SentenceTransformer Matcher ({model_name})...")
            self.st_model = SentenceTransformer(model_name, device='cuda' if self.use_gpu else 'cpu')
        elif model_name.lower() == 'bm25':
            # Support for BM25 ranking
            from rank_bm25 import BM25Okapi
            print(f"Using BM25 Matcher...")
            self.bm25_model = None  # Will be initialized in build_index
        else:
            # Fallback hoặc nếu muốn dùng model khác từ HuggingFace (cần transformer)
            print(f"Warning: Model {model_name} not explicitly handled for WordLlama/SentenceTransformer/BM25/TF-IDF. Defaulting to TF-IDF.")
            self.model_name = 'tfidf'
            self.vectorizer = TfidfVectorizer(
                analyzer='char', 
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1
            )

    def build_index(self, names):
        """
        Xây dựng chỉ mục tìm kiếm từ danh sách tên công ty hệ thống.
        """
        self.corpus_names = names
        processed_names = []
        self.name_mapping = []

        # Warn about corpus entries with consecutive duplicate tokens — these
        # inflate char n-gram similarity for unrelated queries.
        bad = [n for n in names if _has_repeated_tokens(
            remove_accents(clean_company_name(n, remove_stopwords=False)))]
        if bad:
            print(f"  [WARN] {len(bad)} corpus entries have repeated consecutive tokens "
                  f"(e.g. 'SẢN XUẤT SẢN XUẤT'). They will be penalised at search time.")

        # norm_key = remove_accents(cleaned).  Maps to all original indices that
        # produce the same normalised form so that near-duplicate corpus entries
        # (e.g. "XUẤT NHẬP KHẨU" vs "XNK" after normalisation) are returned
        # together when the query matches either one.
        self._norm_to_originals = {}   # norm_key → [original_idx, ...]
        self._orig_to_norm   = {}      # original_idx → norm_key

        for i, name in enumerate(names):
            cleaned = clean_company_name(name, remove_stopwords=self.remove_stopwords)
            no_accent = remove_accents(cleaned)

            processed_names.append(cleaned)
            self.name_mapping.append(i)

            if cleaned != no_accent:
                processed_names.append(no_accent)
                self.name_mapping.append(i)

            # group by no-accent canonical key
            key = no_accent
            self._orig_to_norm[i] = key
            self._norm_to_originals.setdefault(key, []).append(i)

        print(f"Vectorizing {len(processed_names)} variants for {len(names)} companies...")

        if self.model_name in ('tfidf-lsa', 'lsa'):
            from sklearn.decomposition import TruncatedSVD
            # Step 1: sparse TF-IDF (only ~15 GB for 2.4M at 262K dims — OK as sparse)
            tfidf_sparse = self.vectorizer.fit_transform(processed_names)
            # Step 2: LSA — sparse 262K dims → dense lsa_dims dims
            self.svd = TruncatedSVD(n_components=self.lsa_dims, random_state=42)
            lsa_dense = self.svd.fit_transform(tfidf_sparse)  # (n_variants, lsa_dims)
            # L2-normalize so cosine similarity = dot product
            norms = np.linalg.norm(lsa_dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.corpus_vectors = (lsa_dense / norms).astype(np.float32)
            explained = float(self.svd.explained_variance_ratio_.sum())
            print(f"  LSA {self.lsa_dims} dims: {explained:.1%} of TF-IDF variance explained.")
            print(f"  Corpus vector matrix: {self.corpus_vectors.shape}, "
                  f"{self.corpus_vectors.nbytes / 1e9:.2f} GB")
        elif self.model_name == 'tfidf':
            self.corpus_vectors = self.vectorizer.fit_transform(processed_names)
        elif self.model_name == 'tfidf-bm25' or self.model_name == 'hybrid':
            # Hybrid: Build both TF-IDF and BM25 indices
            self.corpus_vectors = self.vectorizer.fit_transform(processed_names)
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.split() for doc in processed_names]
            self.bm25_model = BM25Okapi(tokenized_corpus)
        elif self.model_name == 'tfidf-dense':
            # Sparse: TF-IDF char n-gram (always)
            self.corpus_vectors = self.vectorizer.fit_transform(processed_names)
            # Dense: sentence embedding (not needed for cross-rerank — CE scores at query time)
            if self.fusion != 'cross-rerank':
                print(f"  Encoding dense embeddings for {len(processed_names)} variants...")
                self.dense_vectors = self.st_model.encode(
                    processed_names, batch_size=512,
                    show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
                )
            else:
                # Store processed names so cross-encoder can look up cleaned forms
                self._processed_names = processed_names
        elif self.model_name == 'tfidf-wordllama' or self.model_name == 'hybrid-cross-lang':
            # Stage 1: Build TF-IDF index (sparse, fast)
            self.corpus_vectors = self.vectorizer.fit_transform(processed_names)

            # Stage 2: Build WordLlama embeddings (dense, semantic)
            from wordllama import WordLlama
            print(f"  Building WordLlama multilingual embeddings...")
            self.wl = WordLlama.load_m2v(config="m2v_multilingual")

            # Embed only original names (not variants) to save memory
            self.wordllama_vectors = self.wl.embed(names)
            print(f"  WordLlama vectors: {self.wordllama_vectors.shape}")
        elif self.model_name == 'bm25-dense':
            # Sparse: BM25
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.split() for doc in processed_names]
            self.bm25_model = BM25Okapi(tokenized_corpus)
            # Dense: sentence embedding
            print(f"  Encoding dense embeddings for {len(processed_names)} variants...")
            self.dense_vectors = self.st_model.encode(
                processed_names, batch_size=512,
                show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
            )
        elif 'wordllama' in self.model_name:
            # WordLlama trả về numpy array
            self.corpus_vectors = self.wl.embed(processed_names)
        elif hasattr(self, 'st_model'):
            # SentenceTransformer
            self.corpus_vectors = self.st_model.encode(processed_names, convert_to_numpy=True)
        elif self.model_name == 'bm25':
            # BM25 tokenization (word-level)
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.split() for doc in processed_names]
            self.bm25_model = BM25Okapi(tokenized_corpus)
            
        print(f"{self.model_name.upper()} Index built successfully.")

    def search(self, query, top_k=5, min_score=0.0):
        """
        Tìm kiếm công ty phù hợp nhất cho 1 query.

        Args:
            query: Input query string.
            top_k: Maximum number of results to return.
            min_score: Confidence threshold. Results whose top-1 score is below
                       this value are suppressed entirely (returns [] for that
                       query). Use to trade coverage for precision — only return
                       an answer when the model is sufficiently confident.
                       Default 0.0 (no filtering, backward-compatible).
        """
        query_cleaned = clean_company_name(query, remove_stopwords=self.remove_stopwords)

        if self.model_name in ('tfidf-lsa', 'lsa'):
            # TF-IDF sparse → LSA dense → L2-normalize → dot product = cosine similarity
            tfidf_q = self.vectorizer.transform([query_cleaned])
            lsa_q = self.svd.transform(tfidf_q).astype(np.float32)  # (1, lsa_dims)
            norm_q = np.linalg.norm(lsa_q)
            if norm_q > 0:
                lsa_q = lsa_q / norm_q
            similarities = (self.corpus_vectors @ lsa_q.T).flatten()
        elif self.model_name == 'tfidf':
            query_vec = self.vectorizer.transform([query_cleaned])
            similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        elif self.model_name == 'tfidf-bm25' or self.model_name == 'hybrid':
            query_vec = self.vectorizer.transform([query_cleaned])
            tfidf_scores = cosine_similarity(query_vec, self.corpus_vectors).flatten()
            
            tokenized_query = query_cleaned.split()
            bm25_scores_raw = np.array(self.bm25_model.get_scores(tokenized_query))
            max_bm25 = bm25_scores_raw.max()
            bm25_scores = bm25_scores_raw / max_bm25 if max_bm25 > 0 else bm25_scores_raw
            
            similarities = (self.tfidf_weight * tfidf_scores +
                            self.bm25_weight * bm25_scores)
        elif self.model_name == 'tfidf-dense':
            # ── Sparse pass (always) ──────────────────────────────────────────
            query_vec = self.vectorizer.transform([query_cleaned])
            sparse_scores = cosine_similarity(query_vec, self.corpus_vectors).flatten()

            def _get_dense_scores():
                """Lazy: encode query and return dense similarity array."""
                qd = self.st_model.encode(
                    [query_cleaned], convert_to_numpy=True, normalize_embeddings=True
                )
                ds = (self.dense_vectors @ qd.T).flatten()
                return np.clip(ds, 0, 1)

            if self.fusion == 'cross-rerank':
                # Stage 1: TF-IDF retrieves rerank_n unique companies
                sparse_indices = np.argsort(sparse_scores)[::-1]
                cand_oids = []
                seen_ce = set()
                for idx in sparse_indices:
                    oid = self.name_mapping[idx]
                    if oid not in seen_ce:
                        seen_ce.add(oid)
                        cand_oids.append(oid)
                    if len(cand_oids) >= self.rerank_n:
                        break

                # Stage 2: CrossEncoder scores (query_cleaned, candidate_cleaned) pairs
                cand_cleaned = [
                    remove_accents(clean_company_name(
                        self.corpus_names[oid],
                        remove_stopwords=self.remove_stopwords
                    ))
                    for oid in cand_oids
                ]
                pairs = [(query_cleaned, c) for c in cand_cleaned]
                ce_logits = self.cross_encoder.predict(pairs)
                # Sigmoid-normalise logits → (0, 1)
                ce_norm = {oid: _sigmoid(s)
                           for oid, s in zip(cand_oids, ce_logits)}

                # Build similarities array: -inf for non-candidates
                similarities = np.full(len(sparse_scores), -np.inf)
                oid_to_max = {}
                for idx in range(len(sparse_scores)):
                    oid = self.name_mapping[idx]
                    if oid in ce_norm:
                        s = ce_norm[oid]
                        if s > oid_to_max.get(oid, -np.inf):
                            oid_to_max[oid] = s
                            similarities[idx] = s

            elif self.fusion == 'tfidf-rerank':
                dense_scores = _get_dense_scores()
                # Stage 1: TF-IDF retrieves rerank_n unique companies
                sparse_indices = np.argsort(sparse_scores)[::-1]
                seen = set()
                for idx in sparse_indices:
                    seen.add(self.name_mapping[idx])
                    if len(seen) >= self.rerank_n:
                        break
                # Stage 2: best dense score per candidate
                cand_dense = {}
                for idx, score in enumerate(dense_scores):
                    oid = self.name_mapping[idx]
                    if oid in seen:
                        cand_dense[oid] = max(cand_dense.get(oid, -1.0), score)
                similarities = np.zeros(len(sparse_scores))
                for idx in range(len(sparse_scores)):
                    oid = self.name_mapping[idx]
                    if oid in cand_dense:
                        similarities[idx] = cand_dense[oid]

            elif self.fusion == 'union-rerank':
                dense_scores = _get_dense_scores()
                # TF-IDF top-N
                sparse_indices = np.argsort(sparse_scores)[::-1]
                tfidf_cands = set()
                for idx in sparse_indices:
                    tfidf_cands.add(self.name_mapping[idx])
                    if len(tfidf_cands) >= self.rerank_n:
                        break
                # Dense top-N
                dense_indices = np.argsort(dense_scores)[::-1]
                dense_cands = set()
                for idx in dense_indices:
                    dense_cands.add(self.name_mapping[idx])
                    if len(dense_cands) >= self.rerank_n:
                        break
                union_cands = tfidf_cands | dense_cands
                cand_dense = {}
                for idx, score in enumerate(dense_scores):
                    oid = self.name_mapping[idx]
                    if oid in union_cands:
                        cand_dense[oid] = max(cand_dense.get(oid, -1.0), score)
                similarities = np.zeros(len(sparse_scores))
                for idx in range(len(sparse_scores)):
                    oid = self.name_mapping[idx]
                    if oid in cand_dense:
                        similarities[idx] = cand_dense[oid]

            elif self.fusion == 'adaptive-rerank':
                # Stage 1: TF-IDF — collect top-rerank_n unique companies & scores
                sparse_order = np.argsort(sparse_scores)[::-1]
                cand_tfidf = {}   # orig_id → best tfidf score
                for idx in sparse_order:
                    oid = self.name_mapping[idx]
                    if oid not in cand_tfidf:
                        cand_tfidf[oid] = sparse_scores[idx]
                    if len(cand_tfidf) >= self.rerank_n:
                        break

                top_scores = list(cand_tfidf.values())  # already insertion-ordered (descending)

                # Only invoke dense if top-1 and top-2 scores are too close
                need_rerank = (
                    len(top_scores) >= 2 and
                    (top_scores[0] - top_scores[1]) < self.rerank_threshold
                )

                if need_rerank:
                    dense_scores = _get_dense_scores()
                    cand_dense = {}
                    for idx, score in enumerate(dense_scores):
                        oid = self.name_mapping[idx]
                        if oid in cand_tfidf:
                            cand_dense[oid] = max(cand_dense.get(oid, -1.0), score)
                    similarities = np.zeros(len(sparse_scores))
                    for idx in range(len(sparse_scores)):
                        oid = self.name_mapping[idx]
                        if oid in cand_dense:
                            similarities[idx] = cand_dense[oid]
                else:
                    similarities = sparse_scores   # fast path — no dense call

            elif self.fusion == 'rrf':
                dense_scores = _get_dense_scores()
                similarities = _rrf_fuse([sparse_scores, dense_scores])
            else:  # weighted
                dense_scores = _get_dense_scores()
                similarities = self.sparse_weight * sparse_scores + self.dense_weight * dense_scores

        elif self.model_name == 'tfidf-wordllama' or self.model_name == 'hybrid-cross-lang':
            # ── Stage 1: TF-IDF retrieval (always run - fast!) ────────────────
            query_vec = self.vectorizer.transform([query_cleaned])
            tfidf_scores = cosine_similarity(query_vec, self.corpus_vectors).flatten()

            # Get top 20 candidates from TF-IDF
            tfidf_indices = np.argsort(tfidf_scores)[::-1][:20]

            # Group by original company ID (deduplicate variants)
            cand_tfidf = {}   # orig_id → best tfidf score
            for idx in tfidf_indices:
                oid = self.name_mapping[idx]
                if oid not in cand_tfidf:
                    cand_tfidf[oid] = tfidf_scores[idx]

            # Convert to sorted list
            tfidf_results = [
                {'company_id': oid, 'score': score}
                for oid, score in sorted(cand_tfidf.items(), key=lambda x: x[1], reverse=True)
            ]

            # ── Stage 2: Confidence check - should we rerank? ───────────────
            need_rerank = (
                len(tfidf_results) >= 2 and
                (tfidf_results[0]['score'] - tfidf_results[1]['score']) < self.rerank_threshold
            )

            if need_rerank and self.wl is not None:
                # Low confidence: Use WordLlama to rerank top-10 candidates
                candidate_ids = [r['company_id'] for r in tfidf_results[:10]]
                candidate_names = [self.corpus_names[oid] for oid in candidate_ids]

                # Embed query and candidates with WordLlama
                query_wl = self.wl.embed([query])
                candidates_wl = self.wordllama_vectors[candidate_ids]

                # Compute semantic similarities
                wl_scores = cosine_similarity(query_wl, candidates_wl).flatten()

                # Build similarities array: -inf for non-candidates
                similarities = np.full(len(tfidf_scores), -np.inf)
                for idx, score in zip(candidate_ids, wl_scores):
                    # Find all variant indices for this original company ID
                    for var_idx, oid in enumerate(self.name_mapping):
                        if oid == idx:
                            similarities[var_idx] = float(score)
            else:
                # High confidence or reranking disabled: return TF-IDF results
                similarities = tfidf_scores

        elif self.model_name == 'bm25-dense':
            # Sparse scores: BM25 normalised to 0-1
            tokenized_query = query_cleaned.split()
            bm25_raw = np.array(self.bm25_model.get_scores(tokenized_query))
            max_bm25 = bm25_raw.max()
            sparse_scores = bm25_raw / max_bm25 if max_bm25 > 0 else bm25_raw
            
            # Dense scores
            query_dense = self.st_model.encode(
                [query_cleaned], convert_to_numpy=True, normalize_embeddings=True
            )
            dense_scores = (self.dense_vectors @ query_dense.T).flatten()
            dense_scores = np.clip(dense_scores, 0, 1)
            
            if self.fusion == 'rrf':
                similarities = _rrf_fuse([sparse_scores, dense_scores])
            else:  # weighted
                similarities = self.sparse_weight * sparse_scores + self.dense_weight * dense_scores
        elif 'wordllama' in self.model_name:
            query_vec = self.wl.embed([query_cleaned])
            similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        elif hasattr(self, 'st_model'):
            query_vec = self.st_model.encode([query_cleaned], convert_to_numpy=True)
            similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        elif self.model_name == 'bm25':
            tokenized_query = query_cleaned.split()
            similarities = self.bm25_model.get_scores(tokenized_query)
        
        # Lấy top k — oversample to handle dedup and norm-key expansion
        indices = np.argsort(similarities)[-top_k * 5:][::-1]

        results = []
        seen_norm_keys = set()   # dedup by canonical normalised form
        for idx in indices:
            score = similarities[idx]
            if score <= 0:
                continue

            original_idx = self.name_mapping[idx]
            norm_key = self._orig_to_norm.get(original_idx)

            if norm_key is None or norm_key in seen_norm_keys:
                continue
            seen_norm_keys.add(norm_key)

            # Expand: emit all corpus entries that share this normalised form.
            # Typically just one entry; >1 only for near-duplicate corpus names
            # (e.g. "CÔNG TY TNHH XUẤT NHẬP KHẨU ABC" and
            #        "CÔNG TY TNHH XNK ABC" both → same norm_key).
            for oid in self._norm_to_originals[norm_key]:
                results.append({
                    "company": self.corpus_names[oid],
                    "score": float(score),
                    "_norm_key": norm_key,
                })

            if len(results) >= top_k:
                break

        # ── Post-processing ──────────────────────────────────────────────────

        # 1. Repeated-token penalty: demote corpus entries whose cleaned name
        #    contains consecutive duplicate tokens (data-quality artifact such as
        #    "SẢN XUẤT SẢN XUẤT"), which can spuriously dominate char n-gram scores.
        REPEAT_PENALTY = 0.85
        for r in results:
            if _has_repeated_tokens(r['_norm_key']):
                r['score'] *= REPEAT_PENALTY

        # Re-sort after penalty
        results.sort(key=lambda x: x['score'], reverse=True)

        # 2. Entity-type aware tie-breaking: if the query explicitly names an
        #    entity type (tnhh/cp/cn/td/…) and top-1 has a different entity type,
        #    promote the highest-scoring result that DOES match—provided its score
        #    is within ENTITY_GAP_THRESHOLD of top-1.
        ENTITY_GAP_THRESHOLD = 0.20
        query_et = _extract_entity_type(query_cleaned)
        if query_et and len(results) > 1:
            top1_et = _extract_entity_type(results[0]['_norm_key'])
            if top1_et != query_et:
                # Find the best matching result
                for j in range(1, len(results)):
                    cand_et = _extract_entity_type(results[j]['_norm_key'])
                    if cand_et == query_et:
                        gap = results[0]['score'] - results[j]['score']
                        if gap <= ENTITY_GAP_THRESHOLD:
                            # Promote: move results[j] to position 0
                            results.insert(0, results.pop(j))
                        break  # only check the highest-scored matching candidate

        # Strip internal fields before returning
        for r in results:
            r.pop('_norm_key', None)

        # Apply confidence threshold: suppress results when top-1 score is too low
        if min_score > 0.0 and results and results[0]['score'] < min_score:
            return []

        return results

    def save_index(self, path: str):
        """
        Save the built index to disk for later reuse.

        This saves:
        - TfidfVectorizer model
        - SVD model (if using LSA)
        - Dense embeddings (if using dense model)
        - Corpus metadata (names, mappings)

        Args:
            path: Directory path to save index files
        """
        import pickle
        import joblib
        from pathlib import Path

        index_path = Path(path)
        index_path.mkdir(parents=True, exist_ok=True)

        print(f"[CompanyMatcher] Saving index to: {index_path}")

        # Save vectorizer
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, index_path / "vectorizer.joblib")
            print(f"[CompanyMatcher] Saved vectorizer")

        # Save SVD model (if using LSA)
        if self.svd is not None:
            joblib.dump(self.svd, index_path / "svd.joblib")
            print(f"[CompanyMatcher] Saved SVD model")

        # Save dense vectors (if using dense model)
        if hasattr(self, 'dense_vectors') and self.dense_vectors is not None:
            np.save(index_path / "dense_vectors.npy", self.dense_vectors)
            print(f"[CompanyMatcher] Saved dense vectors: {self.dense_vectors.shape}")

        # Save WordLlama vectors (if using hybrid model)
        if hasattr(self, 'wordllama_vectors') and self.wordllama_vectors is not None:
            np.save(index_path / "wordllama_vectors.npy", self.wordllama_vectors)
            print(f"[CompanyMatcher] Saved WordLlama vectors: {self.wordllama_vectors.shape}")

        # Save corpus vectors
        if self.corpus_vectors is not None:
            np.save(index_path / "corpus_vectors.npy", self.corpus_vectors)
            print(f"[CompanyMatcher] Saved corpus vectors: {self.corpus_vectors.shape}")

        # Save BM25 model (if exists)
        if hasattr(self, 'bm25_model') and self.bm25_model is not None:
            with open(index_path / "bm25_model.pkl", 'wb') as f:
                pickle.dump(self.bm25_model, f)
            print(f"[CompanyMatcher] Saved BM25 model")

        # Save corpus metadata
        metadata = {
            'corpus_names': self.corpus_names,
            'name_mapping': self.name_mapping,
            '_orig_to_norm': self._orig_to_norm,
            '_norm_to_originals': self._norm_to_originals,
            'model_name': self.model_name,
            'lsa_dims': self.lsa_dims,
            'max_features': self.max_features,
            'remove_stopwords': self.remove_stopwords,
            'use_gpu': self.use_gpu,
            'dense_model_name': self.dense_model_name,
            'sparse_weight': self.sparse_weight,
            'dense_weight': self.dense_weight,
            'fusion': self.fusion,
            'rerank_n': self.rerank_n,
            'rerank_threshold': self.rerank_threshold
        }

        # Save processed names if available (needed for cross-rerank)
        if hasattr(self, '_processed_names'):
            metadata['_processed_names'] = self._processed_names

        with open(index_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        print(f"[CompanyMatcher] Saved metadata ({len(self.corpus_names)} companies)")

        print(f"[CompanyMatcher] Index saved successfully")

    @classmethod
    def load_index(cls, path: str):
        """
        Load a previously saved index from disk.

        Args:
            path: Directory path containing saved index files

        Returns:
            CompanyMatcher instance with loaded index
        """
        import pickle
        import joblib
        from pathlib import Path

        index_path = Path(path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index path not found: {index_path}")

        print(f"[CompanyMatcher] Loading index from: {index_path}")

        # Load metadata first
        with open(index_path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        print(f"[CompanyMatcher] Loaded metadata for {len(metadata['corpus_names'])} companies")

        # Create matcher instance with saved config
        matcher = cls(
            model_name=metadata['model_name'],
            lsa_dims=metadata.get('lsa_dims', 512),
            max_features=metadata.get('max_features', 100000),
            remove_stopwords=metadata['remove_stopwords'],
            use_gpu=metadata.get('use_gpu', False),
            dense_model_name=metadata.get('dense_model_name', 'BAAI/bge-m3'),
            sparse_weight=metadata.get('sparse_weight', 0.5),
            dense_weight=metadata.get('dense_weight', 0.5),
            fusion=metadata.get('fusion', 'weighted'),
            rerank_n=metadata.get('rerank_n', 10),
            rerank_threshold=metadata.get('rerank_threshold', 0.05)
        )

        # Restore metadata
        matcher.corpus_names = metadata['corpus_names']
        matcher.name_mapping = metadata['name_mapping']
        matcher._orig_to_norm = metadata['_orig_to_norm']
        matcher._norm_to_originals = metadata['_norm_to_originals']

        # Restore processed names if available
        if '_processed_names' in metadata:
            matcher._processed_names = metadata['_processed_names']

        # Load vectorizer
        vectorizer_path = index_path / "vectorizer.joblib"
        if vectorizer_path.exists():
            matcher.vectorizer = joblib.load(vectorizer_path)
            print(f"[CompanyMatcher] Loaded vectorizer")

        # Load SVD model (if exists)
        svd_path = index_path / "svd.joblib"
        if svd_path.exists():
            matcher.svd = joblib.load(svd_path)
            print(f"[CompanyMatcher] Loaded SVD model")

        # Load corpus vectors
        vectors_path = index_path / "corpus_vectors.npy"
        if vectors_path.exists():
            matcher.corpus_vectors = np.load(vectors_path)
            print(f"[CompanyMatcher] Loaded corpus vectors: {matcher.corpus_vectors.shape}")

        # Load dense vectors (if exists)
        dense_path = index_path / "dense_vectors.npy"
        if dense_path.exists():
            matcher.dense_vectors = np.load(dense_path)
            print(f"[CompanyMatcher] Loaded dense vectors: {matcher.dense_vectors.shape}")

        # Load WordLlama vectors (if exists)
        wl_path = index_path / "wordllama_vectors.npy"
        if wl_path.exists():
            from wordllama import WordLlama
            matcher.wordllama_vectors = np.load(wl_path)
            # Reload WordLlama model
            matcher.wl = WordLlama.load_m2v(config="m2v_multilingual")
            print(f"[CompanyMatcher] Loaded WordLlama vectors: {matcher.wordllama_vectors.shape}")

        # Load BM25 model (if exists)
        bm25_path = index_path / "bm25_model.pkl"
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                matcher.bm25_model = pickle.load(f)
            print(f"[CompanyMatcher] Loaded BM25 model")

        # Load SentenceTransformer/CrossEncoder model (if using dense model)
        if metadata['model_name'] in ['tfidf-dense', 'bm25-dense']:
            from sentence_transformers import SentenceTransformer

            dense_model_name = metadata.get('dense_model_name', 'BAAI/bge-m3')

            if matcher.fusion == 'cross-rerank':
                from sentence_transformers import CrossEncoder
                matcher.cross_encoder = CrossEncoder(
                    dense_model_name,
                    device='cuda' if matcher.use_gpu else 'cpu'
                )
                print(f"[CompanyMatcher] Loaded CrossEncoder: {dense_model_name}")
            elif hasattr(matcher, 'st_model'):
                matcher.st_model = SentenceTransformer(
                    dense_model_name,
                    device='cuda' if matcher.use_gpu else 'cpu'
                )
                print(f"[CompanyMatcher] Loaded SentenceTransformer: {dense_model_name}")

        print(f"[CompanyMatcher] Index loaded successfully")

        return matcher


if __name__ == "__main__":
    # Demo
    corpus = [
        "Công ty Cổ phần Sữa Việt Nam",
        "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam",
        "Tập đoàn Viễn thông Quân đội"
    ]
    matcher = CompanyMatcher()
    matcher.build_index(corpus)
    
    query = "VINAMILK"
    print(f"Query: {query}")
    print(matcher.search(query))
    
    query2 = "ngan hang vietcombank"
    print(f"Query: {query2}")
    print(matcher.search(query2))
