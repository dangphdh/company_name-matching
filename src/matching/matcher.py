import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_company_name, remove_accents

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
                 rerank_threshold=0.05):
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
            rerank_n: Candidates retrieved per retriever for rerank strategies
            rerank_threshold: Score gap below which adaptive-rerank triggers (default 0.05)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.remove_stopwords = remove_stopwords
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
        self.dense_model_name = dense_model_name
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.fusion = fusion
        self.rerank_n = rerank_n
        self.rerank_threshold = rerank_threshold
        self.index = None
        self.corpus_names = []
        self.corpus_vectors = None
        
        if model_name == 'tfidf' or model_name == 'tfidf-char-ngram':
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
        
        for i, name in enumerate(names):
            cleaned = clean_company_name(name, remove_stopwords=self.remove_stopwords)
            no_accent = remove_accents(cleaned)
            
            processed_names.append(cleaned)
            self.name_mapping.append(i)
            
            if cleaned != no_accent:
                processed_names.append(no_accent)
                self.name_mapping.append(i)

        print(f"Vectorizing {len(processed_names)} variants for {len(names)} companies...")
        
        if self.model_name == 'tfidf':
            self.corpus_vectors = self.vectorizer.fit_transform(processed_names)
        elif self.model_name == 'tfidf-bm25' or self.model_name == 'hybrid':
            # Hybrid: Build both TF-IDF and BM25 indices
            self.corpus_vectors = self.vectorizer.fit_transform(processed_names)
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.split() for doc in processed_names]
            self.bm25_model = BM25Okapi(tokenized_corpus)
        elif self.model_name == 'tfidf-dense':
            # Sparse: TF-IDF char n-gram
            self.corpus_vectors = self.vectorizer.fit_transform(processed_names)
            # Dense: sentence embedding
            print(f"  Encoding dense embeddings for {len(processed_names)} variants...")
            self.dense_vectors = self.st_model.encode(
                processed_names, batch_size=512,
                show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
            )
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

    def search(self, query, top_k=5):
        """
        Tìm kiếm công ty phù hợp nhất cho 1 query.
        """
        query_cleaned = clean_company_name(query, remove_stopwords=self.remove_stopwords)
        
        if self.model_name == 'tfidf':
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

            if self.fusion == 'tfidf-rerank':
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
        
        # Lấy top k
        indices = np.argsort(similarities)[-top_k*3:][::-1] # Lấy dư để filter ids trùng
        
        results = []
        seen_ids = set()
        for idx in indices:
            score = similarities[idx]
            if score <= 0: continue
            
            original_idx = self.name_mapping[idx]
            if original_idx not in seen_ids:
                results.append({
                    "company": self.corpus_names[original_idx],
                    "score": float(score)
                })
                seen_ids.add(original_idx)
                if len(results) >= top_k:
                    break
                
        return results

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
