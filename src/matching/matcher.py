import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_company_name, remove_accents

class CompanyMatcher:
    def __init__(self, model_name='tfidf', use_gpu=False, remove_stopwords=True, tfidf_weight=0.5, bm25_weight=0.5):
        """
        Khởi tạo mô hình matching.
        
        Args:
            model_name: Model to use ('tfidf', 'bm25', 'tfidf-bm25', 'wordllama', 'sbert', etc.)
            use_gpu: Whether to use GPU
            remove_stopwords: Whether to remove stopwords during preprocessing
            tfidf_weight: Weight for TF-IDF in hybrid model (0.0-1.0)
            bm25_weight: Weight for BM25 in hybrid model (0.0-1.0)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.remove_stopwords = remove_stopwords
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
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
            # For BM25, use word-level tokenization (split by spaces and punctuation)
            tokenized_corpus = [doc.split() for doc in processed_names]
            self.bm25_model = BM25Okapi(tokenized_corpus)
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
            # Hybrid: Combine TF-IDF and BM25 scores
            # TF-IDF score via cosine similarity (0-1 range)
            query_vec = self.vectorizer.transform([query_cleaned])
            tfidf_scores = cosine_similarity(query_vec, self.corpus_vectors).flatten()
            
            # BM25 score (word-level)
            tokenized_query = query_cleaned.split()
            bm25_scores_raw = np.array(self.bm25_model.get_scores(tokenized_query))
            
            # Normalize BM25 scores to 0-1 range (min-max normalization)
            max_bm25 = bm25_scores_raw.max()
            if max_bm25 > 0:
                bm25_scores = bm25_scores_raw / max_bm25
            else:
                bm25_scores = bm25_scores_raw
            
            # Weighted combination
            similarities = (self.tfidf_weight * tfidf_scores + 
                          self.bm25_weight * bm25_scores)
        elif 'wordllama' in self.model_name:
            query_vec = self.wl.embed([query_cleaned])
            similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        elif hasattr(self, 'st_model'):
            query_vec = self.st_model.encode([query_cleaned], convert_to_numpy=True)
            similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        elif self.model_name == 'bm25':
            # BM25 scoring (word-level)
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
