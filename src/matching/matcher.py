import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_company_name, remove_accents

class CompanyMatcher:
    def __init__(self, model_name='tfidf', use_gpu=False, remove_stopwords=True):
        """
        Khởi tạo mô hình matching.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.remove_stopwords = remove_stopwords
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
        else:
            # Fallback hoặc nếu muốn dùng model khác từ HuggingFace (cần transformer)
            print(f"Warning: Model {model_name} not explicitly handled for WordLlama/TF-IDF. Defaulting to TF-IDF.")
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
        elif 'wordllama' in self.model_name:
            # WordLlama trả về numpy array
            self.corpus_vectors = self.wl.embed(processed_names)
            
        print(f"{self.model_name.upper()} Index built successfully.")

    def search(self, query, top_k=5):
        """
        Tìm kiếm công ty phù hợp nhất cho 1 query.
        """
        query_cleaned = clean_company_name(query, remove_stopwords=self.remove_stopwords)
        
        if self.model_name == 'tfidf':
            query_vec = self.vectorizer.transform([query_cleaned])
            similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        elif 'wordllama' in self.model_name:
            query_vec = self.wl.embed([query_cleaned])
            similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        
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
