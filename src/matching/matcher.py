import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_company_name, remove_accents

class CompanyMatcher:
    def __init__(self, model_name='tfidf', use_gpu=False):
        """
        Khởi tạo mô hình TF-IDF (Sử dụng scikit-learn để tránh lỗi native DLL trên Windows).
        """
        print(f"Using Pure Python/Sklearn TF-IDF Matcher...")
        self.vectorizer = TfidfVectorizer(
            analyzer='char', 
            ngram_range=(2, 5),
            sublinear_tf=True,
            min_df=1
        )
        self.index = None
        self.corpus_names = []
        self.corpus_vectors = None

    def build_index(self, names):
        """
        Xây dựng chỉ mục tìm kiếm từ danh sách tên công ty hệ thống.
        """
        self.corpus_names = names
        processed_names = []
        self.name_mapping = []
        
        for i, name in enumerate(names):
            cleaned = clean_company_name(name)
            no_accent = remove_accents(cleaned)
            
            processed_names.append(cleaned)
            self.name_mapping.append(i)
            
            if cleaned != no_accent:
                processed_names.append(no_accent)
                self.name_mapping.append(i)

        print(f"Vectorizing {len(processed_names)} variants for {len(names)} companies...")
        self.corpus_vectors = self.vectorizer.fit_transform(processed_names)
        print("TF-IDF Index built successfully.")

    def search(self, query, top_k=5):
        """
        Tìm kiếm công ty phù hợp nhất cho 1 query.
        """
        query_cleaned = clean_company_name(query)
        query_vec = self.vectorizer.transform([query_cleaned])
        
        # Tính toán cosine similarity
        similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        
        # Lấy top k
        indices = np.argsort(similarities)[-top_k*2:][::-1] # Lấy dư để filter
        
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
