import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.preprocess import clean_company_name, remove_accents

class CompanyMatcher:
    def __init__(self, model_name='BAAI/bge-m3', use_gpu=False):
        """
        Khởi tạo mô hình embedding và index FAISS.
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus_names = []
        self.corpus_embeddings = None

    def build_index(self, names):
        """
        Xây dựng chỉ mục tìm kiếm từ danh sách tên công ty hệ thống.
        """
        self.corpus_names = names
        # Tiền xử lý: tạo dataset kép (có dấu và không dấu) để tăng độ phủ
        processed_names = []
        name_mapping = [] # mapping từ index trong FAISS về index trong self.corpus_names
        
        for i, name in enumerate(names):
            cleaned = clean_company_name(name)
            no_accent = remove_accents(cleaned)
            
            processed_names.append(cleaned)
            name_mapping.append(i)
            
            if cleaned != no_accent:
                processed_names.append(no_accent)
                name_mapping.append(i)

        print(f"Encoding {len(processed_names)} variants for {len(names)} companies...")
        embeddings = self.model.encode(processed_names, show_progress_bar=True)
        self.corpus_embeddings = np.array(embeddings).astype('float32')
        self.name_mapping = name_mapping

        # Khởi tạo FAISS Index (Cosine Similarity)
        dimension = self.corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Inner Product on normalized vectors = Cosine Sim
        faiss.normalize_L2(self.corpus_embeddings)
        self.index.add(self.corpus_embeddings)
        print("Index built successfully.")

    def search(self, query, top_k=5):
        """
        Tìm kiếm công ty phù hợp nhất cho 1 query.
        """
        query_cleaned = clean_company_name(query)
        # Có thể search cả bản có dấu và không dấu của query rồi lấy max score
        query_vector = self.model.encode([query_cleaned]).astype('float32')
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        seen_ids = set()
        for score, idx in zip(distances[0], indices[0]):
            original_idx = self.name_mapping[idx]
            if original_idx not in seen_ids:
                results.append({
                    "company": self.corpus_names[original_idx],
                    "score": float(score)
                })
                seen_ids.add(original_idx)
                
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
