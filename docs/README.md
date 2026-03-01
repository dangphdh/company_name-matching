# Vietnamese Company Name Matching - Documentation

Welcome to the documentation for the high-performance Vietnamese company name matching system.

## 📚 Documentation Index

- [Quick Start](#quick-start) - Get started in 5 minutes
- [Architecture](architecture.md) - System design and technical decisions
- [Matching Guide](matching-guide.md) - How to use matching algorithms
- [Spark & Databricks](spark-databricks.md) - Distributed processing setup
- [Evaluation](evaluation.md) - Performance metrics and benchmarks

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.matching.matcher import CompanyMatcher

# Initialize matcher with hybrid model (recommended)
matcher = CompanyMatcher(model_name='tfidf-bm25')

# Build index from company names
companies = [
    "CÔNG TY TNHH SỮA VIỆT NAM",
    "Ngân hàng TMCP Ngoại thương Việt Nam",
    "Tập đoàn Hòa Phát"
]
matcher.build_index(companies)

# Search for matches
results = matcher.search("Vinamilk", top_k=5)
# Returns: [{'company': '...', 'score': 0.95}, ...]
```

## 🎯 Key Features

- **High Accuracy**: >99% Top-1 accuracy on Vietnamese company names
- **Fast Performance**: <3ms latency per query (no GPU required)
- **Robust Matching**: Handles abbreviations, typos, no-accent text, word reordering
- **Multiple Models**: TF-IDF, BM25, Hybrid, and LSA support
- **Scalable**: Local Spark or Databricks for large-scale processing

## 📊 Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | ~99.8% |
| Top-3 Accuracy | 100% |
| Avg Latency | ~2ms per query |
| Corpus Size | 1,000+ companies |
| Test Queries | 50,000+ variants |

## 🏗️ Architecture

The system uses a three-stage pipeline:

1. **Preprocessing** (`src/preprocess.py`)
   - NFC Unicode normalization
   - Vietnamese accent handling
   - Comprehensive stopword removal

2. **Vectorization** (`src/matching/matcher.py`)
   - TF-IDF Char N-gram (2-5) for semantic similarity
   - BM25 for term-level relevance
   - Optional LSA for dimensionality reduction
   - WordLlama embeddings support

3. **Similarity Search**
   - Cosine similarity with dual indexing
   - Accented + no-accent variant indexing
   - Hybrid scoring with configurable weights

## 📖 Additional Guides

### For Vietnamese Users
Xin lưu ý rằng hệ thống này được tối ưu hóa đặc biệt cho tên công ty Việt Nam, với xử lý tiếng Việt chuyên sâu.

### For Developers
- See [CLAUDE.md](../CLAUDE.md) for AI-assisted development guidelines
- Check [architecture.md](architecture.md) for detailed technical decisions
- Refer to [matching-guide.md](matching-guide.md) for API documentation

### For Data Engineers
- See [spark-databricks.md](spark-databricks.md) for distributed processing
- Learn about scaling to millions of companies

## 🛠️ Project Structure

```
├── main.py                 # Quick demo script
├── requirements.txt        # Core dependencies
├── requirements-spark-local.txt  # Spark for local development
├── src/
│   ├── preprocess.py       # Vietnamese text preprocessing
│   ├── matching/
│   │   └── matcher.py      # Matching algorithms (TF-IDF, BM25, LSA, Hybrid)
│   └── synthetic/
│       ├── combinatorial.py  # Rule-based variant generation
│       └── generator.py      # LLM-based synthetic data
├── scripts/
│   ├── scrape_infocom.py    # Web scraper for company data
│   ├── generate_eval_dataset.py  # Create evaluation datasets
│   └── evaluate_matching.py      # Benchmark accuracy & latency
├── examples/
│   ├── spark_local_example.py           # Local Spark demo
│   └── databricks_connect_example.py    # Databricks Connect demo
├── data/
│   ├── sample_system_names.txt  # Base corpus (1000+ companies)
│   └── eval/                     # Generated evaluation datasets
└── config/
    ├── spark_config.py           # Spark configuration utilities
    └── llm_config.yaml          # LLM API configuration
```

## 🧪 Running Examples

```bash
# Quick demo
python main.py

# Generate evaluation dataset
python scripts/generate_eval_dataset.py

# Run benchmarks
python scripts/evaluate_matching.py

# Local Spark experimentation
python examples/spark_local_example.py
```

## 📝 License

This project is developed for educational and research purposes in entity matching.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tests pass for new features
- Documentation is updated accordingly

---

Need help? Check the specific guides in the sidebar or open an issue.
