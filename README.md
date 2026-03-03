# Vietnamese Company Name Matching

High-performance Vietnamese company name matching system with >99% accuracy and <3ms latency.

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎯 **High Accuracy**: >99% Top-1 accuracy on Vietnamese company names
- ⚡ **Fast Performance**: <3ms latency per query (no GPU required)
- 🔧 **Robust Matching**: Handles abbreviations, typos, no-accent text, word reordering
- 🚀 **Scalable**: Support for local Spark or Databricks for large-scale processing
- 🌏 **Vietnamese Optimized**: Custom preprocessing and stopword removal for Vietnamese text

## 📦 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run demo
python main.py

# Run evaluation
python scripts/evaluate_matching.py
```

## 💡 Usage

```python
from src.matching.matcher import CompanyMatcher

# Initialize with hybrid model (recommended)
matcher = CompanyMatcher(model_name='tfidf-bm25')

# Build index
companies = ["CÔNG TY TNHH SỮA VIỆT NAM", "Ngân hàng TMCP Ngoại thương Việt Nam"]
matcher.build_index(companies)

# Search
results = matcher.search("Vinamilk", top_k=5)
# Returns: [{'company': '...', 'score': 0.95}, ...]
```

## 📚 Documentation

Full documentation is available in the [docs/](docs/) directory:

- [Overview](docs/README.md) - Complete project documentation
- [Architecture](docs/architecture.md) - System design and technical decisions
- [Matching Guide](docs/matching-guide.md) - API reference and usage examples
- [Evaluation](docs/evaluation.md) - Performance metrics and benchmarks
- [Spark & Databricks](docs/spark-databricks.md) - Distributed processing setup

## 📊 Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | ~99.8% |
| Top-3 Accuracy | 100% |
| Avg Latency | ~2ms per query |
| Corpus Size | 1,000+ companies |
| Test Queries | 50,000+ variants |

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn (TF-IDF, BM25)
- Optional: Spark/Databricks for distributed processing
- Vietnamese NLP with custom preprocessing

## 📁 Project Structure

```
├── main.py                  # Quick demo script
├── requirements.txt         # Core dependencies
├── src/
│   ├── preprocess.py        # Vietnamese text preprocessing
│   ├── matching/
│   │   └── matcher.py       # Matching algorithms
│   └── synthetic/
│       ├── combinatorial.py # Rule-based variant generation
│       └── generator.py     # LLM-based synthetic data
├── scripts/
│   ├── scrape_infocom.py    # Web scraper
│   ├── generate_eval_dataset.py  # Create evaluation datasets
│   └── evaluate_matching.py      # Run benchmarks
├── examples/
│   ├── spark_local_example.py     # Local Spark demo
│   └── databricks_connect_example.py  # Databricks demo
├── data/
│   └── sample_system_names.txt  # Base corpus
└── docs/                  # Complete documentation
    ├── README.md          # Overview
    ├── architecture.md    # System design
    ├── matching-guide.md  # API reference
    ├── evaluation.md      # Performance metrics
    └── spark-databricks.md  # Spark setup
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

## 📖 Additional Guides

- For AI-assisted development, see [CLAUDE.md](CLAUDE.md)
- For technical details, see [docs/architecture.md](docs/architecture.md)
- For API documentation, see [docs/matching-guide.md](docs/matching-guide.md)
- For Spark/Databricks, see [docs/spark-databricks.md](docs/spark-databricks.md)

## 📝 License

This project is developed for educational and research purposes in entity matching.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tests pass for new features
- Documentation is updated accordingly

---

**Need help?** Check the [documentation](docs/) or open an issue.
