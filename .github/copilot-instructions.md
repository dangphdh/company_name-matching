# Vietnamese Company Name Matching - AI Coding Guidelines

## Project Overview
This is a high-performance Vietnamese company name matching system using TF-IDF Char N-gram vectors (>99% Top-1 accuracy, <3ms latency). Core architecture: preprocessing → vectorization → similarity search.

## Key Architecture Patterns

### Preprocessing (src/preprocess.py)
- **NFC Unicode normalization** for consistent Vietnamese text
- **Extensive stop word removal**: Company type terms (TNHH, CP, MTV, TM DV, XNK, etc.) in both accented/unaccented forms
- **Accent removal** for robust no-accent matching
- **Focus on brand names**: Strip "Công ty TNHH ABC" → "ABC"

### Matching (src/matching/matcher.py)
- **TF-IDF Char N-gram (2-5)**: Optimized for Vietnamese abbreviations/typos
- **Dual indexing**: Store both accented and unaccented variants
- **Cosine similarity search** with deduplication
- **Model options**: TF-IDF (default) or WordLlama embeddings

### Synthetic Data Generation
- **Combinatorial (src/synthetic/combinatorial.py)**: Rule-based variants (abbreviations, no-accent, uppercase)
- **LLM-based (src/synthetic/generator.py)**: GLM-4 generates typos, informal names, English translations

## Critical Workflows

### Quick Demo
```python
from src.matching.matcher import CompanyMatcher
matcher = CompanyMatcher()  # Defaults to TF-IDF
matcher.build_index(["Công ty TNHH Sữa Việt Nam", ...])
results = matcher.search("Vinamilk")  # Returns [{"company": "Công ty TNHH Sữa Việt Nam", "score": 0.95}]
```

### Evaluation Pipeline
1. Generate dataset: `python scripts/generate_eval_dataset.py` (creates data/eval/corpus.jsonl & queries.jsonl)
2. Run evaluation: `python scripts/evaluate_matching.py` (measures Top-1/Top-3 accuracy, latency)
3. Test variants: TF-IDF with/without stopwords, WordLlama models

### Data Collection
- Scrape real companies: `python scripts/scrape_infocom.py`
- Base corpus: data/sample_system_names.txt (1000+ clean names)

## Project Conventions

### Vietnamese Text Handling
- Always use `unicodedata.normalize('NFC', text)` for consistency
- Remove accents with custom `remove_accents()` function (preserves Vietnamese mapping)
- Clean company names with `clean_company_name(text, remove_stopwords=True)`

### Configuration
- LLM config: config/llm_config.yaml (supports OpenAI/Zhipu GLM)
- API keys: Prefer .env over hardcoded values
- Model selection: "tfidf" (default), "wordllama-l2", "wordllama-l3"

### Testing
- Unit tests: `python -m unittest tests/`
- Evaluation metrics: Top-1/Top-3 accuracy, average latency
- Test data: JSONL format with id/name for corpus, text/target_id/method for queries

### Dependencies
- Core: numpy, scikit-learn, openai, pyyaml
- Vietnamese: underthesea (optional for advanced NLP)
- Data: pandas, beautifulsoup4, requests

## Common Patterns

### Adding New Company Types
Update `src/preprocess.py` stop_words lists (both accented and unaccented):
```python
stop_words = [r'\bcty\b', r'\bcông ty\b', ...]  # Add new patterns
stop_words_no_accent = [r'\bcong ty\b', ...]   # Corresponding no-accent versions
```

### Extending Synthetic Generation
For combinatorial: Add to `CombinatorialGenerator.type_variants` dict
For LLM: Modify prompts in `src/synthetic/prompts.py`

### Performance Optimization
- Char N-gram range (2-5) balances accuracy vs speed
- Remove stopwords reduces noise, improves matching
- Dual variants (accented + no-accent) in index handles real-world inputs

## Integration Points
- **LLM APIs**: Configurable OpenAI or Zhipu GLM endpoints
- **Data Sources**: Web scraping (Infocom), static corpus files
- **Output Formats**: JSON for API responses, JSONL for evaluation datasets