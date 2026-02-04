# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance Vietnamese company name matching system that achieves >99% Top-1 accuracy with <3ms latency. The system uses TF-IDF Char N-gram vectors optimized for Vietnamese text processing, handling common real-world variants like abbreviations, no-accent writing, typos, and word reordering.

**Tech Stack:**
- Python 3.x with scikit-learn for TF-IDF vectorization
- Optional WordLlama embeddings for alternative matching models
- LLM-based synthetic data generation (OpenAI/Zhipu GLM)
- Vietnamese NLP with custom preprocessing and stopword removal

## Core Architecture

The system follows a three-stage pipeline:

1. **Preprocessing** (`src/preprocess.py`): NFC Unicode normalization, Vietnamese accent handling, extensive stopword removal for company types
2. **Vectorization** (`src/matching/matcher.py`): TF-IDF Char N-gram (2-5) or WordLlama embeddings
3. **Similarity Search**: Cosine similarity with dual indexing (accented + no-accent variants)

### Key Design Decisions

**Why TF-IDF Char N-gram over embeddings:**
- Handles Vietnamese abbreviations better (e.g., "TNHH" vs "Trách nhiệm hữu hạn")
- Sub-millisecond latency without GPU requirement
- Robust to typos and character-level variations
- Excellent performance on word reordering ("A&P TM DV" vs "TM DV A&P")

**Dual Variant Indexing:**
Every company name generates two index entries - accented and no-accent versions. This ensures queries like "Vinamilk" match "CÔNG TY TNHH SỮA VIỆT NAM" correctly.

**Stopword Strategy:**
Comprehensive removal of company type terms (TNHH, CP, MTV, TM DV, XNK, etc.) in both accented and unaccented forms. This focuses matching on the actual brand name rather than legal entity structure.

## Common Development Commands

### Setup and Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure LLM API (optional, for synthetic data generation only)
# Edit config/llm_config.yaml or set .env variables
```

### Running Tests
```bash
# Run all unit tests
python -m unittest tests/

# Run specific test file
python -m unittest tests.test_generators
```

### Data Pipeline
```bash
# Scrape real company data (optional - for expanding corpus)
python scripts/scrape_infocom.py

# Generate evaluation dataset from corpus
python scripts/generate_eval_dataset.py
# Output: data/eval/corpus.jsonl, data/eval/queries.jsonl

# Run matching evaluation
python scripts/evaluate_matching.py
# Tests multiple model/stopword combinations

# Run fuzzy matching baseline comparison
python scripts/evaluate_fuzzy.py
```

### Quick Demo
```bash
# Run main demo with sample queries
python main.py
```

## File Structure

```
src/
├── preprocess.py              # Vietnamese text normalization, stopword removal
├── matching/
│   └── matcher.py            # CompanyMatcher class: TF-IDF/WordLlama matching
└── synthetic/
    ├── combinatorial.py      # Rule-based variant generation
    ├── generator.py          # LLM-based synthetic data generation
    └── prompts.py            # LLM prompts for variant generation

scripts/
├── scrape_infocom.py         # Web scraper for company data
├── generate_eval_dataset.py  # Creates corpus/queries JSONL files
└── evaluate_matching.py      # Accuracy/latency benchmarking

data/
├── sample_system_names.txt   # Base corpus (1000+ companies)
└── eval/                     # Generated evaluation datasets (gitignored)
```

## Vietnamese Text Processing Conventions

**Always normalize Vietnamese text:**
```python
from src.preprocess import normalize_vietnamese_text, clean_company_name
text = normalize_vietnamese_text(raw_input)  # NFC normalization
cleaned = clean_company_name(text, remove_stopwords=True)
```

**Accent handling:**
- Use `remove_accents()` from `src/preprocess.py` for no-accent variants
- Custom implementation preserves Vietnamese character mappings
- Both accented and no-accent versions are indexed for search

**Stopword patterns:**
When adding new company types, update both accented and unaccented lists in `clean_company_name()`:
```python
stop_words = [r'\bcty\b', r'\bcông ty\b', ...]  # Add new patterns
stop_words_no_accent = [r'\bcong ty\b', ...]   # Corresponding no-accent
```

## Using the Matcher

```python
from src.matching.matcher import CompanyMatcher

# Initialize with default TF-IDF model
matcher = CompanyMatcher()
# Or specify model: "tfidf", "wordllama-l2", "wordllama-l3"

# Build search index
matcher.build_index([
    "CÔNG TY TNHH SỮA VIỆT NAM",
    "Ngân hàng TMCP Ngoại thương Việt Nam",
    # ... more companies
])

# Search for matches
results = matcher.search("Vinamilk", top_k=5)
# Returns: [{"company": "...", "score": 0.95}, ...]
```

**Matcher options:**
- `model_name`: "tfidf" (default), "wordllama-l2", "wordllama-l3"
- `remove_stopwords`: True (default) or False
- `use_gpu`: For WordLlama models (if available)

## Evaluation Metrics

The system uses three key metrics:
- **Top-1 Accuracy**: First result matches target company ID
- **Top-3 Accuracy**: Target in top 3 results
- **Latency**: Average processing time per query (ms)

Current baseline (TF-IDF, stopwords removed):
- Top-1: ~99.8%
- Top-3: 100%
- Latency: ~2ms per query

## Configuration Files

**LLM Configuration** (`config/llm_config.yaml`):
```yaml
api_provider: "zai"  # or "openai"
model_name: "glm-4.7"
temperature: 0.3
max_tokens: 2000
batch_size: 15
```

API keys should be set in `.env`:
- `ZAI_API_KEY`, `ZAI_API_BASE` for Zhipu GLM
- `OPENAI_API_KEY`, `OPENAI_API_BASE` for OpenAI

## Extending the System

**Adding new synthetic generation patterns:**
- Rule-based: Edit `CombinatorialGenerator.type_variants` in `src/synthetic/combinatorial.py`
- LLM-based: Modify prompts in `src/synthetic/prompts.py`

**Optimizing for specific use cases:**
- Adjust `ngram_range=(2, 5)` in `CompanyMatcher.__init__()` for different character patterns
- Tune `remove_stopwords` parameter based on brand name vs legal entity matching needs
- Modify `top_k*3` multiplier in `search()` if many duplicate IDs appear in results

**Performance considerations:**
- Char N-gram range (2-5) balances accuracy vs speed
- Stopword removal significantly reduces noise and improves matching
- Dual indexing (accented + no-accent) handles real-world input variations
- TF-IDF requires no GPU and scales to millions of companies
