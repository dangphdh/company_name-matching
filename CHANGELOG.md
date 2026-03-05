# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Cross-language evaluation script added for BGE3 integration testing (in progress)

## [0.3.0] - 2026-03-03

### Added
- **Production-grade Databricks pipeline** for processing 2M+ company records with batch processing
- **LSA (Latent Semantic Analysis) support** for efficient dimensionality reduction and faster TF-IDF vectorization
- **Memory optimization** with `max_features` parameter to control LSA build memory usage
- Complete **Databricks integration documentation** in `docs/spark-databricks.md`
- Reorganized documentation structure with better categorization

### Performance
- LSA reduces memory footprint for large-scale deployments
- Batch processing enables handling millions of records efficiently

**Insight:** LSA is particularly valuable for Vietnamese text matching because it captures semantic relationships between similar company names while reducing the high-dimensional TF-IDF space, making large-scale matching more feasible.

## [0.2.0] - 2026-02-22

### Added
- **Hybrid TF-IDF + BM25 matcher** combining exact matching (TF-IDF) with keyword importance (BM25)
- **Error analysis tools** for systematic debugging of failed matches
- **Hybrid scoring system** that balances character-level similarity with keyword relevance
- Comprehensive evaluation results highlighting top-performing configurations

### Changed
- Enhanced `CompanyMatcher` with hybrid model support (`model_name='tfidf-bm25'`)
- Improved error analysis scripts with detailed categorization

### Performance
- Hybrid model achieves >99.8% Top-1 accuracy
- Better handling of partial matches and keyword-heavy queries

**Insight:** The hybrid approach leverages TF-IDF's strength at character-level matching (crucial for Vietnamese typos and abbreviations) while BM25 boosts important keywords, creating a more robust matcher that handles edge cases better than either method alone.

## [0.1.0] - 2026-02-05

### Added
- **Initial release** of Vietnamese company name matching system
- **TF-IDF Char N-gram vectorization** (2-5 grams) optimized for Vietnamese text
- **Dual-variant indexing** (accented + no-accent) for robust matching
- **Vietnamese stopword removal** for company types (TNHH, CP, MTV, etc.)
- **Comprehensive preprocessing** pipeline with NFC Unicode normalization
- **Evaluation framework** with Top-1/Top-3 accuracy and latency metrics
- **Synthetic data generation** (combinatorial + LLM-based variants)
- **Web scraper** for expanding company corpus from infocom.vn
- Complete documentation including architecture, matching guide, and evaluation metrics

### Performance
- **>99% Top-1 accuracy** on Vietnamese company names
- **<3ms latency** per query (no GPU required)
- Handles real-world variants: abbreviations, no-accent text, typos, word reordering

**Insight:** Char N-grams (2-5) were chosen over word-level tokenization because they naturally handle Vietnamese abbreviations like "TNHH" matching "Trách nhiệm hữu hạn" and are more robust to typos and character variations common in real-world company name queries.

---

## Versioning Strategy

This project uses **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to API or data formats
- **MINOR**: New features, backwards-compatible additions
- **PATCH**: Bug fixes, minor improvements, documentation updates

## Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future releases
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

## Release Notes Template

When preparing a new release:

1. Review all commits since the last release
2. Categorize changes (Added, Changed, Fixed, etc.)
3. Note breaking changes prominently
4. Include performance impact for significant changes
5. Add migration notes if API changes affect users
6. Update version number in `__version__` if applicable
