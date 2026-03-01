"""
Configuration management for Databricks pipeline.

This module provides configuration loading and management for the company
name matching pipeline. Configuration is loaded from YAML files and can be
overridden with environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the company matching pipeline."""

    # Model configuration
    model_name: str = "tfidf-lsa"  # Use LSA for 2M scale
    lsa_dims: int = 512  # LSA dimensions (reduces memory from 8TB to 4GB)
    remove_stopwords: bool = True
    use_gpu: bool = False  # CPU-only as per requirements

    # Dense model configuration (optional, for hybrid approaches)
    dense_model_name: str = "BAAI/bge-m3"
    fusion: str = "adaptive-rerank"
    rerank_n: int = 50  # Candidates to retrieve for reranking
    rerank_threshold: float = 0.08  # Score gap threshold for adaptive reranking

    # Matching parameters
    top_k: int = 5
    min_score: float = 0.0

    # Delta Lake paths
    bronze_path: str = "/delta/bronze/banking_transactions"
    silver_path: str = "/delta/silver/companies_cleaned"
    gold_index_path: str = "/delta/gold/company_index"
    gold_matches_path: str = "/delta/gold/transaction_matches"

    # Stage configurations
    stage1_partitions: int = 200  # Target partitions after preprocessing
    stage2_dedup_strategy: str = "first"  # "first" or "longest"
    stage3_batch_size: int = 10000  # Batch size for index building

    # Performance tuning
    shuffle_partitions: int = 200
    adaptive_query: bool = True
    arrow_enabled: bool = True

    # Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    alert_threshold_avg_score: float = 0.85
    alert_threshold_high_confidence: float = 0.90

    # Databricks cluster configuration
    databricks_cluster_id: Optional[str] = None
    databricks_workspace_url: Optional[str] = None
    databricks_token: Optional[str] = None

    # Local Spark configuration (for testing)
    local_driver_memory: str = "4g"
    local_executor_memory: str = "4g"
    local_cores: int = 4

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


def load_config(config_path: Optional[str] = None, profile: str = "default") -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to YAML config file (default: config/pipeline_config.yaml)
        profile: Profile to use (default, dev, staging, prod)

    Returns:
        PipelineConfig instance

    Environment Variables Override:
        Any config value can be overridden with environment variables.
        Use uppercase names with underscores, e.g.:
        - MODEL_NAME → model_name
        - LSA_DIMS → lsa_dims
        - BRONZE_PATH → bronze_path
    """
    # Default config path
    if config_path is None:
        # Try multiple possible locations
        possible_paths = [
            "src/databricks/config/pipeline_config.yaml",
            "config/pipeline_config.yaml",
            "/Workspace/config/pipeline_config.yaml",  # Databricks workspace
        ]

        for path in possible_paths:
            if Path(path).exists():
                config_path = path
                break
        else:
            # Return default config if no file found
            config_path = None

    # Load YAML file
    config_dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

            # Extract profile-specific config
            if profile in yaml_config.get('profiles', {}):
                config_dict = yaml_config['profiles'][profile]
            else:
                config_dict = yaml_config.get('pipeline', {})

            # Override with Databricks-specific config
            if 'databricks' in yaml_config:
                config_dict.update(yaml_config['databricks'])

    # Override with environment variables
    env_overrides = {
        'model_name': os.getenv('MODEL_NAME'),
        'lsa_dims': os.getenv('LSA_DIMS'),
        'remove_stopwords': os.getenv('REMOVE_STOPWORDS'),
        'use_gpu': os.getenv('USE_GPU'),
        'dense_model_name': os.getenv('DENSE_MODEL_NAME'),
        'fusion': os.getenv('FUSION'),
        'rerank_n': os.getenv('RERANK_N'),
        'rerank_threshold': os.getenv('RERANK_THRESHOLD'),
        'top_k': os.getenv('TOP_K'),
        'min_score': os.getenv('MIN_SCORE'),
        'bronze_path': os.getenv('BRONZE_PATH'),
        'silver_path': os.getenv('SILVER_PATH'),
        'gold_index_path': os.getenv('GOLD_INDEX_PATH'),
        'gold_matches_path': os.getenv('GOLD_MATCHES_PATH'),
        'stage1_partitions': os.getenv('STAGE1_PARTITIONS'),
        'stage2_dedup_strategy': os.getenv('STAGE2_DEDUP_STRATEGY'),
        'shuffle_partitions': os.getenv('SHUFFLE_PARTITIONS'),
        'log_level': os.getenv('LOG_LEVEL'),
        'databricks_cluster_id': os.getenv('DATABRICKS_CLUSTER_ID'),
        'databricks_workspace_url': os.getenv('DATABRICKS_WORKSPACE_URL'),
        'databricks_token': os.getenv('DATABRICKS_TOKEN'),
    }

    # Apply environment overrides (only for non-None values)
    for key, env_value in env_overrides.items():
        if env_value is not None:
            # Type conversion
            if key in ['lsa_dims', 'rerank_n', 'top_k', 'stage1_partitions', 'shuffle_partitions', 'local_cores']:
                config_dict[key] = int(env_value)
            elif key in ['remove_stopwords', 'use_gpu', 'adaptive_query', 'arrow_enabled', 'enable_metrics']:
                config_dict[key] = env_value.lower() in ('true', '1', 'yes', 'on')
            elif key in ['rerank_threshold', 'min_score', 'alert_threshold_avg_score', 'alert_threshold_high_confidence']:
                config_dict[key] = float(env_value)
            else:
                config_dict[key] = env_value

    # Create config object
    return PipelineConfig.from_dict(config_dict)


def save_config(config: PipelineConfig, config_path: str):
    """
    Save pipeline configuration to YAML file.

    Args:
        config: PipelineConfig instance
        config_path: Path to save YAML file
    """
    config_dict = config.to_dict()

    # Remove sensitive values before saving
    config_dict.pop('databricks_token', None)

    yaml_content = {
        'pipeline': config_dict,
        'version': '1.0.0'
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Configuration saved to: {config_path}")
