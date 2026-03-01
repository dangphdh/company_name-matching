"""Pipeline stages for company name matching."""

from .stage1_extract import run_stage1_extract
from .stage2_deduplicate import run_stage2_deduplicate
from .stage3_build_index import run_stage3_build_index
from .stage4_match import run_stage4_match

__all__ = [
    "run_stage1_extract",
    "run_stage2_deduplicate",
    "run_stage3_build_index",
    "run_stage4_match"
]
