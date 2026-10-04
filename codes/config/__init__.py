"""
Configuration module for Full-Triple-Matcher.
"""

from .dataset_config import (
    DATASETS,
    DATASET_ALIASES,
    PROJECT_ROOT,
    get_project_root,
    normalize_dataset_name,
    list_available_datasets,
    resolve_dataset_config,
    validate_dataset_config
)

__all__ = [
    "DATASETS",
    "DATASET_ALIASES",
    "PROJECT_ROOT",
    "get_project_root",
    "normalize_dataset_name",
    "list_available_datasets",
    "resolve_dataset_config",
    "validate_dataset_config"
]
