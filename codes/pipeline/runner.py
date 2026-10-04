"""
Pipeline Runner for Full-Triple-Matcher (FTM).
Provides the official programmatic interface for running FTM experiments.
"""

import os
import sys
from typing import Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
entity_matching_dir = os.path.abspath(os.path.join(current_dir, "..", "entity-matching"))
config_dir = os.path.abspath(os.path.join(current_dir, "..", "config"))
utils_dir = os.path.abspath(os.path.join(current_dir, "..", "utils"))

for p in [entity_matching_dir, config_dir, utils_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from entity_matching import run_pipeline, EntityMatcherPipeline
from dataset_config import resolve_dataset_config, validate_dataset_config, list_available_datasets, DATASETS

__all__ = [
    "run_pipeline",
    "EntityMatcherPipeline",
    "resolve_dataset_config",
    "validate_dataset_config",
    "list_available_datasets",
    "DATASETS"
]
