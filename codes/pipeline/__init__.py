"""
Full-Triple-Matcher (FTM) Pipeline Module.
Provides high-level dataset/KG-agnostic execution pipeline.
"""

from .runner import run_pipeline, EntityMatcherPipeline

__all__ = [
    "run_pipeline",
    "EntityMatcherPipeline"
]
