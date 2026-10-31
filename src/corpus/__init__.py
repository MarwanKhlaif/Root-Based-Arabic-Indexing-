"""Corpus acquisition, filtering, and ingestion module."""

from src.corpus.acquire_wikipedia import (
    is_valid_article,
    run_acquisition_pipeline,
    EXCLUDED_NAMESPACES,
)

__all__ = [
    "is_valid_article",
    "run_acquisition_pipeline",
    "EXCLUDED_NAMESPACES",
]
