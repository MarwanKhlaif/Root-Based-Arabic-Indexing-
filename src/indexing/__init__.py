"""
Indexing and stemming subsystem for Arabic Information Retrieval.
Milestone 3 - Systems S0–S5.
"""

from .stemmers import get_stemmer, stem_larkey, stem_isri, stem_khoja

__all__ = [
    "get_stemmer",
    "stem_larkey",
    "stem_isri",
    "stem_khoja",
]
