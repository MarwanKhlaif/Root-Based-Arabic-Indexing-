"""
Stemmer Factory and Common Exports for Systems S0–S5.

Provides unified factory access `get_stemmer(system_id: str)` and re-exports
for all indexing stemmers:
- S2: Larkey light10 affix-stripper (`larkey.py`)
- S3: ISRI morphological pattern root extractor (`isri.py`)
- S4: Khoja dictionary-validated root extractor (`khoja.py`)
- S0 / S1: Identity surface passers
- S5: Identity passer for character trigram text indexing
"""

from typing import Callable, Optional
try:
    from .larkey import stem_text as stem_larkey, LarkeyStemmer
    from .isri import stem_text as stem_isri, ISRIStemmerWrapper
    from .khoja import stem_text as stem_khoja, KhojaStemmer
except ImportError:
    # Allow standalone testing within blueprints
    from blueprint_larkey import stem_text as stem_larkey, LarkeyStemmer
    from blueprint_isri import stem_text as stem_isri, ISRIStemmerWrapper
    from blueprint_khoja import stem_text as stem_khoja, KhojaStemmer


def _identity_stem(text: Optional[str]) -> str:
    """Identity stemmer for surface systems (S0, S1) and trigram text (S5)."""
    if not text:
        return ""
    return " ".join(text.split())


def get_stemmer(system_id: str) -> Callable[[Optional[str]], str]:
    """
    Retrieve stemmer function for a given system ID or algorithm name.

    Args:
        system_id: One of 's0', 's1', 's2', 's3', 's4', 's5' (case-insensitive)
                   or descriptive name ('surface', 'simple', 'larkey', 'light10',
                   'isri', 'khoja', 'trgm', 'trigram').

    Returns:
        Callable pure function conforming to `stem_text(text: Optional[str]) -> str`.

    Raises:
        ValueError: If system_id is unrecognized.
    """
    if not system_id:
        raise ValueError("system_id cannot be empty")

    sys_key = system_id.strip().lower()

    if sys_key in ("s0", "s1", "surface", "simple", "arabic_simple"):
        return _identity_stem
    elif sys_key in ("s2", "larkey", "light10", "light"):
        return stem_larkey
    elif sys_key in ("s3", "isri", "pattern"):
        return stem_isri
    elif sys_key in ("s4", "khoja", "root", "dictionary"):
        return stem_khoja
    elif sys_key in ("s5", "trgm", "trigram", "pg_trgm"):
        return _identity_stem
    else:
        raise ValueError(
            f"Unknown stemmer system_id: {system_id!r}. "
            f"Supported systems: 's0', 's1', 's2', 's3', 's4', 's5'."
        )


__all__ = [
    "get_stemmer",
    "stem_larkey",
    "LarkeyStemmer",
    "stem_isri",
    "ISRIStemmerWrapper",
    "stem_khoja",
    "KhojaStemmer",
]
