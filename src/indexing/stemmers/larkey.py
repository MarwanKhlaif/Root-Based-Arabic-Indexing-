"""
Larkey Light10 Arabic Stemmer (System S2).

Reference:
    Larkey, L. S., Ballesteros, L., & Connell, M. E. (2002).
    Improving STEM: Light stemming and word clusters for Arabic information retrieval.
    In Proceedings of the 25th annual international ACM SIGIR conference on
    Research and development in information retrieval (pp. 370-374).
"""

from typing import Optional, Tuple


LARKEY_PREFIXES: Tuple[str, ...] = (
    "وال",
    "فال",
    "كال",
    "بال",
    "لل",
    "ال",
)

LARKEY_SUFFIXES: Tuple[str, ...] = (
    "ها",
    "ان",
    "ات",
    "ون",
    "ين",
    "يه",
    "ية",
    "ه",
    "ة",
    "ي",
)

MIN_STEM_LEN: int = 3


def stem_token(token: str, min_stem_len: int = MIN_STEM_LEN) -> str:
    """
    Stem a single Arabic token using Larkey light10 affix stripping.

    Rules:
    1. Strip one prefix from ('وال', 'فال', 'كال', 'بال', 'لل', 'ال')
       if remaining length >= min_stem_len (default 3).
    2. Strip one suffix from ('ها', 'ان', 'ات', 'ون', 'ين', 'يه', 'ية', 'ه', 'ة', 'ي')
       if remaining length >= min_stem_len (default 3).

    Args:
        token: Normalized Arabic token string.
        min_stem_len: Minimum length of remaining stem (>= 3).

    Returns:
        Stemmed token string.
    """
    if not token or len(token) < min_stem_len:
        return token

    stem = token

    # Prefix stripping (longest matching prefix first)
    for prefix in LARKEY_PREFIXES:
        if stem.startswith(prefix) and len(stem) - len(prefix) >= min_stem_len:
            stem = stem[len(prefix):]
            break

    # Suffix stripping (longest matching suffix first)
    for suffix in LARKEY_SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= min_stem_len:
            stem = stem[:-len(suffix)]
            break

    return stem


class LarkeyStemmer:
    """Larkey Light10 Stemmer class adhering to interface contract."""

    def __init__(
        self,
        prefixes: Tuple[str, ...] = LARKEY_PREFIXES,
        suffixes: Tuple[str, ...] = LARKEY_SUFFIXES,
        min_stem_len: int = MIN_STEM_LEN,
    ):
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.min_stem_len = min_stem_len

    def stem_token(self, token: str) -> str:
        """Stem a single word token."""
        return stem_token(token, self.min_stem_len)

    def stem_text(self, text: Optional[str]) -> str:
        """
        Stem whitespace-delimited text according to IR interface contract.

        Args:
            text: Normalized Arabic text string.

        Returns:
            Space-delimited stemmed string suitable for to_tsvector('simple', ...).
        """
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""
        return " ".join(stem_token(t, self.min_stem_len) for t in tokens)

    def __call__(self, text: Optional[str]) -> str:
        return self.stem_text(text)


_DEFAULT_LARKEY = LarkeyStemmer()


def stem_text(text: Optional[str]) -> str:
    """Module-level pure function adhering to interface contract."""
    return _DEFAULT_LARKEY.stem_text(text)


__all__ = [
    "LarkeyStemmer",
    "stem_token",
    "stem_text",
    "LARKEY_PREFIXES",
    "LARKEY_SUFFIXES",
    "MIN_STEM_LEN",
]
