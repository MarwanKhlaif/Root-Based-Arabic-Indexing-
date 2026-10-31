"""
ISRI Morphological Pattern Stemmer (System S3).

Wraps NLTK's ISRIStemmer with tokenization, short-token safety,
and IR interface contract conformance.

Reference:
    Taghva, K., Elkhoury, R., & Coombs, J. (2005).
    Arabic stemming without a root dictionary.
    In International Conference on Information Technology: Coding and Computing (ITCC'05)
    (Vol. 1, pp. 152-157). IEEE.
"""

from typing import Optional, Set
from nltk.stem.isri import ISRIStemmer


class ISRIStemmerWrapper:
    """ISRI morphological pattern stemmer wrapper adhering to IR contract."""

    def __init__(
        self,
        filter_stopwords: bool = False,
        stopwords: Optional[Set[str]] = None,
        min_token_len: int = 3,
    ):
        self.stemmer = ISRIStemmer()
        self.filter_stopwords = filter_stopwords
        self.stopwords = stopwords or set()
        self.min_token_len = min_token_len

    def stem_token(self, token: str) -> str:
        """
        Stem a single Arabic token using ISRI heuristic pattern matching.

        Args:
            token: Normalized Arabic token string.

        Returns:
            Extracted root or original token if short/non-Arabic.
        """
        if not token:
            return ""

        # Short tokens (length < min_token_len) or non-Arabic tokens are returned verbatim
        if len(token) < self.min_token_len:
            return token

        # If token does not contain any Arabic letters, return verbatim
        if not any("\u0600" <= c <= "\u06FF" for c in token):
            return token

        # Optional stopword filtering
        if self.filter_stopwords and token in self.stopwords:
            return ""

        try:
            stemmed = self.stemmer.stem(token)
            return stemmed if stemmed else token
        except Exception:
            return token

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

        stemmed_tokens = []
        for token in tokens:
            st = self.stem_token(token)
            if st:
                stemmed_tokens.append(st)

        return " ".join(stemmed_tokens)

    def __call__(self, text: Optional[str]) -> str:
        return self.stem_text(text)


_DEFAULT_ISRI = ISRIStemmerWrapper()


def stem_text(text: Optional[str]) -> str:
    """Module-level pure function adhering to interface contract."""
    return _DEFAULT_ISRI.stem_text(text)


__all__ = [
    "ISRIStemmerWrapper",
    "stem_text",
]
