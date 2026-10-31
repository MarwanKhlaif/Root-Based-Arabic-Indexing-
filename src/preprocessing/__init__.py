"""Preprocessing and Normalization Module for Arabic IR Indexing."""

from src.preprocessing.normalize import (
    clean_whitespace,
    normalize_alif,
    normalize_arabic,
    normalize_digits,
    normalize_ta_marbuta,
    normalize_ya,
    remove_bidi_and_control_chars,
    remove_punctuation_and_symbols,
    strip_dagger_alif,
    strip_quranic_annotations,
    strip_tashkeel,
    strip_tatweel,
)

__all__ = [
    "normalize_arabic",
    "strip_tashkeel",
    "strip_dagger_alif",
    "strip_quranic_annotations",
    "strip_tatweel",
    "normalize_alif",
    "normalize_ya",
    "normalize_ta_marbuta",
    "normalize_digits",
    "remove_punctuation_and_symbols",
    "remove_bidi_and_control_chars",
    "clean_whitespace",
]
