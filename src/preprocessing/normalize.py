"""
Strict Unicode-Based Arabic Text Normalization Engine.

Handles 11 categories of Unicode transformations:
1. Tashkeel / diacritics stripping (U+064B..U+0652)
2. Dagger Alif (U+0670) and Quranic annotation marks (U+0610..U+061A, U+0653..U+065F, U+06D6..U+06ED)
3. Tatweel / Kashida removal (U+0640)
4. Alif unification (أ, إ, آ, ٱ -> ا)
5. Ya / Alif Maqsura unification (ى, Persian ی, Pashto ې -> ي)
6. Ta Marbuta normalization (ة -> ه) with configurable toggle `normalize_teh=True`
7. Numeral conversion (Arabic-Indic ٠-٩ and Eastern ۰-۹ -> ASCII 0-9)
8. Punctuation, symbols, quotes, and brackets replacement with whitespace
9. Zero-width and BiDi control characters removal (U+200B..U+200F, U+202A..U+202E, U+2066..U+2069, U+FEFF, \x00)
10. Whitespace collapsing and trimming
11. Canonical NFKC decomposition and null-byte (\x00) safety for PostgreSQL COPY
"""

import re
import unicodedata
from typing import Optional

# Category 1: Tashkeel / Short Vowels & Tanwin & Shaddah & Sukun
TASHKEEL_REGEX = re.compile(r"[\u064B-\u0652]")

# Category 2: Dagger Alif & Quranic Annotation Marks
DAGGER_ALIF_REGEX = re.compile(r"\u0670")
TATWEEL_DAGGER_ALIF_REGEX = re.compile(r"\u0640\u0670")
QURANIC_MARKS_REGEX = re.compile(r"[\u0610-\u061A\u0653-\u065F\u06D6-\u06ED]")

# Category 3: Tatweel / Kashida
TATWEEL_REGEX = re.compile(r"\u0640")

# Category 4: Alif Unification (أ, إ, آ, ٱ -> ا)
ALIF_REGEX = re.compile(r"[\u0622\u0623\u0625\u0671]")

# Category 5: Ya / Alif Maqsura Unification (ى, Persian ی, Pashto ې -> ي)
YA_REGEX = re.compile(r"[\u0649\u06CC\u06D0]")

# Category 6: Ta Marbuta (ة -> ه)
TA_MARBUTA_REGEX = re.compile(r"\u0629")

# Category 7: Numeral Translation Mapping (٠-٩ and ۰-۹ -> 0-9)
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EASTERN_ARABIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WESTERN_DIGITS = "0123456789"
DIGIT_MAP = str.maketrans(
    ARABIC_INDIC_DIGITS + EASTERN_ARABIC_DIGITS,
    WESTERN_DIGITS + WESTERN_DIGITS,
)

# Category 8: Punctuation, Quotes, Brackets, and Math Symbols -> Replaced by Space
PUNCT_REGEX = re.compile(
    r"[\u060C\u061B\u061F\u066A\u066B\u066C\u066D\u06D4«»\"\'\(\)\[\]\{\}⟨⟩.,;:!؟?/\-+*%=#@^&|_~`\\$<>]"
)

# Category 9: Zero-Width, BiDi Control Characters, BOM, and C-string Nulls
BIDI_CONTROL_REGEX = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\x00]")

# Category 10: Whitespace
WHITESPACE_REGEX = re.compile(r"\s+")


def strip_tashkeel(text: Optional[str]) -> str:
    """Strip basic Arabic vocalization diacritics (Harakat, Tanwin, Shaddah, Sukun)."""
    if not text:
        return ""
    return TASHKEEL_REGEX.sub("", text)


def strip_dagger_alif(text: Optional[str]) -> str:
    """Strip standalone Dagger Alif (U+0670)."""
    if not text:
        return ""
    return DAGGER_ALIF_REGEX.sub("", text)


def strip_quranic_annotations(text: Optional[str]) -> str:
    """Strip extended Quranic annotation signs, honorifics, and pause marks."""
    if not text:
        return ""
    return QURANIC_MARKS_REGEX.sub("", text)


def strip_tatweel(text: Optional[str]) -> str:
    """Strip Tatweel / Kashida typographic stretching character (U+0640)."""
    if not text:
        return ""
    return TATWEEL_REGEX.sub("", text)


def normalize_alif(text: Optional[str]) -> str:
    """Unify all Alif variants (أ, إ, آ, ٱ) into bare Alif (ا)."""
    if not text:
        return ""
    return ALIF_REGEX.sub("\u0627", text)


def normalize_ya(text: Optional[str]) -> str:
    """Unify Alef Maksura (ى) and foreign Yeh variants (ی, ې) into Arabic Yeh (ي)."""
    if not text:
        return ""
    return YA_REGEX.sub("\u064A", text)


def normalize_ta_marbuta(text: Optional[str]) -> str:
    """Normalize Teh Marbuta (ة) to Heh (ه)."""
    if not text:
        return ""
    return TA_MARBUTA_REGEX.sub("\u0647", text)


def normalize_digits(text: Optional[str]) -> str:
    """Convert Arabic-Indic (٠-٩) and Eastern (۰-۹) digits to ASCII (0-9)."""
    if not text:
        return ""
    return text.translate(DIGIT_MAP)


def remove_punctuation_and_symbols(text: Optional[str]) -> str:
    """Replace punctuation, symbols, quotes, and brackets with a space."""
    if not text:
        return ""
    return PUNCT_REGEX.sub(" ", text)


def remove_bidi_and_control_chars(text: Optional[str]) -> str:
    """Delete invisible zero-width and BiDi direction control characters."""
    if not text:
        return ""
    return BIDI_CONTROL_REGEX.sub("", text)


def clean_whitespace(text: Optional[str]) -> str:
    """Collapse consecutive whitespace into a single space and strip boundaries."""
    if not text:
        return ""
    return WHITESPACE_REGEX.sub(" ", text).strip()


def normalize_arabic(text: Optional[str], normalize_teh: bool = True) -> str:
    """
    Execute full 11-category Unicode Arabic normalization pipeline.

    Args:
        text: Raw input string containing Arabic text.
        normalize_teh: If True (default), normalizes ة -> ه. If False, preserves ة.

    Returns:
        Clean, normalized, space-delimited string safe for PostgreSQL COPY and IR indexing.
    """
    if not text:
        return ""

    # Category 11: NFKC decomposition of presentation form ligatures (e.g. Lam-Alef \uFEFB)
    text = unicodedata.normalize("NFKC", text)

    # Category 9: Remove zero-width, BiDi control characters, and null bytes (\x00)
    text = remove_bidi_and_control_chars(text)

    # Category 2: Uthmani script resolution: tatweel + dagger alif (ـٰ) represents medial alif
    text = TATWEEL_DAGGER_ALIF_REGEX.sub("\u0627", text)

    # Category 1 & 2: Strip Harakat, Dagger Alif, and Quranic marks
    text = strip_tashkeel(text)
    text = strip_dagger_alif(text)
    text = strip_quranic_annotations(text)

    # Category 3: Strip Tatweel
    text = strip_tatweel(text)

    # Category 4: Unify Alif variants to bare Alif
    text = normalize_alif(text)

    # Category 5: Unify Ya / Alif Maqsura to dotted Yeh
    text = normalize_ya(text)

    # Category 6: Configurable Ta Marbuta normalization
    if normalize_teh:
        text = normalize_ta_marbuta(text)

    # Category 7: Normalize numerals to ASCII digits
    text = normalize_digits(text)

    # Category 8: Replace punctuation and symbols with space
    text = remove_punctuation_and_symbols(text)

    # Category 10: Collapse whitespace and strip
    text = clean_whitespace(text)

    return text


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
