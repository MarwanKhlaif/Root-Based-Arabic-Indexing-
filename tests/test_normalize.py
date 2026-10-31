"""
Comprehensive Unit Test Suite for Arabic Preprocessing & Normalization Engine.
Verifies all 11 Unicode transformation categories and linguistic edge cases.
Milestone 2 Test Suite (27 Unit Tests).
"""

import pytest
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


@pytest.mark.unit
def test_strip_tashkeel_basic_vowels():
    """Test 1: Strip basic short vowels (Fatha, Damma, Kasra)."""
    raw = "مَكْتَبَةُ الكُتُبِ المُفِيدَةِ"
    assert strip_tashkeel(raw) == "مكتبة الكتب المفيدة"
    assert normalize_arabic(raw) == "مكتبه الكتب المفيده"


@pytest.mark.unit
def test_strip_tanwin():
    """Test 2: Strip Tanwin Fath, Damm, and Kasr."""
    raw = "كِتَاباً جَدِيداً وَقَلَمٌ فِي حَقِيبَةٍ"
    assert strip_tashkeel(raw) == "كتابا جديدا وقلم في حقيبة"
    assert normalize_arabic(raw) == "كتابا جديدا وقلم في حقيبه"


@pytest.mark.unit
def test_strip_shaddah():
    """Test 3: Strip consonant gemination Shaddah (U+0651)."""
    raw = "مُحَمَّدٌ نَبِيٌّ وَمُعَلِّمٌ قَدِيرٌ"
    assert strip_tashkeel(raw) == "محمد نبي ومعلم قدير"
    assert normalize_arabic(raw) == "محمد نبي ومعلم قدير"


@pytest.mark.unit
def test_strip_sukun():
    """Test 4: Strip Sukun vowelless consonant marker (U+0652)."""
    raw = "قُمْ فَأَنْذِرْ"
    assert strip_tashkeel(raw) == "قم فأنذر"
    assert normalize_arabic(raw) == "قم فانذر"


@pytest.mark.unit
def test_strip_dagger_alif_classical_words():
    """Test 5: Strip Dagger Alif (U+0670) in classical demonstratives and divine names."""
    raw = "هٰذا بَرَكَاتُ الرَّحْمٰنِ وَإِلٰهُ الكَوْنِ ذٰلِكَ الكِتَابُ"
    expected = "هذا بركات الرحمن واله الكون ذلك الكتاب"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_strip_quranic_annotations_and_uthmani_ligatures():
    """Test 6: Correctly handle Uthmani Quranic script and strip combining marks."""
    raw = "وَٱلصَّـٰٓفَّـٰتِ صَفًّۭا"
    expected = "والصافات صفا"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_strip_tatweel_single():
    """Test 7: Strip single Tatweel / Kashida (U+0640)."""
    raw = "جـمـيـل"
    assert strip_tatweel(raw) == "جميل"
    assert normalize_arabic(raw) == "جميل"


@pytest.mark.unit
def test_strip_tatweel_multiple_consecutive():
    """Test 8: Strip long elongated sequences of Tatweel."""
    raw = "الــــــــعــــــــلــــــــمُ نُـــــورٌ"
    assert normalize_arabic(raw) == "العلم نور"


@pytest.mark.unit
def test_strip_tatweel_with_interleaved_diacritics():
    """Test 9: Strip Tatweels interleaved with short vowels and diacritics."""
    raw = "تـَـفـْـعـِـيـل"
    assert normalize_arabic(raw) == "تفعيل"


@pytest.mark.unit
def test_normalize_alif_hamza_above():
    """Test 10: Unify Alef with Hamza Above (أ -> ا)."""
    raw = "أَحْمَدُ أَرْسَلَ أَمْرًا أَمَامَ أُمِّهِ"
    expected = "احمد ارسل امرا امام امه"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_alif_hamza_below():
    """Test 11: Unify Alef with Hamza Below (إ -> ا)."""
    raw = "إِبْرَاهِيمُ سَافَرَ إِلَى إِسْبَانِيَا"
    expected = "ابراهيم سافر الي اسبانيا"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_alif_madda():
    """Test 12: Unify Alef with Madda Above (آ -> ا)."""
    raw = "آثَارٌ تَارِيخِيَّةٌ وَآرَاءٌ قَيِّمَةٌ"
    expected = "اثار تاريخيه واراء قيمه"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_alif_wasla():
    """Test 13: Unify Alef Wasla (ٱ -> ا)."""
    raw = "ٱسْتِقْلَالُ ٱلدَّوْلَةِ"
    expected = "استقلال الدوله"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_alif_mixed_sentence():
    """Test 14: Unify sentence containing all 4 Alif variants (أ, إ, آ, ٱ)."""
    raw = "أَآلْآنَ إِذَا ٱسْتَقَرَّ الأَمْرُ"
    expected = "االان اذا استقر الامر"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_alif_maqsura_to_ya():
    """Test 15: Unify Alef Maksura to dotted Yeh (ى -> ي)."""
    raw = "عَلَى مُسْتَشْفَى هُدَى قَضَى القَاضِي"
    expected = "علي مستشفي هدي قضي القاضي"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_persian_and_urdu_yeh():
    """Test 16: Unify Persian/Urdu Yeh (ی U+06CC) to Arabic Yeh (ي U+064A)."""
    raw = "دانشگاه تهران در ایران است"
    expected = "دانشگاه تهران در ايران است"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_ta_marbuta_default_enabled():
    """Test 17: Default normalization of Teh Marbuta to Heh (ة -> ه)."""
    raw = "مَكْتَبَةُ الجَامِعَةِ الكَبِيرَةِ"
    expected = "مكتبه الجامعه الكبيره"
    assert normalize_arabic(raw) == expected
    assert normalize_arabic(raw, normalize_teh=True) == expected


@pytest.mark.unit
def test_normalize_ta_marbuta_configurable_disabled():
    """Test 18: Preserving Teh Marbuta when normalize_teh=False is set."""
    raw = "مَكْتَبَةُ الجَامِعَةِ الكَبِيرَةِ"
    expected = "مكتبة الجامعة الكبيرة"
    assert normalize_arabic(raw, normalize_teh=False) == expected


@pytest.mark.unit
def test_normalize_arabic_indic_digits():
    """Test 19: Convert Arabic-Indic digits (٠-٩) to ASCII (0-9)."""
    raw = "تَأَسَّسَتْ عَامَ ١٩٩٤م وَانْتَهَتْ فِي ٢٠٢٦م"
    expected = "تاسست عام 1994م وانتهت في 2026م"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_normalize_eastern_persian_digits():
    """Test 20: Convert Eastern Arabic-Indic digits (۰-۹) to ASCII (0-9)."""
    raw = "شماره ثبت ۴۵۶۷۸۹ و ۰۱۲۳"
    expected = "شماره ثبت 456789 و 0123"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_remove_arabic_and_latin_punctuation():
    """Test 21: Replace Arabic/Latin punctuation, guillemets, and brackets with space."""
    raw = "هَلْ كَتَبَ؟ نَعَمْ، كَتَبَ؛ وَ«قَرَأَ» [الكِتَاب]!"
    expected = "هل كتب نعم كتب و قرا الكتاب"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_remove_bidi_and_control_chars():
    """Test 22: Delete invisible zero-width characters (ZWNJ, LRM, RLM)."""
    raw = "ال\u200Cعلم \u200Eوالثقافة\u200F"
    expected = "العلم والثقافه"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_edge_case_empty_and_none():
    """Test 23: Safe handling of None, empty strings, and pure whitespace/punctuation."""
    assert normalize_arabic("") == ""
    assert normalize_arabic(None) == ""
    assert normalize_arabic("   \t\n   ") == ""
    assert normalize_arabic("... ،،، ؟؟؟ !!!") == ""


@pytest.mark.unit
def test_edge_case_mixed_language():
    """Test 24: Preservation of Latin words and numerals in bilingual context."""
    raw = "قاعدة بيانات PostgreSQL 16 للإسترجاع"
    expected = "قاعده بيانات PostgreSQL 16 للاسترجاع"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_edge_case_presentation_form_ligatures():
    """Test 25: NFKC canonical decomposition of Lam-Alef ligatures (U+FEFB, etc.)."""
    raw = "\uFEFB \uFEF7 \uFEF9 \uFEF5"
    expected = "لا لا لا لا"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_edge_case_null_bytes_safety():
    """Test 26: Strip null bytes (\\x00) to protect PostgreSQL COPY streaming."""
    raw = "نص يحتوي على \x00 بايت صفرية"
    expected = "نص يحتوي علي بايت صفريه"
    assert normalize_arabic(raw) == expected


@pytest.mark.unit
def test_comprehensive_integrated_sample():
    """Test 27: Integrated test combining all 11 normalization categories simultaneously."""
    raw = (
        "هٰذا هُوَ القُرْآنُ الكَرِيمُ في مَكْتَبَةِ الإِسْكَنْدَرِيَّةِ عام ٢٠٢٦م، "
        "بـِـسِعْر ١٠٠$! و«عَلى» شاطِئ البَحْر."
    )
    expected = (
        "هذا هو القران الكريم في مكتبه الاسكندريه عام 2026م بسعر 100 و علي شاطئ البحر"
    )
    assert normalize_arabic(raw) == expected
