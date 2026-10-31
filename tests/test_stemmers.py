"""
Comprehensive Unit Test Suite for Arabic Stemmer Engines (S2, S3, S4, Factory).

Tests cover:
- Interface contracts (stem_text pure function, None/empty/whitespace, return types)
- S2 Larkey light10 affix stripping and min-length constraint (>= 3)
- S3 ISRI morphological pattern stemmer (NLTK wrapper, short token safety)
- S4 Khoja dictionary root stemmer (4,748 canonical roots, Awzan, weak roots, geminate roots)
- Stemmer factory (`get_stemmer`) and alias resolution
- All 20 golden linguistic benchmark queries from `queries/golden_queries_20.json`
"""

import json
import os
import pytest

try:
    from src.indexing.stemmers import (
        get_stemmer,
        stem_larkey,
        LarkeyStemmer,
        stem_isri,
        ISRIStemmerWrapper,
        stem_khoja,
        KhojaStemmer,
    )
    from src.indexing.stemmers.larkey import LARKEY_PREFIXES, LARKEY_SUFFIXES
    from src.indexing.stemmers.khoja import load_canonical_roots
except ImportError:
    # Standalone blueprint import fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from blueprint_init import (
        get_stemmer,
        stem_larkey,
        LarkeyStemmer,
        stem_isri,
        ISRIStemmerWrapper,
        stem_khoja,
        KhojaStemmer,
    )
    from blueprint_larkey import LARKEY_PREFIXES, LARKEY_SUFFIXES
    from blueprint_khoja import load_canonical_roots


# ============================================================================
# 1. Interface Contract Tests
# ============================================================================
class TestInterfaceContracts:
    """Verifies that all stemmers strictly satisfy the IR interface contract."""

    @pytest.mark.parametrize("stem_fn", [stem_larkey, stem_isri, stem_khoja])
    def test_empty_and_none_input(self, stem_fn):
        assert stem_fn("") == ""
        assert stem_fn(None) == ""
        assert stem_fn("   ") == ""
        assert stem_fn("\t\n  \r") == ""

    @pytest.mark.parametrize("stem_fn", [stem_larkey, stem_isri, stem_khoja])
    def test_single_token_return_type(self, stem_fn):
        res = stem_fn("كتاب")
        assert isinstance(res, str)
        assert len(res) > 0

    @pytest.mark.parametrize("stem_fn", [stem_larkey, stem_isri, stem_khoja])
    def test_multi_token_spacing_preservation(self, stem_fn):
        text = "يكتبون في المكتبة الوطنية"
        res = stem_fn(text)
        assert isinstance(res, str)
        tokens = res.split()
        assert len(tokens) == 4
        # Multi-spaces collapsed into single space
        assert "  " not in res

    @pytest.mark.parametrize("stem_fn", [stem_larkey, stem_isri, stem_khoja])
    def test_non_arabic_tokens(self, stem_fn):
        text = "PostgreSQL 16 python3"
        res = stem_fn(text)
        assert "PostgreSQL" in res
        assert "16" in res
        assert "python3" in res

    def test_callable_class_instances(self):
        larkey = LarkeyStemmer()
        isri = ISRIStemmerWrapper()
        khoja = KhojaStemmer()

        assert larkey("الكتاب") == "كتاب"
        assert isri("يكتبون") == "كتب"
        assert khoja("المكتبة") == "كتب"


# ============================================================================
# 2. Larkey Light10 Stemmer (S2) Tests
# ============================================================================
class TestLarkeyStemmer:
    """Tests S2 Larkey light10 affix-stripper rules and constraints."""

    def test_prefix_stripping(self):
        stemmer = LarkeyStemmer()
        assert stemmer.stem_token("الكتاب") == "كتاب"
        assert stemmer.stem_token("والكتاب") == "كتاب"
        assert stemmer.stem_token("فالكتاب") == "كتاب"
        assert stemmer.stem_token("كالطالب") == "طالب"
        assert stemmer.stem_token("بالقلم") == "قلم"
        assert stemmer.stem_token("للأستاذ") == "أستاذ"

    def test_suffix_stripping(self):
        stemmer = LarkeyStemmer()
        assert stemmer.stem_token("كتابها") == "كتاب"
        assert stemmer.stem_token("طالبان") == "طالب"
        assert stemmer.stem_token("طالبات") == "طالب"
        assert stemmer.stem_token("معلمون") == "معلم"
        assert stemmer.stem_token("معلمين") == "معلم"
        assert stemmer.stem_token("مدرسته") == "مدرست"
        assert stemmer.stem_token("مكتبة") == "مكتب"
        assert stemmer.stem_token("كتابي") == "كتاب"

    def test_compound_prefix_and_suffix_stripping(self):
        stemmer = LarkeyStemmer()
        # "المكتبات": prefix "ال" stripped -> "مكتبات", suffix "ات" stripped -> "مكتب"
        assert stemmer.stem_token("المكتبات") == "مكتب"
        # "بالعملية": prefix "بال" stripped -> "عملية", suffix "ية" stripped -> "عمل"
        assert stemmer.stem_token("بالعملية") == "عمل"

    def test_min_stem_length_constraint(self):
        stemmer = LarkeyStemmer(min_stem_len=3)
        # Short words should NOT be stripped below length 3
        assert stemmer.stem_token("يد") == "يد"
        assert stemmer.stem_token("دم") == "دم"
        assert stemmer.stem_token("أب") == "أب"
        # "علي" ending in 'ي': stripping 'ي' leaves 'عل' (len 2 < 3) -> untouched
        assert stemmer.stem_token("علي") == "علي"
        # "أبي" ending in 'ي': leaves 'أب' (len 2 < 3) -> untouched
        assert stemmer.stem_token("أبي") == "أبي"

    def test_preserves_derivational_forms(self):
        """Larkey preserves derivational patterns without collapsing them to root."""
        stemmer = LarkeyStemmer()
        stem_noun = stemmer.stem_token("مدرسة")
        stem_verb = stemmer.stem_token("يدرس")
        # Light stemmer keeps them distinct
        assert stem_noun != stem_verb
        assert stem_noun == "مدرس"


# ============================================================================
# 3. ISRI Morphological Pattern Stemmer (S3) Tests
# ============================================================================
class TestISRIStemmer:
    """Tests S3 ISRI dictionary-free heuristic pattern matcher."""

    def test_regular_sound_verbs(self):
        stemmer = ISRIStemmerWrapper()
        assert stemmer.stem_token("يكتبون") == "كتب"
        assert stemmer.stem_token("يدرسون") == "درس"
        assert stemmer.stem_token("يخرجون") == "خرج"
        assert stemmer.stem_token("يعلمون") == "علم"

    def test_infix_form_viii_stripping(self):
        stemmer = ISRIStemmerWrapper()
        # Infix -ت- stripped in Form VIII
        assert stemmer.stem_token("انتصار") == "نصر"
        assert stemmer.stem_token("منتصر") == "نصر"

    def test_form_x_prefix_stripping(self):
        stemmer = ISRIStemmerWrapper()
        assert stemmer.stem_token("استخراج") == "خرج"
        assert stemmer.stem_token("استسلام") == "سلم"

    def test_broken_plurals(self):
        stemmer = ISRIStemmerWrapper()
        assert stemmer.stem_token("أشجار") == "شجر"
        assert stemmer.stem_token("قلوب") == "قلب"

    def test_proper_noun_overstemming_bahrain(self):
        """Demonstrates known over-stemming failure on Bahrain."""
        stemmer = ISRIStemmerWrapper()
        # ISRI strips -ين and reduces البحرين to بحر
        assert stemmer.stem_token("البحرين") == "بحر"


# ============================================================================
# 4. Khoja Dictionary-Based Root Stemmer (S4) Tests
# ============================================================================
class TestKhojaStemmer:
    """Tests S4 Khoja root extraction and canonical dictionary validation."""

    def test_canonical_dictionary_loading(self):
        roots = load_canonical_roots()
        assert len(roots) == 4748
        # Check canonical triliteral (3822), quadriliteral (923), and quinqueliteral (3) counts
        tri = [r for r in roots if len(r) == 3]
        quad = [r for r in roots if len(r) == 4]
        quinq = [r for r in roots if len(r) == 5]
        assert len(tri) == 3822
        assert len(quad) == 923
        assert len(quinq) == 3

    def test_sound_triliteral_roots(self):
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("كاتب") == "كتب"
        assert stemmer.stem_token("المكتبة") == "كتب"
        assert stemmer.stem_token("يكتبون") == "كتب"
        assert stemmer.stem_token("مدرسة") == "درس"
        assert stemmer.stem_token("مدارس") == "درس"
        assert stemmer.stem_token("محكمة") == "حكم"
        assert stemmer.stem_token("محاكم") == "حكم"
        assert stemmer.stem_token("تعليم") == "علم"

    def test_broken_plurals(self):
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("أشجار") == "شجر"
        assert stemmer.stem_token("قلوب") == "قلب"
        assert stemmer.stem_token("طرائق") == "طرق"
        assert stemmer.stem_token("أصحاب") == "صحب"
        assert stemmer.stem_token("صحابة") == "صحب"

    def test_assimilated_weak_roots(self):
        """Assimilated root (initial radical weak: وقف)."""
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("وقف") == "وقف"
        assert stemmer.stem_token("يقف") == "وقف"

    def test_hollow_weak_roots(self):
        """Hollow root (medial radical weak: قول)."""
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("قال") == "قول"
        assert stemmer.stem_token("يقول") == "قول"

    def test_defective_weak_roots(self):
        """Defective root (final radical weak: رمي)."""
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("رمى") == "رمي"
        assert stemmer.stem_token("يرمي") == "رمي"

    def test_doubly_weak_roots(self):
        """Doubly weak root (initial and final weak: وفي)."""
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("وفى") == "وفي"
        assert stemmer.stem_token("توفي") == "وفي"

    def test_geminate_doubled_roots(self):
        """Geminate root where 2-letter form doubles last letter."""
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("رد") == "ردد"
        assert stemmer.stem_token("حل") == "حلل"
        assert stemmer.stem_token("سد") == "سدد"

    def test_geopolitical_overstemming_bahrain(self):
        """Empirically verifies Bahrain over-stemming failure."""
        stemmer = KhojaStemmer()
        assert stemmer.stem_token("البحرين") == "بحر"

    def test_foreign_word_preservation(self):
        stemmer = KhojaStemmer()
        # Words in strange words list are preserved verbatim
        assert stemmer.stem_token("تلفزة") == "تلفزة"
        assert stemmer.stem_token("برلمان") == "برلمان"


# ============================================================================
# 5. Factory and Common Re-export Tests
# ============================================================================
class TestStemmerFactory:
    """Tests get_stemmer() factory and system dispatching."""

    def test_factory_system_resolution(self):
        s0_fn = get_stemmer("s0")
        s1_fn = get_stemmer("s1")
        s2_fn = get_stemmer("s2")
        s3_fn = get_stemmer("s3")
        s4_fn = get_stemmer("s4")
        s5_fn = get_stemmer("s5")

        test_text = "يكتبون في المكتبة"
        assert s0_fn(test_text) == "يكتبون في المكتبة"
        assert s1_fn(test_text) == "يكتبون في المكتبة"
        assert s2_fn(test_text) == "يكتب في مكتب"
        assert s3_fn(test_text) == "كتب في كتب"
        assert s4_fn(test_text) == "كتب في كتب"
        assert s5_fn(test_text) == "يكتبون في المكتبة"

    def test_factory_case_and_aliases(self):
        assert get_stemmer("S2") == stem_larkey
        assert get_stemmer("Larkey") == stem_larkey
        assert get_stemmer("light10") == stem_larkey
        assert get_stemmer("S3") == stem_isri
        assert get_stemmer("ISRI") == stem_isri
        assert get_stemmer("S4") == stem_khoja
        assert get_stemmer("Khoja") == stem_khoja
        assert get_stemmer("S5")("نص") == "نص"

    def test_factory_invalid_id_raises(self):
        with pytest.raises(ValueError):
            get_stemmer("invalid_sys")
        with pytest.raises(ValueError):
            get_stemmer("")


# ============================================================================
# 6. Golden 20 Linguistic Queries Suite Tests
# ============================================================================
class TestGoldenQueries20Coverage:
    """Verifies coverage across all 20 golden linguistic queries."""

    @pytest.fixture(scope="class")
    @classmethod
    def golden_queries_data(cls):
        queries_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "queries",
            "golden_queries_20.json",
        )
        if not os.path.isfile(queries_path):
            queries_path = "/Users/drbingo/Downloads/بحث ماليوان/queries/golden_queries_20.json"
        with open(queries_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_all_20_roots_present_in_khoja_lexicon(self, golden_queries_data):
        roots = load_canonical_roots()
        for q in golden_queries_data["queries"]:
            clean_root = q["root"].replace("-", "")
            assert (
                clean_root in roots
            ), f"Golden root {clean_root} (Query {q['query_id']}) missing from Khoja lexicon"

    def test_golden_queries_execution_across_all_stemmers(self, golden_queries_data):
        larkey_fn = get_stemmer("s2")
        isri_fn = get_stemmer("s3")
        khoja_fn = get_stemmer("s4")

        for q in golden_queries_data["queries"]:
            term = q["query_term"]
            # All stemmers must execute without error
            res_larkey = larkey_fn(term)
            res_isri = isri_fn(term)
            res_khoja = khoja_fn(term)

            assert isinstance(res_larkey, str) and len(res_larkey) > 0
            assert isinstance(res_isri, str) and len(res_isri) > 0
            assert isinstance(res_khoja, str) and len(res_khoja) > 0

    def test_q01_kitab_derivation_chain(self):
        """Q01: Sound regular verb كتب."""
        text = "كاتب مكتوب مكتبة يكتبون كتبوا كتاب"
        res_s2 = stem_larkey(text)
        res_s3 = stem_isri(text)
        res_s4 = stem_khoja(text)

        # S3 root stemmer collapses to root كتب
        assert "كتب" in res_s3
        # S4 root stemmer collapses to root كتب
        assert "كتب" in res_s4
        # S2 light stemmer preserves distinct stems
        assert "كاتب" in res_s2

    def test_q10_shajar_broken_plural(self):
        """Q10: Broken plural أشجار -> شجر."""
        text = "الأشجار باسقة والشجرة مثمرة"
        res_s2 = stem_larkey(text)
        res_s3 = stem_isri(text)
        res_s4 = stem_khoja(text)

        # Root stemmers resolve broken plural to شجر
        assert "شجر" in res_s3
        assert "شجر" in res_s4
        # Light stemmer keeps أشجار
        assert "أشجار" in res_s2

    def test_q14_qal_hollow_verb(self):
        """Q14: Hollow verb قال -> قول."""
        text = "قال الشاهد الحق ويقول الصدق"
        res_s3 = stem_isri(text)
        res_s4 = stem_khoja(text)

        assert "قول" in res_s3 or "قال" in res_s3
        assert "قول" in res_s4
