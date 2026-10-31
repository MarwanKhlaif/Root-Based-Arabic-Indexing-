"""
test_indexing_config.py - Tests for Milestone 3 S1 Stopwords and S5 Trigram Configurations.
"""

import pytest
from src.db.connection import get_connection


def test_arabic_simple_configuration_registered():
    """Verify that text search configuration arabic_simple is registered in PostgreSQL."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT cfgname FROM pg_ts_config WHERE cfgname = 'arabic_simple';")
            row = cur.fetchone()
            assert row is not None, "Text search configuration 'arabic_simple' is not registered."


def test_arabic_stopwords_dictionary_registered():
    """Verify that text search dictionary arabic_stopwords is registered in PostgreSQL."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT dictname FROM pg_ts_dict WHERE dictname = 'arabic_stopwords';")
            row = cur.fetchone()
            assert row is not None, "Text search dictionary 'arabic_stopwords' is not registered."


def test_s1_stopword_removal_canonical():
    """Verify canonical prompt query: 'في المكتبة' -> stopword 'في' removed, 'المكتبة' preserved."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_tsvector('arabic_simple', 'في المكتبة')::text;")
            res = cur.fetchone()[0]
            # 'في' must NOT be in tsvector, and 'المكتبة' (or 'المكتبه') must be present at pos 2
            assert "'في'" not in res, f"Stopword 'في' was not removed: {res}"
            assert "'المكتبة':2" in res or "'المكتبه':2" in res, f"Target word not preserved correctly: {res}"


def test_s1_plainto_tsquery_stopword_stripping():
    """Verify query-time stopword elimination."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT plainto_tsquery('arabic_simple', 'في المكتبة')::text;")
            res = cur.fetchone()[0]
            assert "'المكتبة'" in res or "'المكتبه'" in res
            assert "'في'" not in res


def test_s1_normalized_text_stopword_handling():
    """Verify that normalized Arabic text has stopwords filtered out."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_tsvector('arabic_simple', 'كتب الطالب في المكتبه كتابا مفيدا عن تاريخ العلوم')::text;")
            res = cur.fetchone()[0]
            assert "'في'" not in res
            assert "'عن'" not in res
            assert "'كتب':1" in res
            assert "'الطالب':2" in res
            assert "'المكتبه':4" in res


def test_s5_pg_trgm_active():
    """Verify pg_trgm extension is active."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_trgm';")
            assert cur.fetchone() is not None


def test_s5_similarity_collapse_on_long_document():
    """
    Verify Reviewer 1's adversarial advisory:
    On real 500-word Wikipedia articles, similarity() collapses to < 0.02 (well below 0.3),
    causing the % operator to return False, while word_similarity() is 1.0 (operator %> returns True).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    similarity(%s, s5_trgm_text) AS full_sim,
                    word_similarity(%s, s5_trgm_text) AS word_sim,
                    (s5_trgm_text %%> %s) AS op_word_sim,
                    (s5_trgm_text %% %s) AS op_full_sim
                FROM docs
                WHERE id <= 500 AND s5_trgm_text LIKE '%%المكتبه%%'
                LIMIT 1;
            """, ('المكتبه', 'المكتبه', 'المكتبه', 'المكتبه'))
            row = cur.fetchone()
            assert row is not None, "No document found containing 'المكتبه'"
            full_sim, word_sim, op_word_sim, op_full_sim = row

            # Full document similarity drops below 0.02 on long Wikipedia articles
            assert full_sim < 0.02, f"Expected full similarity < 0.02, got {full_sim}"
            # Word similarity detects exact match as 1.0
            assert word_sim == 1.0, f"Expected word_similarity 1.0, got {word_sim}"
            # Word similarity operator %> is TRUE
            assert op_word_sim is True, "Expected word similarity operator %> to be TRUE"
            # Full similarity operator % is FALSE (because full_sim < 0.3)
            assert op_full_sim is False, "Expected full similarity operator % to be FALSE"


def test_s5_like_substring_acceleration():
    """Verify LIKE operator behavior on normalized trigram text."""
    long_article = "مقدمة المقالة. " * 50 + "تأسست المكتبه الوطنية عام 1920. " + "خاتمة المقالة. " * 50
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT (%s LIKE '%%المكتبه%%') AS match_result;", (long_article,))
            assert cur.fetchone()[0] is True
