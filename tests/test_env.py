"""
Unit and Integration Tests for PostgreSQL 16 Environment and Baseline Failure.
Milestone 1 Test Suite.
"""

import json
import os
import pytest
import psycopg
from src.db.connection import get_connection, get_db_config
from src.db.verify_env import run_environment_checks
from src.benchmarks.verify_baseline_failure import execute_baseline_demonstration


@pytest.mark.db
def test_database_connection():
    """Verify that PostgreSQL is running and accepts connections."""
    conn = get_connection()
    assert conn is not None
    conn.close()


@pytest.mark.db
def test_environment_verification_passes():
    """Verify that all environment verification checks in verify_env pass."""
    report = run_environment_checks()
    assert report["connected"] is True
    assert report["is_pg16"] is True
    assert report["server_encoding"] == "UTF8"
    assert "pg_trgm" in report["extensions"]
    assert "unaccent" in report["extensions"]
    assert len(report["missing_extensions"]) == 0
    assert report["has_native_arabic_root_ts"] is False
    assert report["has_builtin_arabic_ts"] is True
    assert report["builtin_arabic_matches_root"] is False
    assert len(report["european_stemmers"]) >= 15
    assert report["checks_passed"] is True


@pytest.mark.db
def test_baseline_failure_query_evaluates_false():
    """Verify that the supervisor's baseline query evaluates to FALSE."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_tsvector('simple', 'يكتبون في المكتبة') @@ to_tsquery('simple', 'كتب');")
            match_result = cur.fetchone()[0]
            assert match_result is False


@pytest.mark.db
def test_baseline_tsvector_tokens():
    """Verify exact tokenization of 'يكتبون في المكتبة' under simple config."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_tsvector('simple', 'يكتبون في المكتبة')::text;")
            tsvector_str = cur.fetchone()[0]
            assert "'المكتبة':3" in tsvector_str
            assert "'في':2" in tsvector_str
            assert "'يكتبون':1" in tsvector_str


@pytest.mark.db
def test_baseline_failure_demonstration_execution():
    """Verify the full baseline failure demonstration dataset generation."""
    data = execute_baseline_demonstration()
    assert data["match_result"] is False
    assert len(data["ts_debug_tokens"]) >= 3
    assert len(data["morphological_matrix"]) == 8

    # Verify that only the surface word 'كتب' matches in S0
    matches = [item["word"] for item in data["morphological_matrix"] if item["actual"] is True]
    assert matches == ["كتب"]


@pytest.mark.unit
def test_golden_queries_suite():
    """Verify golden queries JSON structure, schema, and count."""
    queries_path = "queries/golden_queries_20.json"
    assert os.path.exists(queries_path)

    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "queries" in data
    assert data["metadata"]["total_queries"] == 20
    assert len(data["queries"]) == 20
    assert len(data["metadata"]["categories"]) == 5

    required_keys = {
        "query_id", "query_term", "root", "root_type", "category_id",
        "category_name_en", "category_name_ar", "expected_surface_forms",
        "relevance_criteria", "expected_system_behavior", "over_stemming_risk"
    }
    for q in data["queries"]:
        assert required_keys.issubset(set(q.keys()))
        assert len(q["expected_surface_forms"]) > 0
        assert "0" in q["relevance_criteria"] and "3" in q["relevance_criteria"]


@pytest.mark.unit
def test_documentation_files_present():
    """Verify that all required Milestone 1 documentation files exist and are populated."""
    docs = [
        "docs/system_design_and_literature_review.md",
        "docs/s0_s5_specifications.md",
        "docs/baseline_failure_demo.md",
    ]
    for doc in docs:
        assert os.path.exists(doc), f"Missing document: {doc}"
        assert os.path.getsize(doc) > 1000, f"Document too small or empty: {doc}"
