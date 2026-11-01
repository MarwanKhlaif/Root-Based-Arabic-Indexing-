#!/usr/bin/env python3
"""
Adversarial Multi-Word Search Test Harness (Challenger Web 1).
Empirically tests complex, boundary, and edge-case multi-word queries
across systems S0 to S5 against the live PostgreSQL database and FastAPI endpoints.
"""

import time
import json
import urllib.request
import pytest
from starlette.testclient import TestClient

from src.web.app import app
from src.web.search_service import (
    check_db_health,
    search_all_systems,
    execute_system_search,
    transform_query,
)

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_live_database_health():
    """Verify live PostgreSQL has 100,000 docs and all 6 GIN indexes."""
    health = check_db_health()
    assert health["connected"] is True, f"Database not connected: {health['error']}"
    assert health["total_documents"] == 100000, f"Expected 100,000 docs, got {health['total_documents']}"
    assert len(health["indexes_active"]) == 6, f"Expected 6 active GIN indexes, got {health['indexes_active']}"


# ==============================================================================
# TEST SUITE A: 5-to-8-word Natural Multi-Word Queries
# ==============================================================================
COMPLEX_QUERIES = [
    "تطور علوم الهندسة والجبر في العصر العباسي",
    "تاريخ الطب والصيدلة عند العرب والمسلمين",
    "أثر الحضارة الإسلامية في النهضة الأوروبية الحديثة",
    "نشأة وتطور علم الفلك والمراصد الفلكية",
    "مصادر ومراجع النحو والصرف في البصرة والكوفة",
]

@pytest.mark.parametrize("query", COMPLEX_QUERIES)
def test_complex_multi_word_queries(query):
    """
    Verify 5-to-8 word natural queries return non-zero results across S0-S5,
    scores are non-negative, and results are sorted by score descending.
    """
    res = search_all_systems(query, limit=10)
    assert res["is_demo_mode"] is False, "Expected live search, but demo mode was returned"
    
    systems = res["systems"]
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        assert s in systems, f"System {s} missing from results"
        sys_res = systems[s]
        assert sys_res["error"] is None, f"{s} returned error: {sys_res['error']}"
        results = sys_res["results"]
        # Non-zero return check
        assert len(results) > 0, f"{s} returned 0 results for complex query: '{query}'"
        # Latency check: should be under 12 seconds
        assert sys_res["latency_ms"] < 12000, f"{s} latency exceeded 12s: {sys_res['latency_ms']}ms"
        
        # Verify scores are non-negative and descending
        scores = [r["score"] for r in results]
        for score in scores:
            assert score >= 0.0, f"{s} has negative score: {score}"
        
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"{s} scores NOT sorted descending for query '{query}': "
                f"rank {i+1} score {scores[i]} < rank {i+2} score {scores[i+1]}"
            )


# ==============================================================================
# TEST SUITE B: Stopwords-Only and Stopword-Heavy Queries
# ==============================================================================
STOPWORD_QUERIES = [
    "من هو الذي كان في",
    "ما هي تلك التي كانت هناك",
    "كان في ذلك الوقت بعض من هؤلاء",
    "إلى أين ومن أين وفي أي مكان",
]

@pytest.mark.parametrize("query", STOPWORD_QUERIES)
def test_stopword_queries(query):
    """
    Verify queries composed mostly or entirely of Arabic stopwords do not crash,
    do not time out, and return results or handle gracefully.
    """
    res = search_all_systems(query, limit=5)
    assert res["is_demo_mode"] is False
    systems = res["systems"]
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        assert s in systems
        sys_res = systems[s]
        assert sys_res["error"] is None, f"{s} crashed/errored on stopword query '{query}': {sys_res['error']}"
        scores = [r["score"] for r in sys_res["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{s} scores not descending for '{query}'"


# ==============================================================================
# TEST SUITE C: Punctuation, Quotes, and Boolean Terms
# ==============================================================================
BOOLEAN_PUNCTUATION_QUERIES = [
    '"بيت الحكمة" أو "دار العلم"',
    '"الرياضيات والفلك" أو "الهندسة المعمارية"',
    'تاريخ الأدب العربي - الجاهلي والإسلامي',
    'ابن سينا: القانون في الطب (مخطوطة)',
    '"جامعة الأزهر" القاهرة',
    'مكتبة الإسكندرية / دار الكتب والوثائق القومية',
]

@pytest.mark.parametrize("query", BOOLEAN_PUNCTUATION_QUERIES)
def test_boolean_punctuation_queries(query):
    """
    Verify queries with quotes, colons, slashes, dashes, and Arabic 'أو'
    execute without SQL syntax errors and return results.
    """
    res = search_all_systems(query, limit=5)
    assert res["is_demo_mode"] is False
    systems = res["systems"]
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        sys_res = systems[s]
        assert sys_res["error"] is None, f"{s} error on query '{query}': {sys_res['error']}"
        assert len(sys_res["results"]) > 0, f"{s} returned 0 results for '{query}'"
        scores = [r["score"] for r in sys_res["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{s} scores not descending for '{query}'"


# ==============================================================================
# TEST SUITE D: Unmatched / Non-existent Words Combined with Matched Words
# ==============================================================================
UNMATCHED_MIXED_QUERIES = [
    "خوارزميات زيبرازوربيوس العباسية القديمة",
    "كتاب بروتوكولات حكماء إكسترازون",
    "مستشفى بغداد العضدي الفراكتالي الزوردابادي",
]

@pytest.mark.parametrize("query", UNMATCHED_MIXED_QUERIES)
def test_unmatched_mixed_queries(query):
    """
    When a multi-word query contains non-existent words alongside common words,
    the fallback tier should still retrieve documents containing the real words!
    """
    res = search_all_systems(query, limit=5)
    assert res["is_demo_mode"] is False
    systems = res["systems"]
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        sys_res = systems[s]
        assert sys_res["error"] is None, f"{s} error on mixed query '{query}': {sys_res['error']}"
        assert len(sys_res["results"]) > 0, f"{s} returned 0 results for mixed query '{query}' (fallback failed!)"
        scores = [r["score"] for r in sys_res["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{s} scores not descending for '{query}'"


# ==============================================================================
# TEST SUITE E: Rare and Compound Morphological Forms
# ==============================================================================
MORPHOLOGICAL_QUERIES = [
    "فسيكفيكهم الله وهو السميع العليم",
    "استغفاراتهم وتوسلاتهم في كتب التصوف",
    "فالمتوكلون على الله في التاريخ الإسلامي",
]

@pytest.mark.parametrize("query", MORPHOLOGICAL_QUERIES)
def test_morphological_queries(query):
    """
    Test complex morphological forms, affixes, and enclitics across stemmers.
    """
    res = search_all_systems(query, limit=5)
    assert res["is_demo_mode"] is False
    systems = res["systems"]
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        sys_res = systems[s]
        assert sys_res["error"] is None, f"{s} error on query '{query}': {sys_res['error']}"
        assert len(sys_res["results"]) > 0, f"{s} returned 0 results for '{query}'"
        scores = [r["score"] for r in sys_res["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{s} scores not descending for '{query}'"


# ==============================================================================
# TEST SUITE F: Pathological, Injection & Special Characters
# ==============================================================================
PATHOLOGICAL_QUERIES = [
    "100% طبيعي وسليم",
    "أ_ب_ج درجات الحساب",
    '"غير مكتمل الاقتباس',
    "تاريخ & جغرافيا | فلك ! طب",
    "<script>alert('test')</script> تاريخ",
    "SELECT * FROM docs WHERE id = 1",
    "'' OR 1=1 --",
    "\\x00 نصوص غريبة",
]

@pytest.mark.parametrize("query", PATHOLOGICAL_QUERIES)
def test_pathological_queries(query):
    """
    Verify malicious, SQL injection, wildcard, and unclosed quotes queries
    do not crash the database or server and handle cleanly.
    """
    res = search_all_systems(query, limit=5)
    assert res["is_demo_mode"] is False
    systems = res["systems"]
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        sys_res = systems[s]
        assert sys_res["error"] is None, f"{s} raised unhandled error on '{query}': {sys_res['error']}"
        scores = [r["score"] for r in sys_res["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{s} scores not descending for '{query}'"


# ==============================================================================
# TEST SUITE G: Ultra-Long Query (20+ words)
# ==============================================================================
def test_ultra_long_query():
    """
    Test 20+ words query to evaluate latency and robustness under heavy token load.
    """
    query = (
        "في إطار التاريخ الإسلامي والحضارة العربية تميزت بغداد ودمشق والقاهرة "
        "بإنشاء المدارس والمكتبات والمستشفيات والعلماء الذين ساهموا في تطور المعرفة الإنسانية"
    )
    t0 = time.perf_counter()
    res = search_all_systems(query, limit=10)
    dur = time.perf_counter() - t0
    
    assert res["is_demo_mode"] is False
    assert dur < 10.0, f"Ultra long query took too long: {dur:.2f}s"
    
    systems = res["systems"]
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        sys_res = systems[s]
        assert sys_res["error"] is None, f"{s} failed on long query: {sys_res['error']}"
        assert len(sys_res["results"]) > 0, f"{s} returned 0 results for ultra-long query"
        scores = [r["score"] for r in sys_res["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{s} scores not descending for long query"


# ==============================================================================
# TEST SUITE H: REST API Endpoint End-to-End Test via Live HTTP Server
# ==============================================================================
def test_rest_api_via_live_docker():
    """Verify live HTTP server at http://localhost:8000/api/search handles adversarial queries."""
    payload = json.dumps({"query": "تطور علوم الهندسة والجبر في العصر العباسي", "limit": 10}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/search",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
    assert data["db_connected"] is True
    assert data["is_demo_mode"] is False
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        assert len(data["systems"][s]["results"]) > 0
        scores = [r["score"] for r in data["systems"][s]["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{s} scores not sorted via live HTTP"
