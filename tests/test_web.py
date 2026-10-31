"""
Unit and Integration Tests for Arabic IR Web Interface & Search Engine.
Tests query transformations across S0–S5, snippet highlighting,
golden queries catalog, benchmark metric loaders, and FastAPI REST endpoints.
"""

import pytest
from starlette.testclient import TestClient

from src.web.app import app
from src.web.search_service import (
    transform_query,
    generate_trigrams,
    highlight_snippet,
    load_golden_queries,
    load_benchmark_metrics,
    load_indexing_benchmarks,
    load_overstemming_data,
    get_arabic_stopwords,
    search_all_systems,
)


@pytest.fixture(scope="module")
def client():
    """Create reusable Starlette TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. SEARCH SERVICE & MORPHOLOGICAL TRANSFORMATION TESTS
# ==============================================================================

class TestQueryTransformation:
    """Test morphological transformations across S0 through S5."""

    def test_sound_verb_transformation(self):
        """Test transformation of standard sound verb 'يكتبون'."""
        res = transform_query("يكتبون")
        systems = res["systems"]

        assert "S0" in systems
        assert "S1" in systems
        assert "S2" in systems
        assert "S3" in systems
        assert "S4" in systems
        assert "S5" in systems

        assert systems["S0"]["transformed_query"] == "يكتبون"
        assert systems["S2"]["transformed_query"] == "يكتب"
        assert systems["S3"]["transformed_query"] == "كتب"
        assert systems["S4"]["transformed_query"] == "كتب"
        assert "يكت" in systems["S5"]["trigrams"]

    def test_stopwords_removal_in_s1(self):
        """Test that S1 purges grammatical particles and stopwords."""
        res = transform_query("البحث في المكتبة من أجل القراءة")
        s1 = res["systems"]["S1"]

        # 'في', 'من' should be removed
        assert "في" in s1["removed_stopwords"]
        assert "من" in s1["removed_stopwords"]
        assert "في" not in s1["tokens"]
        assert "من" not in s1["tokens"]

    def test_affix_stripping_in_s2(self):
        """Test that S2 Larkey light10 strips common prefixes and suffixes."""
        res = transform_query("المكتبات والمعلمين")
        s2 = res["systems"]["S2"]

        # 'المكتبات' -> 'مكتب', 'والمعلمين' -> 'معلم'
        tokens = s2["tokens"]
        assert "مكتب" in tokens or any("مكتب" in t for t in tokens)

    def test_empty_and_whitespace_query(self):
        """Test handling of empty or blank query strings."""
        res = transform_query("   ")
        assert res["raw_query"] == ""
        assert res["normalized_query"] == ""
        for sys_id in ["S0", "S1", "S2", "S3", "S4", "S5"]:
            assert res["systems"][sys_id]["tokens"] == []

    def test_trigram_generation(self):
        """Test character trigrams generation with boundary padding."""
        trigrams = generate_trigrams("كتب")
        # 'كتب' with '  كتب ' produces: '  ك', ' كت', 'كتب', 'تب '
        assert "كتب" in trigrams
        assert " كت" in trigrams
        assert "تب " in trigrams


class TestSnippetHighlighter:
    """Test safe HTML snippet generation and keyword highlighting."""

    def test_highlight_matching_keyword(self):
        """Test that keyword is enclosed in <mark> tag."""
        body = "هذا المقال يتحدث عن علوم المكتبات والتوثيق في العصر الحديث."
        keywords = ["المكتبات"]
        snippet = highlight_snippet(body, keywords)

        assert "<mark" in snippet
        assert "المكتبات</mark>" in snippet

    def test_xss_protection(self):
        """Test that raw HTML or script tags are escaped properly."""
        body = "<script>alert('xss')</script> مقال حول تاريخ الكتب."
        snippet = highlight_snippet(body, ["الكتب"])

        assert "<script>" not in snippet
        assert "&lt;script&gt;" in snippet

    def test_empty_body(self):
        """Test that empty or None text returns empty string."""
        assert highlight_snippet("", ["كلمة"]) == ""
        assert highlight_snippet(None, ["كلمة"]) == ""


class TestDataLoader:
    """Test loading and integrity of benchmark resources."""

    def test_golden_queries_count_and_keys(self):
        """Ensure all 20 golden queries are present with mandatory keys."""
        queries = load_golden_queries()
        assert len(queries) == 20
        for q in queries:
            assert "query_id" in q
            assert "query_term" in q
            assert "category_id" in q
            assert "root" in q
            assert "expected_surface_forms" in q

    def test_benchmark_metrics_contains_all_systems(self):
        """Ensure IR evaluation metrics summary contains all S0-S5."""
        metrics = load_benchmark_metrics()
        summary = metrics.get("systems_macro_summary", {})
        for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
            assert s in summary
            assert "macro_p_at_10" in summary[s]
            assert "macro_recall_at_100" in summary[s]
            assert "macro_map" in summary[s]
            assert "macro_ndcg_at_10" in summary[s]

    def test_indexing_benchmarks_contains_gin_sizes(self):
        """Ensure GIN index benchmarks contain MB sizes."""
        benchmarks = load_indexing_benchmarks()
        assert "benchmarks" in benchmarks
        systems = [b["system_id"] for b in benchmarks["benchmarks"]]
        for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
            assert s in systems


# ==============================================================================
# 2. FASTAPI ENDPOINT TESTS
# ==============================================================================

class TestFastAPIEndpoints:
    """Test all REST API endpoints served by the application."""

    def test_get_index_html(self, client):
        """Verify the main UI page returns status 200 with RTL layout."""
        response = client.get("/")
        assert response.status_code == 200
        assert "dir=\"rtl\"" in response.text
        assert "منظومة الفهرسة النصية" in response.text

    def test_get_status_endpoint(self, client):
        """Verify DB status check endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "connected" in data
        assert "database" in data

    def test_post_transform_endpoint(self, client):
        """Verify query transformation API endpoint."""
        response = client.post("/api/transform", json={"query": "يكتبون في المكتبة"})
        assert response.status_code == 200
        data = response.json()
        assert "systems" in data
        assert "S0" in data["systems"]
        assert "S4" in data["systems"]
        assert data["systems"]["S4"]["transformed_query"] == "كتب في كتب"

    def test_post_transform_empty_query(self, client):
        """Verify query transformation handles empty queries gracefully."""
        response = client.post("/api/transform", json={"query": ""})
        assert response.status_code == 200
        data = response.json()
        assert data["raw_query"] == ""

    def test_post_search_endpoint(self, client):
        """Verify multi-system search endpoint returns results."""
        response = client.post(
            "/api/search",
            json={"query": "كتب", "systems": ["S0", "S1", "S2", "S3", "S4", "S5"], "limit": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "systems" in data
        for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
            assert s in data["systems"]
            assert "results" in data["systems"][s]
            assert "latency_ms" in data["systems"][s]

    def test_get_golden_queries_endpoint(self, client):
        """Verify golden queries listing endpoint."""
        response = client.get("/api/golden-queries")
        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 20
        assert len(data["categories"]) == 5

    def test_get_single_golden_query_endpoint(self, client):
        """Verify retrieving details of a single golden query (Q01)."""
        response = client.get("/api/golden-queries/Q01")
        assert response.status_code == 200
        data = response.json()
        assert data["query"]["query_id"] == "Q01"
        assert data["query"]["query_term"] == "كتب"
        assert len(data["candidates"]) > 0

    def test_get_single_golden_query_not_found(self, client):
        """Verify 404 is returned for non-existent query ID."""
        response = client.get("/api/golden-queries/Q999")
        assert response.status_code == 404

    def test_get_benchmarks_summary_endpoint(self, client):
        """Verify macro benchmarks summary endpoint."""
        response = client.get("/api/benchmarks/summary")
        assert response.status_code == 200
        data = response.json()
        assert "systems" in data
        assert len(data["systems"]) == 6
        s2 = next(s for s in data["systems"] if s["system_id"] == "S2")
        assert s2["p_at_10"] > 0.35

    def test_get_overstemming_endpoint(self, client):
        """Verify over-stemming analysis endpoint."""
        response = client.get("/api/benchmarks/overstemming")
        assert response.status_code == 200
        data = response.json()
        assert "overall_summary" in data or "metadata" in data
