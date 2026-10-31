"""
FastAPI Web Application for Arabic Full-Text Indexing & Evaluation in PostgreSQL 16.
Serves interactive RTL user interface, REST endpoints for multi-system search,
real-time morphological query transformations, and benchmark visualizations.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.web.search_service import (
    transform_query,
    check_db_health,
    search_all_systems,
    load_golden_queries,
    load_qrels_graded,
    load_benchmark_metrics,
    load_indexing_benchmarks,
    load_overstemming_data,
    SYSTEM_METADATA,
)

app = FastAPI(
    title="Arabic IR Indexing Evaluation (PostgreSQL 16)",
    description="Interactive Web Interface comparing S0 to S5 full-text indexing configurations.",
    version="1.0.0",
)

# Static and template directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic models for request bodies
class TransformRequest(BaseModel):
    query: str = Field(..., description="Arabic search query to transform")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    systems: Optional[List[str]] = Field(
        default=["S0", "S1", "S2", "S3", "S4", "S5"],
        description="List of systems to execute (S0, S1, S2, S3, S4, S5)",
    )
    limit: Optional[int] = Field(default=10, ge=1, le=100, description="Max results per system")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the primary Single-Page Application interface."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Index template not found")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_system_status():
    """Retrieve PostgreSQL connection health and corpus statistics."""
    health = check_db_health()
    return JSONResponse(content=health)


@app.post("/api/transform")
async def transform_arabic_query(req: TransformRequest):
    """
    Perform instant morphological transformation across S0–S5.
    Returns normalized text, affixes, roots, and trigrams.
    """
    if not req.query or not req.query.strip():
        return JSONResponse(content={"raw_query": "", "systems": {}})
    result = transform_query(req.query)
    return JSONResponse(content=result)


@app.post("/api/search")
async def execute_multi_search(req: SearchRequest):
    """
    Execute search across selected indexing configurations.
    Returns ranked documents, highlighted snippets, scores, and execution latency.
    """
    cleaned_query = req.query.strip() if req.query else ""
    if not cleaned_query:
        return JSONResponse(content={
            "query": "",
            "systems": {},
            "message": "يرجى كتابة كلمة أو عبارة للبحث"
        })

    results = search_all_systems(
        raw_query=cleaned_query,
        selected_systems=req.systems,
        limit=req.limit or 10,
    )
    return JSONResponse(content=results)


@app.get("/api/golden-queries")
async def get_all_golden_queries():
    """Retrieve list of the 20 golden linguistic queries with categories and roots."""
    queries = load_golden_queries()
    # Group by category
    categories_map: dict = {}
    for q in queries:
        cat_id = q.get("category_id", 0)
        cat_name = q.get("category_name_ar", "أخرى")
        if cat_id not in categories_map:
            categories_map[cat_id] = {
                "category_id": cat_id,
                "category_name_ar": cat_name,
                "category_name_en": q.get("category_name_en", ""),
                "queries": [],
            }
        categories_map[cat_id]["queries"].append(q)

    return JSONResponse(content={
        "total_queries": len(queries),
        "queries": queries,
        "categories": list(categories_map.values()),
    })


@app.get("/api/golden-queries/{query_id}")
async def get_single_golden_query(query_id: str):
    """Retrieve in-depth linguistic details and benchmark metrics for a single query."""
    queries = load_golden_queries()
    qrels = load_qrels_graded()
    metrics = load_benchmark_metrics()

    target_query = next((q for q in queries if q.get("query_id", "").upper() == query_id.upper()), None)
    if not target_query:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")

    # Find candidates
    candidates = []
    if qrels and "queries" in qrels and query_id in qrels["queries"]:
        candidates = qrels["queries"][query_id].get("candidates", [])

    # Find per-system benchmark results
    per_system_metrics = {}
    for item in metrics.get("per_query_results", []):
        if item.get("query_id") == query_id:
            sys_id = item.get("system_id")
            per_system_metrics[sys_id] = item

    return JSONResponse(content={
        "query": target_query,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "metrics": per_system_metrics,
    })


@app.get("/api/benchmarks/summary")
async def get_benchmarks_summary():
    """Retrieve macro IR metrics summary and comparison tables across S0–S5."""
    ir_data = load_benchmark_metrics()
    indexing_data = load_indexing_benchmarks()

    macro_summary = ir_data.get("systems_macro_summary", {})
    indexes = {item.get("system_id"): item for item in indexing_data.get("benchmarks", [])}

    combined = []
    for sys_id, meta in SYSTEM_METADATA.items():
        m = macro_summary.get(sys_id, {})
        idx = indexes.get(sys_id, {})

        combined.append({
            "system_id": sys_id,
            "system_name": meta["name"],
            "morphological_level": meta["level"],
            "badge_color": meta["badge_color"],
            "color_hex": meta["color_hex"],
            "p_at_10": m.get("macro_p_at_10", meta["p_at_10"]),
            "map": m.get("macro_map", 0.0),
            "ndcg_at_10": m.get("macro_ndcg_at_10", 0.0),
            "recall_at_100": m.get("macro_recall_at_100", meta["recall_at_100"]),
            "cold_latency_ms": m.get("macro_cold_latency_ms", 0.0),
            "warm_latency_ms": m.get("macro_warm_latency_ms", 0.0),
            "index_size_mb": idx.get("index_size_mb", meta["index_size_mb"]),
            "index_size_pretty": idx.get("index_size_pretty", meta["index_size_pretty"]),
            "relative_size_pct": idx.get("relative_size_pct_s0", 100.0),
            "build_duration_sec": idx.get("build_duration_sec", 0.0),
            "throughput_docs_sec": idx.get("build_throughput_docs_sec", 0.0),
        })

    return JSONResponse(content={
        "timestamp": ir_data.get("metadata", {}).get("timestamp_utc"),
        "corpus": ir_data.get("metadata", {}).get("corpus", "Arabic Wikipedia 100,000 Articles"),
        "total_queries": ir_data.get("metadata", {}).get("total_queries", 20),
        "systems": combined,
    })


@app.get("/api/benchmarks/overstemming")
async def get_overstemming_summary():
    """Retrieve over-stemming and semantic collision findings."""
    data = load_overstemming_data()
    return JSONResponse(content=data)
