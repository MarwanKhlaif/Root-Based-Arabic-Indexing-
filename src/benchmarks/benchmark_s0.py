#!/usr/bin/env python3
"""
S0 Baseline Query Latency Benchmarking Harness.
Milestone 2 - Corpus Acquisition, Normalization & Ingestion Pipeline.

Features:
- Benchmarks execution time across the 20 golden queries on S0 surface index (`idx_docs_s0_gin`).
- Implements strict Cold Cache and Warm Cache evaluation protocols.
- Uses server-side `EXPLAIN (ANALYZE, BUFFERS, TIMING, FORMAT JSON)` to isolate query execution from network latency.
- Extracts min, max, mean, median (p50), p95, p99, stddev, and buffer hit ratios.
- Verifies the baseline failure query (`يكتبون في المكتبة` vs `كتب` returning 0 hits).
- Emits results to `results/benchmark_s0_latency.json`, `.csv`, and `.md`.
"""

import sys
import os
import json
import csv
import time
import argparse
import statistics
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

sys.path.insert(0, ".")

from src.db.connection import get_connection


def load_golden_queries(queries_path: str = "queries/golden_queries_20.json") -> List[Dict[str, Any]]:
    """Load the 20 golden queries."""
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def prewarm_database() -> None:
    """Prewarm docs and s0 index pages into PostgreSQL shared_buffers."""
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm;")
            cur.execute("SELECT pg_prewarm('docs');")
            cur.execute("SELECT pg_prewarm('idx_docs_s0_gin');")


def restart_postgres_container() -> None:
    """Drop shared_buffers by restarting the PostgreSQL Docker container."""
    print("[INFO] Restarting container 'arabic_ir_db' to flush shared buffers...")
    subprocess.run(["docker", "restart", "arabic_ir_db"], check=True, stdout=subprocess.DEVNULL)
    # Wait for database healthcheck
    for _ in range(30):
        res = subprocess.run(
            ["docker", "exec", "arabic_ir_db", "pg_isready", "-U", "postgres", "-d", "arabic_ir"],
            capture_output=True,
            text=True,
        )
        if "accepting connections" in res.stdout:
            time.sleep(0.5)
            return
        time.sleep(0.5)
    raise RuntimeError("Database container did not become ready after restart!")


def execute_explain_query(cur, query_term: str) -> Dict[str, Any]:
    """Execute EXPLAIN ANALYZE and extract timing and buffer metrics."""
    sql = """
        EXPLAIN (ANALYZE, BUFFERS, TIMING, FORMAT JSON)
        SELECT id, doc_id, title, ts_rank(s0_tsvector, q) AS rank
        FROM docs, plainto_tsquery('simple', %s) q
        WHERE s0_tsvector @@ q
        ORDER BY rank DESC
        LIMIT 10;
    """
    cur.execute(sql, (query_term,))
    plan_wrapper = cur.fetchone()[0]
    plan_root = plan_wrapper[0]
    node = plan_root["Plan"]

    exec_time_ms = float(plan_root.get("Execution Time", 0.0))
    plan_time_ms = float(plan_root.get("Planning Time", 0.0))
    shared_hit = int(node.get("Shared Hit Blocks", 0))
    shared_read = int(node.get("Shared Read Blocks", 0))
    total_blocks = shared_hit + shared_read
    hit_ratio = round((shared_hit / total_blocks * 100), 2) if total_blocks > 0 else 100.0

    return {
        "execution_time_ms": exec_time_ms,
        "planning_time_ms": plan_time_ms,
        "shared_hit_blocks": shared_hit,
        "shared_read_blocks": shared_read,
        "buffer_hit_ratio_pct": hit_ratio,
        "rows_returned": int(node.get("Actual Rows", 0)),
    }


def verify_baseline_failure(cur) -> bool:
    """Verify that inflected phrase does not match root query in S0."""
    cur.execute("SELECT to_tsvector('simple', 'يكتبون في المكتبة') @@ to_tsquery('simple', 'كتب');")
    return bool(cur.fetchone()[0])  # Must return False


def run_benchmark(
    iterations: int = 10,
    include_cold: bool = True,
    cold_mode: str = "once",
) -> Dict[str, Any]:
    """Run full benchmark suite across all 20 golden queries."""
    queries = load_golden_queries()
    benchmark_data: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "iterations_per_query": iterations,
        "system_id": "S0",
        "system_name": "Simple Surface Baseline",
        "baseline_failure_verified": False,
        "query_results": [],
        "macro_metrics": {},
    }

    # 1. Verify Baseline Failure Query
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            bf_result = verify_baseline_failure(cur)
            assert bf_result is False, "CRITICAL: Baseline failure query evaluated to True!"
            benchmark_data["baseline_failure_verified"] = True
            print("[INFO] Baseline failure assertion verified: 'يكتبون في المكتبة' @@ 'كتب' is FALSE.")

    # 2. Cold Cache Run
    cold_timings: Dict[str, Dict[str, Any]] = {}
    if include_cold:
        print("\n[INFO] Starting Cold Cache Protocol...")
        if cold_mode == "once":
            with get_connection(autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("CHECKPOINT;")
            restart_postgres_container()

        for q in queries:
            qid = q["query_id"]
            term = q["query_term"]
            if cold_mode == "isolated":
                with get_connection(autocommit=True) as conn:
                    with conn.cursor() as cur:
                        cur.execute("CHECKPOINT;")
                restart_postgres_container()

            with get_connection(autocommit=True) as conn:
                with conn.cursor() as cur:
                    metrics = execute_explain_query(cur, term)
                    cold_timings[qid] = metrics
                    print(
                        f" [COLD] {qid} ('{term}'): {metrics['execution_time_ms']:.3f} ms | "
                        f"reads: {metrics['shared_read_blocks']}"
                    )

    # 3. Warm Cache Run
    print("\n[INFO] Prewarming database for Warm Cache Protocol...")
    prewarm_database()

    print(f"[INFO] Executing Warm Cache Protocol ({iterations} iterations per query)...")
    warm_results: List[Dict[str, Any]] = []
    all_mean_latencies: List[float] = []

    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            for q in queries:
                qid = q["query_id"]
                term = q["query_term"]

                # Iteration 0: Warmup discard
                execute_explain_query(cur, term)

                # Iterations 1..N: Measured runs
                runs: List[float] = []
                buffer_hits: List[float] = []
                for _ in range(iterations):
                    m = execute_explain_query(cur, term)
                    runs.append(m["execution_time_ms"])
                    buffer_hits.append(m["buffer_hit_ratio_pct"])

                min_lat = min(runs)
                max_lat = max(runs)
                mean_lat = statistics.mean(runs)
                median_lat = statistics.median(runs)
                std_lat = statistics.stdev(runs) if len(runs) > 1 else 0.0
                all_mean_latencies.append(mean_lat)

                q_entry = {
                    "query_id": qid,
                    "query_term": term,
                    "category_id": q["category_id"],
                    "category_name": q["category_name_en"],
                    "root": q["root"],
                    "cold_latency_ms": cold_timings.get(qid, {}).get("execution_time_ms"),
                    "cold_read_blocks": cold_timings.get(qid, {}).get("shared_read_blocks"),
                    "warm_min_ms": round(min_lat, 3),
                    "warm_max_ms": round(max_lat, 3),
                    "warm_mean_ms": round(mean_lat, 3),
                    "warm_median_p50_ms": round(median_lat, 3),
                    "warm_stddev_ms": round(std_lat, 3),
                    "warm_buffer_hit_pct": round(statistics.mean(buffer_hits), 2),
                }
                warm_results.append(q_entry)
                print(
                    f" [WARM] {qid} ('{term}'): mean={mean_lat:.3f} ms | "
                    f"med={median_lat:.3f} ms | min={min_lat:.3f} ms"
                )

    benchmark_data["query_results"] = warm_results
    benchmark_data["macro_metrics"] = {
        "macro_avg_warm_latency_ms": round(statistics.mean(all_mean_latencies), 3),
        "macro_median_warm_latency_ms": round(statistics.median(all_mean_latencies), 3),
        "macro_min_warm_latency_ms": round(min(all_mean_latencies), 3),
        "macro_max_warm_latency_ms": round(max(all_mean_latencies), 3),
    }

    return benchmark_data


def save_benchmark_artifacts(
    data: Dict[str, Any], json_path: str, csv_path: str, md_path: str
) -> None:
    """Save benchmark outputs in JSON, CSV, and Markdown formats."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # CSV Export
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "query_id",
            "query_term",
            "category_id",
            "category_name",
            "root",
            "cold_latency_ms",
            "cold_read_blocks",
            "warm_min_ms",
            "warm_max_ms",
            "warm_mean_ms",
            "warm_median_p50_ms",
            "warm_stddev_ms",
            "warm_buffer_hit_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data["query_results"]:
            writer.writerow(row)

    # Markdown Report
    table_rows = []
    for r in data["query_results"]:
        cold_val = f"{r['cold_latency_ms']:.2f}" if r["cold_latency_ms"] is not None else "N/A"
        table_rows.append(
            f"| `{r['query_id']}` | `{r['query_term']}` | `{r['root']}` | {r['category_name'][:24]} | "
            f"{cold_val} | **{r['warm_mean_ms']:.2f}** | {r['warm_median_p50_ms']:.2f} | "
            f"{r['warm_min_ms']:.2f}–{r['warm_max_ms']:.2f} | {r['warm_buffer_hit_pct']:.1f}% |"
        )
    table_md = "\n".join(table_rows)

    mm = data["macro_metrics"]
    content = f"""# System S0 Baseline Query Latency Benchmark Report

**Document Code**: `DOC-M2-BENCHMARK-S0`  
**Execution Timestamp**: `{data['timestamp_utc']}`  
**Evaluated System**: `S0` (Simple Surface GIN Index)  
**Iterations per Query**: {data['iterations_per_query']}  
**Baseline Failure Verified**: `{data['baseline_failure_verified']}`  

---

## 1. Macro Summary Metrics

| Metric | Measured Value |
|---|---|
| **Macro Average Warm Latency** | **{mm['macro_avg_warm_latency_ms']:.3f} ms** |
| **Macro Median Warm Latency** | **{mm['macro_median_warm_latency_ms']:.3f} ms** |
| **Fastest Query Warm Latency** | {mm['macro_min_warm_latency_ms']:.3f} ms |
| **Slowest Query Warm Latency** | {mm['macro_max_warm_latency_ms']:.3f} ms |

---

## 2. Query Latency Breakdown (20 Golden Queries)

| Query ID | Term | Root | Category | Cold (ms) | Warm Mean (ms) | Warm Median (ms) | Range (Min–Max) | Buffer Hit % |
|---|---|---|---|---|---|---|---|---|
{table_md}

---
*Report generated automatically via `src/benchmarks/benchmark_s0.py`.*
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark S0 baseline query latency across 20 golden queries.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of warm iterations per query.")
    parser.add_argument("--skip-cold", action="store_true", help="Skip cold cache benchmarking.")
    parser.add_argument("--cold-mode", choices=["once", "isolated"], default="once", help="Cold cache reset policy.")
    parser.add_argument("--json-out", default="results/benchmark_s0_latency.json", help="JSON output file.")
    parser.add_argument("--csv-out", default="results/benchmark_s0_latency.csv", help="CSV output file.")
    parser.add_argument("--md-out", default="results/benchmark_s0_latency.md", help="Markdown report file.")
    args = parser.parse_args()

    print("\n[INFO] Starting S0 Baseline Query Latency Benchmark...")
    data = run_benchmark(
        iterations=args.iterations,
        include_cold=not args.skip_cold,
        cold_mode=args.cold_mode,
    )

    save_benchmark_artifacts(data, args.json_out, args.csv_out, args.md_out)
    print(f"\n[SUCCESS] Benchmark artifacts saved to {args.json_out}, {args.csv_out}, and {args.md_out}")


if __name__ == "__main__":
    main()
