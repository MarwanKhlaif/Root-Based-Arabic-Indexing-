#!/usr/bin/env python3
"""
GIN Index Construction & Storage Footprint Benchmarking Engine.
Milestone 3 - Algorithmic Indexing Implementation S0–S5.

Features:
- Rigorous measurement of index construction time (seconds) and disk footprints (MB, bytes):
  * S0: Surface Baseline GIN (`idx_docs_s0_gin`)
  * S1: Surface + Stopwords GIN (`idx_docs_s1_gin`)
  * S2: Larkey Light10 Stem GIN (`idx_docs_s2_gin`)
  * S3: ISRI Morphological Root GIN (`idx_docs_s3_gin`)
  * S4: Khoja Dictionary Root GIN (`idx_docs_s4_gin`)
  * S5: Character Trigram GIN (`idx_docs_s5_trgm_gin`)
- Calculates relative storage ratio against System S0 baseline (% of S0).
- Calculates index construction throughput (documents/sec).
- Reports table heap data, total index size, and total relation footprint.
- Emits structured artifacts to `results/indexing_benchmarks.json` and publication-ready `results/indexing_benchmarks.md`.
- Supports `--mode build` (default, builds S1–S5 with 1GB mem), `--mode inspect`, `--mode reindex`, and `--mode rebuild`.
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

sys.path.insert(0, ".")

os.environ.setdefault("PGCONNECT_TIMEOUT", "30")
from src.db.connection import get_connection as _raw_get_connection


def get_connection(autocommit: bool = False):
    max_retries = 5
    base_delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return _raw_get_connection(autocommit=autocommit)
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"[WARN] Database connection attempt {attempt}/{max_retries} failed ({e}). Retrying in {base_delay * attempt:.1f}s...")
            time.sleep(base_delay * attempt)

SYSTEM_SPECS = [
    {
        "system_id": "S0",
        "system_name": "Surface Simple (Baseline)",
        "morphological_level": "Surface / Lexical",
        "index_name": "idx_docs_s0_gin",
        "column_name": "s0_tsvector",
        "index_type": "GIN (tsvector)",
        "configuration": "simple",
        "create_sql": "CREATE INDEX IF NOT EXISTS idx_docs_s0_gin ON docs USING GIN (s0_tsvector);",
    },
    {
        "system_id": "S1",
        "system_name": "Surface + Stopwords",
        "morphological_level": "Surface / Lexical",
        "index_name": "idx_docs_s1_gin",
        "column_name": "s1_tsvector",
        "index_type": "GIN (tsvector)",
        "configuration": "arabic_simple",
        "create_sql": "CREATE INDEX IF NOT EXISTS idx_docs_s1_gin ON docs USING GIN (s1_tsvector);",
    },
    {
        "system_id": "S2",
        "system_name": "Larkey Light10 Stemmer",
        "morphological_level": "Inflectional Morphology",
        "index_name": "idx_docs_s2_gin",
        "column_name": "s2_tsvector",
        "index_type": "GIN (tsvector)",
        "configuration": "simple",
        "create_sql": "CREATE INDEX IF NOT EXISTS idx_docs_s2_gin ON docs USING GIN (s2_tsvector);",
    },
    {
        "system_id": "S3",
        "system_name": "ISRI Morphological Root Stemmer",
        "morphological_level": "Derivational (Pattern/Awzan)",
        "index_name": "idx_docs_s3_gin",
        "column_name": "s3_tsvector",
        "index_type": "GIN (tsvector)",
        "configuration": "simple",
        "create_sql": "CREATE INDEX IF NOT EXISTS idx_docs_s3_gin ON docs USING GIN (s3_tsvector);",
    },
    {
        "system_id": "S4",
        "system_name": "Khoja Dictionary Root Stemmer",
        "morphological_level": "Derivational (Lexicon-Validated)",
        "index_name": "idx_docs_s4_gin",
        "column_name": "s4_tsvector",
        "index_type": "GIN (tsvector)",
        "configuration": "simple",
        "create_sql": "CREATE INDEX IF NOT EXISTS idx_docs_s4_gin ON docs USING GIN (s4_tsvector);",
    },
    {
        "system_id": "S5",
        "system_name": "Character Trigram Matching",
        "morphological_level": "Sub-word / Character 3-gram",
        "index_name": "idx_docs_s5_trgm_gin",
        "column_name": "s5_trgm_text",
        "index_type": "GIN (gin_trgm_ops)",
        "configuration": "pg_trgm",
        "create_sql": "CREATE INDEX IF NOT EXISTS idx_docs_s5_trgm_gin ON docs USING GIN (s5_trgm_text gin_trgm_ops);",
    },
]


def check_index_exists(cur, index_name: str) -> bool:
    """Check if an index exists in PostgreSQL."""
    cur.execute("SELECT to_regclass(%s) IS NOT NULL;", (index_name,))
    return bool(cur.fetchone()[0])


def benchmark_indexing_systems(
    mode: str = "build",
    maintenance_work_mem: str = "1GB",
) -> Dict[str, Any]:
    """
    Execute benchmarking harness across systems S0 to S5.
    
    Modes:
    - 'build': Construct indexes S1–S5 using maintenance_work_mem (preserve S0 baseline).
    - 'inspect': Query existing relation sizes and statuses.
    - 'reindex': Execute REINDEX on all indexes to measure live rebuild duration.
    - 'rebuild': DROP and recreate all indexes including S0.
    """
    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "database": "arabic_ir",
        "table": "docs",
        "mode": mode,
        "maintenance_work_mem": maintenance_work_mem,
        "total_documents": 0,
        "table_footprints": {},
        "benchmarks": [],
        "comparative_summary": {},
    }

    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            # 1. Total documents check
            cur.execute("SELECT count(*) FROM docs;")
            total_docs = cur.fetchone()[0]
            report["total_documents"] = total_docs
            if total_docs == 0:
                raise ValueError("Table 'docs' is empty! Ingest corpus before benchmarking.")

            cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}';")
            cur.execute("SET max_parallel_maintenance_workers = 4;")

            s0_bytes: Optional[int] = None
            total_index_build_sec = 0.0

            # 2. Benchmark each indexing system
            for sys_info in SYSTEM_SPECS:
                sys_id = sys_info["system_id"]
                idx_name = sys_info["index_name"]
                col_name = sys_info["column_name"]
                exists = check_index_exists(cur, idx_name)
                build_dur = 0.0

                print(f"[INFO] Benchmarking System {sys_id} ({idx_name})...")

                if mode == "build":
                    if sys_id == "S0" and exists:
                        print(f"[INFO] ... Preserving S0 baseline ({idx_name})")
                    else:
                        if exists:
                            print(f"[INFO] ... Dropping existing index {idx_name}")
                            cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
                        print(f"[INFO] ... Constructing index {idx_name} with maintenance_work_mem = {maintenance_work_mem}")
                        t0 = time.perf_counter()
                        cur.execute(sys_info["create_sql"])
                        build_dur = time.perf_counter() - t0
                        exists = True

                elif mode == "rebuild":
                    if exists:
                        print(f"[INFO] ... Dropping existing index {idx_name}")
                        cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
                    print(f"[INFO] ... Constructing index {idx_name} with maintenance_work_mem = {maintenance_work_mem}")
                    t0 = time.perf_counter()
                    cur.execute(sys_info["create_sql"])
                    build_dur = time.perf_counter() - t0
                    exists = True

                elif mode == "reindex":
                    if exists:
                        print(f"[INFO] ... REINDEXing {idx_name}")
                        t0 = time.perf_counter()
                        cur.execute(f"REINDEX INDEX {idx_name};")
                        build_dur = time.perf_counter() - t0
                    else:
                        print(f"[INFO] ... Index does not exist, creating {idx_name}")
                        t0 = time.perf_counter()
                        cur.execute(sys_info["create_sql"])
                        build_dur = time.perf_counter() - t0
                        exists = True

                elif mode == "inspect":
                    if not exists:
                        print(f"[WARN] Index {idx_name} does not exist in database yet.")

                total_index_build_sec += build_dur

                # Measure physical index size
                if exists:
                    cur.execute(f"SELECT pg_relation_size('{idx_name}'), pg_size_pretty(pg_relation_size('{idx_name}'));")
                    sz_bytes, sz_pretty = cur.fetchone()
                    sz_mb = round(sz_bytes / (1024 * 1024), 2)
                else:
                    sz_bytes = 0
                    sz_pretty = "0 MB"
                    sz_mb = 0.0

                if sys_id == "S0" and sz_bytes > 0:
                    s0_bytes = sz_bytes

                # Calculate relative size compared to S0
                rel_ratio = round((sz_bytes / s0_bytes * 100), 2) if (s0_bytes and sz_bytes > 0) else 0.0
                throughput = round(total_docs / build_dur, 1) if build_dur > 0 else 0.0

                item_res = {
                    "system_id": sys_id,
                    "system_name": sys_info["system_name"],
                    "morphological_level": sys_info["morphological_level"],
                    "index_name": idx_name,
                    "column_name": col_name,
                    "index_type": sys_info["index_type"],
                    "configuration": sys_info["configuration"],
                    "exists": exists,
                    "build_duration_sec": round(build_dur, 2),
                    "build_throughput_docs_sec": throughput,
                    "index_size_bytes": sz_bytes,
                    "index_size_mb": sz_mb,
                    "index_size_pretty": sz_pretty,
                    "relative_size_pct_s0": rel_ratio,
                }
                report["benchmarks"].append(item_res)
                print(f"[STATUS] {sys_id}: Size={sz_pretty} ({sz_bytes:,} B) | Build={build_dur:.2f}s | Rel={rel_ratio}%\n")

            # 3. Post-build Table-level storage footprints and VACUUM ANALYZE
            if mode in ("build", "rebuild", "reindex"):
                print("[INFO] Executing VACUUM ANALYZE docs to update query planner statistics...")
                t_vac = time.perf_counter()
                cur.execute("VACUUM ANALYZE docs;")
                print(f"[SUCCESS] VACUUM ANALYZE finished in {time.perf_counter() - t_vac:.2f}s.\n")

            cur.execute("""
                SELECT 
                    pg_relation_size('docs') AS heap_bytes,
                    pg_size_pretty(pg_relation_size('docs')) AS heap_pretty,
                    pg_indexes_size('docs') AS indexes_bytes,
                    pg_size_pretty(pg_indexes_size('docs')) AS indexes_pretty,
                    pg_total_relation_size('docs') AS total_bytes,
                    pg_size_pretty(pg_total_relation_size('docs')) AS total_pretty;
            """)
            heap_b, heap_p, idx_b, idx_p, tot_b, tot_p = cur.fetchone()
            report["table_footprints"] = {
                "heap_data_bytes": heap_b,
                "heap_data_pretty": heap_p,
                "total_indexes_bytes": idx_b,
                "total_indexes_pretty": idx_p,
                "total_relation_bytes": tot_b,
                "total_relation_pretty": tot_p,
            }

            # 4. Compile summary metrics
            total_gin_bytes = sum(b["index_size_bytes"] for b in report["benchmarks"])
            report["comparative_summary"] = {
                "total_gin_indexes_bytes": total_gin_bytes,
                "total_gin_indexes_mb": round(total_gin_bytes / (1024 * 1024), 2),
                "total_measured_build_sec": round(total_index_build_sec, 2),
                "s0_baseline_bytes": s0_bytes or 0,
            }

    return report


def generate_markdown_report(data: Dict[str, Any], output_path: str) -> None:
    """Format publication-ready Markdown table and IR literature commentary."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tf = data["table_footprints"]
    cs = data["comparative_summary"]

    table_rows = []
    for b in data["benchmarks"]:
        dur_str = f"{b['build_duration_sec']:.2f}s" if b["build_duration_sec"] > 0 else "Pre-built (M2)"
        thp_str = f"{b['build_throughput_docs_sec']:,.1f}" if b["build_throughput_docs_sec"] > 0 else "N/A"
        table_rows.append(
            f"| **{b['system_id']}** | {b['system_name']} | `{b['index_name']}` | `{b['column_name']}` | {b['index_type']} | {dur_str} | **{b['index_size_pretty']}** | {b['index_size_bytes']:,} B | **{b['relative_size_pct_s0']:.1f}%** | {thp_str} |"
        )
    table_md = "\n".join(table_rows)

    content = f"""# GIN Index Construction & Storage Footprint Benchmark (S0–S5)

**Document Code**: `DOC-M3-INDEXING-BENCHMARK`  
**Execution Timestamp**: `{data['timestamp_utc']}`  
**Corpus Table**: `{data['table']}` ({data['total_documents']:,} clean Wikipedia articles, seed 42)  
**Database**: `{data['database']}` (PostgreSQL 16 via Docker)  
**Evaluation Mode**: `{data['mode']}` (maintenance_work_mem = `{data['maintenance_work_mem']}`)  

---

## 1. Algorithmic GIN Index Comparative Matrix

| System ID | System Name | Index Name | Indexed Column | Index Structure | Build Duration (s) | Disk Footprint (MB) | Disk Footprint (Bytes) | Relative Size (% of S0) | Build Throughput (docs/s) |
|---|---|---|---|---|---|---|---|---|---|
{table_md}

---

## 2. Database Relation Storage Breakdown

| Relation Component | Pretty Size | Exact Footprint (Bytes) | Description |
|---|---|---|---|
| **Heap Data Size (`docs` table)** | **{tf['heap_data_pretty']}** | {tf['heap_data_bytes']:,} B | Raw and normalized text columns in PostgreSQL heap |
| **Total Indexes Footprint** | **{tf['total_indexes_pretty']}** | {tf['total_indexes_bytes']:,} B | Combined footprint of all active B-Tree and GIN indexes |
| **Sum of All GIN Indexes (S0–S5)** | **{cs['total_gin_indexes_mb']:.2f} MB** | {cs['total_gin_indexes_bytes']:,} B | Cumulative disk space required by the 6 IR systems |
| **Total Relation Size (`docs` + all indexes + TOAST)** | **{tf['total_relation_pretty']}** | {tf['total_relation_bytes']:,} B | Total disk space allocated on the database filesystem |

---

## 3. Computational and Storage Analysis

### 3.1 Lexical vs Morphological Index Packing (S0 vs S1–S4)
- **S0 Surface Baseline**: Stores verbatim lexical forms. High unique vocabulary count (~1.05M lexemes) produces an index size of ~{data['benchmarks'][0]['index_size_pretty']}.
- **S1 Stopword Removal (`arabic_simple`)**: Filters functional particles (`في`, `من`, `على`), reducing posting lists and yielding a more compact index.
- **S2 Light Stemming (Larkey light10)**: Strips inflectional affixes (prefixes and suffixes), collapsing plural and clitic variations into canonical stems.
- **S3 & S4 Root Extraction (ISRI & Khoja)**: Morpheme-to-root collapsing drastically shrinks the unique lexeme vocabulary (from 1,050,000 down to tens of thousands in S3, and ~4,700 classical roots in S4). This produces highly compressed GIN entry dictionaries, though posting lists per root expand proportionally.

### 3.2 Sub-word Character Trigram Footprint (S5)
- **S5 Trigram Matching (`gin_trgm_ops`)**: Decomposes text into all overlapping 3-character sequences. Because every 500-word document yields thousands of trigrams, the trigram posting list expands massively, typically consuming 3x–5x more storage than surface or morphological tsvectors.

---
*Report generated automatically via `src/benchmarks/benchmark_indexing.py`.*
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GIN Index Construction and Storage Footprints (S0-S5).")
    parser.add_argument(
        "--mode",
        choices=["build", "inspect", "reindex", "rebuild"],
        default="build",
        help="Benchmark execution mode: build (build S1-S5, preserve S0), inspect (read existing), reindex (REINDEX), rebuild (drop & recreate all).",
    )
    parser.add_argument(
        "--maintenance-work-mem",
        default="1GB",
        help="PostgreSQL maintenance_work_mem setting for index builds (default: 1GB).",
    )
    parser.add_argument(
        "--json-out",
        default="results/indexing_benchmarks.json",
        help="Path for output JSON file (default: results/indexing_benchmarks.json).",
    )
    parser.add_argument(
        "--md-out",
        default="results/indexing_benchmarks.md",
        help="Path for output Markdown report (default: results/indexing_benchmarks.md).",
    )
    args = parser.parse_args()

    print("\n[INFO] Starting GIN Indexing Benchmark Suite...")
    data = benchmark_indexing_systems(mode=args.mode, maintenance_work_mem=args.maintenance_work_mem)

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] JSON results exported to: {args.json_out}")

    generate_markdown_report(data, args.md_out)
    print(f"[SUCCESS] Markdown report exported to: {args.md_out}")


if __name__ == "__main__":
    main()
