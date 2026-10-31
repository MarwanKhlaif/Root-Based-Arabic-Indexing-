#!/usr/bin/env python3
"""
Corpus Statistics & Vocabulary Profiling Engine via PostgreSQL ts_stat.
Milestone 2 - Corpus Acquisition, Normalization & Ingestion Pipeline.

Features:
- Materializes `ts_stat` into a temporary table to scan the corpus tsvector exactly once.
- Calculates total documents, vocabulary size (unique lexemes), average lexeme length.
- Calculates document length distribution (min, max, avg, stddev, percentiles p50..p99, IQR).
- Extracts Top 50 most frequent terms with collection and document frequencies.
- Calculates Hapax Legomena (terms occurring exactly once in the entire collection).
- Reports physical storage footprints (heap data, GIN index, total relation size).
- Exports structured artifacts to `results/corpus_statistics.json` and `results/corpus_statistics.md`.
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List

sys.path.insert(0, ".")

from src.db.connection import get_connection


def compute_corpus_statistics() -> Dict[str, Any]:
    """Execute formal statistical queries and compile comprehensive metrics."""
    data: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "database": "arabic_ir",
        "table": "docs",
        "doc_length_metrics": {},
        "vocabulary_metrics": {},
        "top_50_terms": [],
        "storage_footprints": {},
    }

    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            # 1. Document Count & Length Distribution
            cur.execute("""
                SELECT 
                    count(*) AS total_docs,
                    coalesce(sum(word_count), 0) AS total_words,
                    coalesce(sum(char_count), 0) AS total_chars,
                    coalesce(min(word_count), 0) AS min_words,
                    coalesce(max(word_count), 0) AS max_words,
                    round(coalesce(avg(word_count), 0), 2) AS avg_words_per_doc,
                    round(coalesce(stddev(word_count), 0), 2) AS stddev_words,
                    coalesce(percentile_cont(0.25) WITHIN GROUP (ORDER BY word_count), 0) AS p25_words,
                    coalesce(percentile_cont(0.50) WITHIN GROUP (ORDER BY word_count), 0) AS p50_words,
                    coalesce(percentile_cont(0.75) WITHIN GROUP (ORDER BY word_count), 0) AS p75_words,
                    coalesce(percentile_cont(0.90) WITHIN GROUP (ORDER BY word_count), 0) AS p90_words,
                    coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY word_count), 0) AS p95_words,
                    coalesce(percentile_cont(0.99) WITHIN GROUP (ORDER BY word_count), 0) AS p99_words,
                    round(coalesce(avg(char_count::numeric / nullif(word_count, 0)), 0), 2) AS avg_chars_per_word
                FROM docs;
            """)
            doc_row = cur.fetchone()
            total_docs = int(doc_row[0])
            if total_docs == 0:
                raise ValueError("Table 'docs' is empty! Ingest documents before running corpus statistics.")

            p25, p75 = float(doc_row[7]), float(doc_row[9])
            data["doc_length_metrics"] = {
                "total_documents": total_docs,
                "total_words": int(doc_row[1]),
                "total_characters": int(doc_row[2]),
                "min_words": int(doc_row[3]),
                "max_words": int(doc_row[4]),
                "avg_words_per_doc": float(doc_row[5]),
                "stddev_words": float(doc_row[6]),
                "p25_words": p25,
                "p50_median_words": float(doc_row[8]),
                "p75_words": p75,
                "iqr_words": round(p75 - p25, 2),
                "p90_words": float(doc_row[10]),
                "p95_words": float(doc_row[11]),
                "p99_words": float(doc_row[12]),
                "avg_chars_per_word": float(doc_row[13]),
            }

            # 2. Materialized ts_stat Execution (Supervisor Directive: Execute Once)
            print("[INFO] Materializing ts_stat into session temporary table 's0_vocab_stats'...")
            cur.execute("""
                CREATE TEMP TABLE s0_vocab_stats AS 
                SELECT word, ndoc, nentry 
                FROM ts_stat('SELECT s0_tsvector FROM docs');
            """)

            # 3. Vocabulary Size & Average Lexeme Length
            cur.execute("""
                SELECT 
                    count(*) AS total_unique_lexemes,
                    coalesce(sum(nentry), 0) AS total_indexed_tokens,
                    round(coalesce(avg(length(word)), 0), 2) AS avg_lexeme_length
                FROM s0_vocab_stats;
            """)
            voc_row = cur.fetchone()
            vocab_size = int(voc_row[0])
            total_tokens = int(voc_row[1])
            avg_lexeme_len = float(voc_row[2])

            # 4. Hapax Legomena (Terms appearing exactly once in the corpus)
            cur.execute("SELECT count(*) FROM s0_vocab_stats WHERE nentry = 1;")
            hapax_count = int(cur.fetchone()[0])
            hapax_pct = round((hapax_count / vocab_size * 100), 2) if vocab_size > 0 else 0.0

            data["vocabulary_metrics"] = {
                "total_unique_lexemes": vocab_size,
                "total_indexed_tokens": total_tokens,
                "avg_lexeme_length_chars": avg_lexeme_len,
                "hapax_legomena_count": hapax_count,
                "hapax_legomena_percentage": hapax_pct,
            }

            # 5. Top 50 Most Frequent Lexemes
            cur.execute("""
                SELECT 
                    word, 
                    nentry AS collection_frequency, 
                    ndoc AS document_frequency, 
                    round((ndoc::numeric / %s * 100), 2) AS doc_freq_pct
                FROM s0_vocab_stats 
                ORDER BY nentry DESC 
                LIMIT 50;
            """, (total_docs,))
            for rank, r in enumerate(cur.fetchall(), 1):
                data["top_50_terms"].append({
                    "rank": rank,
                    "term": r[0],
                    "collection_frequency": int(r[1]),
                    "document_frequency": int(r[2]),
                    "doc_frequency_percentage": float(r[3]),
                })

            # 6. Physical Storage Footprints
            cur.execute("""
                SELECT 
                    pg_total_relation_size('docs') AS total_table_bytes,
                    pg_size_pretty(pg_total_relation_size('docs')) AS total_table_pretty,
                    pg_relation_size('docs') AS heap_bytes,
                    pg_size_pretty(pg_relation_size('docs')) AS heap_pretty,
                    pg_indexes_size('docs') AS indexes_bytes,
                    pg_size_pretty(pg_indexes_size('docs')) AS indexes_pretty,
                    coalesce(pg_relation_size('idx_docs_s0_gin'), 0) AS s0_gin_bytes,
                    pg_size_pretty(coalesce(pg_relation_size('idx_docs_s0_gin'), 0)) AS s0_gin_pretty,
                    coalesce(pg_relation_size('uq_docs_doc_id'), 0) AS uq_doc_id_bytes,
                    pg_size_pretty(coalesce(pg_relation_size('uq_docs_doc_id'), 0)) AS uq_doc_id_pretty;
            """)
            stor_row = cur.fetchone()
            data["storage_footprints"] = {
                "total_relation_bytes": int(stor_row[0]),
                "total_relation_pretty": stor_row[1],
                "heap_data_bytes": int(stor_row[2]),
                "heap_data_pretty": stor_row[3],
                "total_indexes_bytes": int(stor_row[4]),
                "total_indexes_pretty": stor_row[5],
                "indexes_bytes": int(stor_row[4]),
                "indexes_pretty": stor_row[5],
                "s0_gin_index_bytes": int(stor_row[6]),
                "s0_gin_index_pretty": stor_row[7],
                "doc_id_btree_bytes": int(stor_row[8]),
                "doc_id_btree_pretty": stor_row[9],
                "uq_doc_id_bytes": int(stor_row[8]),
                "uq_doc_id_pretty": stor_row[9],
            }

    return data


def generate_markdown_report(data: Dict[str, Any], output_path: str) -> None:
    """Generate academic markdown report documenting all statistics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dlen = data["doc_length_metrics"]
    voc = data["vocabulary_metrics"]
    stor = data["storage_footprints"]

    top50_rows = []
    for t in data["top_50_terms"]:
        top50_rows.append(
            f"| {t['rank']} | `{t['term']}` | {t['collection_frequency']:,} | {t['document_frequency']:,} | {t['doc_frequency_percentage']:.2f}% |"
        )
    top50_md = "\n".join(top50_rows)

    content = f"""# Corpus Statistics & Lexicon Profiling Report (PostgreSQL 16)

**Document Code**: `DOC-M2-CORPUS-STATS`  
**Execution Timestamp**: `{data['timestamp_utc']}`  
**Corpus Table**: `{data['table']}`  
**Database**: `{data['database']}`  

---

## 1. Document & Length Metrics

| Metric | Value |
|---|---|
| **Total Ingested Documents** | **{dlen['total_documents']:,}** |
| **Total Words in Corpus** | {dlen['total_words']:,} |
| **Total Characters in Corpus** | {dlen['total_characters']:,} |
| **Average Document Length** | **{dlen['avg_words_per_doc']:.2f} words** |
| **Document Length StdDev** | {dlen['stddev_words']:.2f} |
| **Min / Max Document Length** | {dlen['min_words']} / {dlen['max_words']:,} words |
| **Median ($p_{{50}}$) Length** | **{dlen['p50_median_words']:.1f} words** |
| **25th / 75th Percentile ($p_{{25}} / p_{{75}}$)** | {dlen['p25_words']:.1f} / {dlen['p75_words']:.1f} words (IQR = {dlen['iqr_words']:.1f}) |
| **90th / 95th / 99th Percentile** | {dlen['p90_words']:.1f} / {dlen['p95_words']:.1f} / {dlen['p99_words']:.1f} words |
| **Average Characters per Word** | {dlen['avg_chars_per_word']:.2f} characters |

---

## 2. Vocabulary & Lexicon Metrics (System S0 Surface Baseline)

| Metric | Value |
|---|---|
| **Total Unique Lexemes (Vocabulary Size)** | **{voc['total_unique_lexemes']:,}** |
| **Total Indexed Tokens** | {voc['total_indexed_tokens']:,} |
| **Average Lexeme Length** | **{voc['avg_lexeme_length_chars']:.2f} characters** |
| **Hapax Legomena (Terms with CF = 1)** | **{voc['hapax_legomena_count']:,}** ({voc['hapax_legomena_percentage']:.2f}% of vocabulary) |

---

## 3. Database Physical Storage Footprints

| Relation / Index Component | Pretty Size | Exact Footprint (Bytes) |
|---|---|---|
| **Total Relation Size (`docs` + all indexes)** | **{stor['total_relation_pretty']}** | {stor['total_relation_bytes']:,} B |
| **Heap Data Size (`docs` table)** | {stor['heap_data_pretty']} | {stor['heap_data_bytes']:,} B |
| **Total Indexes Footprint** | {stor['total_indexes_pretty']} | {stor['total_indexes_bytes']:,} B |
| **S0 Surface GIN Index (`idx_docs_s0_gin`)** | **{stor['s0_gin_index_pretty']}** | {stor['s0_gin_index_bytes']:,} B |
| **B-Tree Primary Key / Unique `doc_id`** | {stor['uq_doc_id_pretty']} | {stor['doc_id_btree_bytes']:,} B |

---

## 4. Top 50 Most Frequent Terms (Zipfian Distribution)

| Rank | Lexeme (`word`) | Collection Frequency (CF / `nentry`) | Document Frequency (DF / `ndoc`) | DF % |
|---|---|---|---|---|
{top50_md}

---
*Report generated automatically via `src/analytics/corpus_stats.py`.*
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate formal corpus statistics via ts_stat.")
    parser.add_argument("--json-out", default="results/corpus_statistics.json", help="Path for JSON output.")
    parser.add_argument("--md-out", default="results/corpus_statistics.md", help="Path for Markdown report.")
    args = parser.parse_args()

    print("\n[INFO] Connecting to database 'arabic_ir' and computing corpus statistics...")
    data = compute_corpus_statistics()

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] JSON statistics saved to: {args.json_out}")

    generate_markdown_report(data, args.md_out)
    print(f"[SUCCESS] Markdown report saved to: {args.md_out}")

    d = data["doc_length_metrics"]
    v = data["vocabulary_metrics"]
    s = data["storage_footprints"]
    print("\n" + "=" * 60)
    print(f" Total Documents:      {d['total_documents']:,}")
    print(f" Vocabulary Size:       {v['total_unique_lexemes']:,} unique lexemes")
    print(f" Avg Doc Length:        {d['avg_words_per_doc']:.1f} words")
    print(f" Hapax Legomena:        {v['hapax_legomena_count']:,} ({v['hapax_legomena_percentage']}%)")
    print(f" Heap Data Size:        {s['heap_data_pretty']}")
    print(f" S0 GIN Index Size:     {s['s0_gin_index_pretty']}")
    print(f" Total Relation Size:   {s['total_relation_pretty']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
