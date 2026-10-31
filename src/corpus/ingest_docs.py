#!/usr/bin/env python3
"""
High-Performance Streaming Ingestion Pipeline for Arabic IR Corpus.
Milestone 2 - Corpus Acquisition, Normalization & Ingestion Pipeline.

Features:
- High-throughput streaming COPY into PostgreSQL (>25,000 rows/s) using psycopg 3.
- Pre-sanitization stripping null bytes (\\x00) and handling delimiter safety.
- Two-stage deferred indexing: bulk COPY text -> populate s0_tsvector -> build GIN index -> VACUUM ANALYZE.
- Supports streaming from JSONL, Python generators, or synthetic samples.
"""

import sys
import os
import time
import json
import argparse
from typing import Iterable, Dict, Any, Optional, Generator
from datetime import datetime, timezone

sys.path.insert(0, ".")

from src.db.connection import get_connection
from src.preprocessing.normalize import normalize_arabic


def sanitize_text(value: Optional[str]) -> str:
    """Strip null bytes and control terminators that crash PostgreSQL UTF-8 COPY."""
    if not value:
        return ""
    return str(value).replace("\x00", "").strip()


def generate_synthetic_articles(n_samples: int = 1000) -> Generator[Dict[str, Any], None, None]:
    """Generate clean synthetic Arabic articles for immediate testing and verification."""
    roots = ["كتب", "درس", "علم", "حكم", "سفر", "نصر", "شجر", "بحر", "طرق", "صحب"]
    sample_phrases = [
        "كتب الطالب في المكتبة كتابا مفيدا حول تاريخ العلوم والحضارة الإسلامية ودرس فيه كثيرا.",
        "تعتبر المدرسة والجامعة من أهم مؤسسات التعليم التي تخرج العلماء والباحثين في كل علم.",
        "أصدرت المحكمة حكما قضائيا عادلا بالتعاون مع الحكومة ومجلس القضاء لتعزيز سيادة القانون.",
        "سافر المسافرون في رحلة طويلة عبر البحر وشاهدوا الأشجار العالية في الغابات الواسعة.",
        "انتصر الجيش في المعركة وحقق نصرا تاريخيا كبيرا سجلته كتب التاريخ عبر العصور.",
    ]
    for i in range(1, n_samples + 1):
        r = roots[i % len(roots)]
        phrase = sample_phrases[i % len(sample_phrases)]
        body = f"المقالة البحثية رقم {i}. جذور الكلمات الرئيسية هي {r}. {phrase} " * 15
        words = body.split()
        yield {
            "doc_id": f"syn_{i:06d}",
            "title": f"عنوان المقالة التجريبية {i} حول {r}",
            "body": body,
            "normalized_body": normalize_arabic(body),
            "url": f"https://ar.wikipedia.org/wiki/synthetic_{i}",
            "word_count": len(words),
            "char_count": len(body),
        }


def read_jsonl_corpus(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Stream articles lazily from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ingest_documents(
    docs_iterable: Iterable[Dict[str, Any]],
    batch_size: int = 10000,
    build_s0_index: bool = True,
    truncate_first: bool = False,
) -> Dict[str, Any]:
    """
    Stream documents into table `docs` using high-speed PostgreSQL COPY protocol.
    """
    stats: Dict[str, Any] = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "total_rows_ingested": 0,
        "copy_duration_sec": 0.0,
        "copy_throughput_rows_sec": 0.0,
        "s0_update_duration_sec": 0.0,
        "gin_index_duration_sec": 0.0,
        "vacuum_duration_sec": 0.0,
        "total_pipeline_duration_sec": 0.0,
    }

    pipeline_start = time.perf_counter()

    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            # 1. Optional Truncate
            if truncate_first:
                print("[INFO] Truncating table 'docs'...")
                cur.execute("TRUNCATE TABLE docs RESTART IDENTITY CASCADE;")

            # 2. High-Performance Bulk Streaming COPY
            print("[INFO] Starting high-performance streaming COPY into 'docs'...")
            copy_start = time.perf_counter()
            count = 0

            copy_sql = """
                COPY docs (
                    doc_id, title, body, normalized_body, s5_trgm_text, url, word_count, char_count
                ) FROM STDIN
            """
            with cur.copy(copy_sql) as copy:
                for doc in docs_iterable:
                    title = sanitize_text(doc.get("title", ""))
                    body = sanitize_text(doc.get("body", ""))
                    norm_body = sanitize_text(doc.get("normalized_body", ""))
                    if not norm_body:
                        norm_body = normalize_arabic(body)
                    w_count = int(doc.get("word_count", len(body.split())))
                    c_count = int(doc.get("char_count", len(body)))
                    url = sanitize_text(doc.get("url", ""))
                    doc_id = str(doc.get("doc_id", f"doc_{count+1}"))

                    # Write row tuple directly via binary/text serializer
                    copy.write_row((
                        doc_id,
                        title,
                        body,
                        norm_body,
                        norm_body,  # Populate s5_trgm_text simultaneously
                        url,
                        w_count,
                        c_count,
                    ))
                    count += 1
                    if count % batch_size == 0:
                        print(f"[INFO] ... streamed {count:,} documents")

            copy_duration = time.perf_counter() - copy_start
            throughput = count / copy_duration if copy_duration > 0 else 0.0

            stats["total_rows_ingested"] = count
            stats["copy_duration_sec"] = round(copy_duration, 4)
            stats["copy_throughput_rows_sec"] = round(throughput, 2)
            print(f"[SUCCESS] COPY completed: {count:,} rows in {copy_duration:.3f}s ({throughput:,.1f} rows/s).")

            # 3. Two-Stage Deferred Population (S0 tsvector & GIN Index)
            if build_s0_index and count > 0:
                print("[INFO] Computing S0 tsvector (to_tsvector('simple', normalized_body))...")
                ts_start = time.perf_counter()
                cur.execute("""
                    UPDATE docs 
                    SET s0_tsvector = to_tsvector('simple', normalized_body)
                    WHERE s0_tsvector IS NULL;
                """)
                ts_dur = time.perf_counter() - ts_start
                stats["s0_update_duration_sec"] = round(ts_dur, 4)
                print(f"[SUCCESS] s0_tsvector populated in {ts_dur:.3f}s ({count/ts_dur:,.1f} rows/s).")

                print("[INFO] Constructing GIN index 'idx_docs_s0_gin'...")
                gin_start = time.perf_counter()
                cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_s0_gin ON docs USING GIN (s0_tsvector);")
                gin_dur = time.perf_counter() - gin_start
                stats["gin_index_duration_sec"] = round(gin_dur, 4)
                print(f"[SUCCESS] GIN index built in {gin_dur:.3f}s.")

                print("[INFO] Executing VACUUM ANALYZE docs...")
                vac_start = time.perf_counter()
                cur.execute("VACUUM ANALYZE docs;")
                vac_dur = time.perf_counter() - vac_start
                stats["vacuum_duration_sec"] = round(vac_dur, 4)
                print(f"[SUCCESS] VACUUM ANALYZE completed in {vac_dur:.3f}s.")

    stats["total_pipeline_duration_sec"] = round(time.perf_counter() - pipeline_start, 4)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Arabic Wikipedia corpus into PostgreSQL.")
    parser.add_argument("--input", help="Path to input JSONL corpus file.")
    parser.add_argument("--synthetic-sample", type=int, help="Generate and ingest N synthetic articles for validation.")
    parser.add_argument("--truncate", action="store_true", help="Truncate table docs before ingestion.")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip deferred tsvector and GIN index build.")
    parser.add_argument("--batch-size", type=int, default=10000, help="Logging batch size.")
    args = parser.parse_args()

    if args.synthetic_sample:
        print(f"[INFO] Running in synthetic sample mode with {args.synthetic_sample} documents...")
        stream = generate_synthetic_articles(args.synthetic_sample)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"[ERROR] Input file does not exist: {args.input}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Streaming documents from: {args.input}")
        stream = read_jsonl_corpus(args.input)
    else:
        print("[ERROR] Please provide --input <file.jsonl> or --synthetic-sample <N>", file=sys.stderr)
        sys.exit(1)

    result = ingest_documents(
        docs_iterable=stream,
        batch_size=args.batch_size,
        build_s0_index=not args.skip_indexing,
        truncate_first=args.truncate,
    )
    print("\n[SUMMARY] Ingestion Report:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
