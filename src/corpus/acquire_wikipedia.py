"""
Arabic Wikipedia Corpus Acquisition and Quality Filtering Pipeline.
Downloads wikimedia/wikipedia (20231101.ar), applies strict morphological/content
filters, deduplicates via SHA-256 hashes, deterministically samples 100,000 articles (seed=42),
and prepares high-speed COPY-compatible records while maintaining a <3GB disk footprint.
"""

import os
import sys
import json
import time
import shutil
import random
import hashlib
import argparse
import subprocess
import urllib.request
from typing import List, Dict, Tuple, Set, Optional, Any
from pathlib import Path
import pyarrow.parquet as pq

# Import project normalizer
try:
    from src.preprocessing.normalize import normalize_arabic
except ImportError:
    import re

    def normalize_arabic(text: Optional[str], normalize_teh: bool = True) -> str:
        if not text:
            return ""
        # Strip Tashkeel & Dagger Alif & Quranic marks
        text = re.sub(r"[\u064B-\u0652\u0670\u0610-\u061A\u0653-\u065F\u06D6-\u06ED]", "", text)
        # Strip Tatweel
        text = re.sub(r"\u0640", "", text)
        # Normalize Alifs
        text = re.sub(r"[\u0622\u0623\u0625\u0671]", "\u0627", text)
        # Normalize Ya
        text = re.sub(r"[\u0649\u06CC\u06D0]", "\u064A", text)
        # Normalize Ta Marbuta
        if normalize_teh:
            text = re.sub(r"\u0629", "\u0647", text)
        # Normalize Digits
        for i in range(10):
            text = text.replace(chr(0x0660 + i), str(i)).replace(chr(0x06F0 + i), str(i))
        # Punctuation to space
        text = re.sub(r"[\u060C\u061B\u061F«»\"\'\[\]\(\)\{\}:;,.\-—/\\_!?]", " ", text)
        return re.sub(r"\s+", " ", text).strip()


# Upstream Hugging Face Parquet metadata for wikimedia/wikipedia (20231101.ar)
PARQUET_CHUNKS: List[Dict[str, Any]] = [
    {
        "filename": "0000.parquet",
        "size": 408241089,
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ar/train/0000.parquet",
    },
    {
        "filename": "0001.parquet",
        "size": 172208447,
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ar/train/0001.parquet",
    },
    {
        "filename": "0002.parquet",
        "size": 144792516,
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ar/train/0002.parquet",
    },
    {
        "filename": "0003.parquet",
        "size": 130751823,
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ar/train/0003.parquet",
    },
    {
        "filename": "0004.parquet",
        "size": 65094867,
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ar/train/0004.parquet",
    },
    {
        "filename": "0005.parquet",
        "size": 156441726,
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ar/train/0005.parquet",
    },
    {
        "filename": "0006.parquet",
        "size": 245773803,
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ar/train/0006.parquet",
    },
]

EXCLUDED_NAMESPACES = (
    "قالب:",
    "تصنيف:",
    "بوابة:",
    "مستخدم:",
    "ويكيبيديا:",
    "نقاش:",
    "ملف:",
    "مساعدة:",
    "template:",
    "category:",
    "portal:",
    "user:",
    "wikipedia:",
    "talk:",
    "file:",
    "help:",
)


def is_valid_article(
    title: str,
    text: str,
    min_words: int = 100,
    max_words: int = 25000,
) -> Tuple[bool, str, int]:
    """
    Apply fast, sequential quality checks to Wikipedia article.
    Returns: (is_valid, reason, word_count)
    """
    if not title or not text:
        return False, "empty_content", 0

    title_clean = title.strip()
    title_lower = title_clean.lower()

    # 1. Namespace filter
    for ns in EXCLUDED_NAMESPACES:
        if title_lower.startswith(ns):
            return False, f"namespace_{ns}", 0

    # 2. Disambiguation filter
    if "(توضيح)" in title_clean or "صفحة توضيح" in text[:300]:
        return False, "disambiguation", 0

    # 3. Redirect filter
    text_prefix = text[:150].strip()
    if text_prefix.startswith(("#تحويل", "#REDIRECT", "#redirect")):
        return False, "redirect", 0

    # 4. Length filter
    words = text.split()
    word_count = len(words)
    if word_count < min_words:
        return False, "stub_length", word_count
    if word_count > max_words:
        return False, "bloat_length", word_count

    return True, "valid", word_count


def download_chunk(
    chunk_meta: Dict[str, Any],
    dest_dir: Path,
    chunk_idx: int,
    total_chunks: int,
    max_retries: int = 5,
) -> Path:
    """
    Download a single parquet chunk with progress reporting and resume validation.
    """
    dest_path = dest_dir / chunk_meta["filename"]
    expected_size = chunk_meta["size"]

    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        print(
            f"[{chunk_idx+1}/{total_chunks}] {chunk_meta['filename']} already downloaded "
            f"({expected_size / (1024*1024):.1f} MB). Skipping."
        )
        return dest_path

    print(
        f"[{chunk_idx+1}/{total_chunks}] Downloading {chunk_meta['filename']} "
        f"({expected_size / (1024*1024):.1f} MB)..."
    )
    url = chunk_meta["url"]

    for attempt in range(1, max_retries + 1):
        try:
            # Try curl first for resume support and fast download
            ret = subprocess.run(
                ["curl", "-s", "-L", "--retry", "3", "--retry-delay", "2", "-C", "-", "-o", str(dest_path), url],
                timeout=300
            )
            if ret.returncode == 0 and dest_path.exists() and dest_path.stat().st_size == expected_size:
                print(f"  ✓ Download completed via curl: {chunk_meta['filename']}")
                return dest_path

            # Fallback to urllib if curl did not complete the full size
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Arabic-IR-Researcher/1.0 (Academic Research)"},
            )
            with urllib.request.urlopen(req, timeout=120) as response, open(
                dest_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

            downloaded_size = dest_path.stat().st_size
            if downloaded_size != expected_size:
                raise IOError(
                    f"Incomplete download: expected {expected_size} bytes, got {downloaded_size}"
                )

            print(f"  ✓ Download completed: {chunk_meta['filename']}")
            return dest_path
        except Exception as exc:
            print(f"  [Attempt {attempt}/{max_retries}] Download attempt failed: {exc}")
            if attempt == max_retries:
                if dest_path.exists():
                    dest_path.unlink()
                raise
            time.sleep(3 * attempt)

    return dest_path


def run_acquisition_pipeline(
    output_dir: str = "data/corpus",
    raw_dir: str = "data/raw_parquet",
    sample_size: int = 100000,
    seed: int = 42,
    min_words: int = 100,
    max_words: int = 25000,
    cleanup_raw: bool = True,
) -> Dict[str, Any]:
    """
    Execute full acquisition, filtering, deduplication, deterministic sampling,
    and storage cleanup pipeline.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Arabic Wikipedia Corpus Acquisition & Preprocessing Pipeline")
    print(f"Target: {sample_size:,} articles | Seed: {seed} | Word bounds: [{min_words}, {max_words}]")
    print("=" * 70)

    # 1. Download Parquet Chunks
    downloaded_paths: List[Path] = []
    total_download_bytes = 0
    t_start = time.time()

    for idx, chunk in enumerate(PARQUET_CHUNKS):
        fpath = download_chunk(chunk, raw_path, idx, len(PARQUET_CHUNKS))
        downloaded_paths.append(fpath)
        total_download_bytes += fpath.stat().st_size

    print(
        f"\nAll {len(downloaded_paths)} chunks verified. "
        f"Total raw size: {total_download_bytes / (1024*1024):.1f} MB"
    )

    # 2. Pass 1: Streaming Filter, Deduplication & Candidate Indexing
    print("\n[Pass 1/2] Scanning articles, applying quality filters, and tracking candidates...")
    seen_titles: Set[str] = set()
    seen_hashes: Set[str] = set()
    candidate_index: List[Tuple[int, int, int]] = []  # (int(doc_id), chunk_idx, row_in_chunk)

    filter_stats = {
        "scanned_total": 0,
        "filtered_namespace": 0,
        "filtered_disambiguation": 0,
        "filtered_redirect": 0,
        "filtered_stub": 0,
        "filtered_bloat": 0,
        "filtered_duplicate_title": 0,
        "filtered_duplicate_hash": 0,
        "eligible_candidates": 0,
    }

    t_pass1 = time.time()
    for chunk_idx, chunk_file in enumerate(downloaded_paths):
        table = pq.read_table(chunk_file)
        filter_stats["scanned_total"] += table.num_rows

        row_offset = 0
        for batch in table.to_batches(max_chunksize=10000):
            ids = batch.column("id").to_pylist()
            titles = batch.column("title").to_pylist()
            texts = batch.column("text").to_pylist()

            for i, (doc_id, title, text) in enumerate(zip(ids, titles, texts)):
                row_in_chunk = row_offset + i
                valid, reason, wc = is_valid_article(title, text, min_words, max_words)

                if not valid:
                    if reason.startswith("namespace"):
                        filter_stats["filtered_namespace"] += 1
                    elif reason == "disambiguation":
                        filter_stats["filtered_disambiguation"] += 1
                    elif reason == "redirect":
                        filter_stats["filtered_redirect"] += 1
                    elif reason == "stub_length":
                        filter_stats["filtered_stub"] += 1
                    elif reason == "bloat_length":
                        filter_stats["filtered_bloat"] += 1
                    continue

                # Title Deduplication
                norm_title = normalize_arabic(title).strip().lower()
                if norm_title in seen_titles:
                    filter_stats["filtered_duplicate_title"] += 1
                    continue

                # Content Hash Deduplication
                norm_text = normalize_arabic(text)
                content_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    filter_stats["filtered_duplicate_hash"] += 1
                    continue

                seen_titles.add(norm_title)
                seen_hashes.add(content_hash)

                int_id = int(doc_id) if str(doc_id).isdigit() else 0
                candidate_index.append((int_id, chunk_idx, row_in_chunk))
                filter_stats["eligible_candidates"] += 1

            row_offset += len(ids)

    print(f"Pass 1 complete in {time.time() - t_pass1:.1f}s.")
    print(f"  Scanned: {filter_stats['scanned_total']:,} articles")
    print(f"  Eligible Candidates: {filter_stats['eligible_candidates']:,}")
    print(
        f"  Filtered: Stubs={filter_stats['filtered_stub']:,}, "
        f"Disambig={filter_stats['filtered_disambiguation']:,}, "
        f"Duplicates={filter_stats['filtered_duplicate_hash']:,}"
    )

    if filter_stats["eligible_candidates"] < sample_size:
        raise ValueError(
            f"Insufficient eligible candidates ({filter_stats['eligible_candidates']:,}) "
            f"for target sample ({sample_size:,})"
        )

    # 3. Deterministic Sampling (seed=42)
    print(
        f"\nSorting candidates canonically and sampling exactly {sample_size:,} articles "
        f"(seed={seed})..."
    )
    candidate_index.sort(key=lambda item: item[0])

    random.seed(seed)
    sampled_records = random.sample(candidate_index, sample_size)
    sampled_records.sort(key=lambda item: item[0])

    # Organize sampled rows by chunk for single-pass reading
    chunk_row_map: Dict[int, Set[int]] = {}
    for int_id, chunk_idx, row_in_chunk in sampled_records:
        chunk_row_map.setdefault(chunk_idx, set()).add(row_in_chunk)

    # 4. Pass 2: Extract Sampled Articles and Write to JSONL
    output_jsonl = out_path / "corpus_100k.jsonl"
    print(f"\n[Pass 2/2] Extracting sampled documents to {output_jsonl}...")

    t_pass2 = time.time()
    total_written = 0
    word_lengths: List[int] = []

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for chunk_idx, chunk_file in enumerate(downloaded_paths):
            target_rows = chunk_row_map.get(chunk_idx, set())
            if not target_rows:
                continue

            table = pq.read_table(chunk_file)
            row_offset = 0

            for batch in table.to_batches(max_chunksize=10000):
                ids = batch.column("id").to_pylist()
                titles = batch.column("title").to_pylist()
                texts = batch.column("text").to_pylist()
                urls = batch.column("url").to_pylist()

                for i, (doc_id, title, text, url) in enumerate(zip(ids, titles, texts, urls)):
                    curr_row = row_offset + i
                    if curr_row in target_rows:
                        # Clean null bytes for PostgreSQL UTF-8 safety
                        clean_body = text.replace("\x00", "")
                        clean_title = title.replace("\x00", "")
                        norm_body = normalize_arabic(clean_body)
                        wc = len(clean_body.split())
                        cc = len(clean_body)
                        word_lengths.append(wc)

                        record = {
                            "doc_id": str(doc_id),
                            "title": clean_title,
                            "body": clean_body,
                            "normalized_body": norm_body,
                            "url": url,
                            "word_count": wc,
                            "char_count": cc,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_written += 1

                row_offset += len(ids)

    print(
        f"Pass 2 complete in {time.time() - t_pass2:.1f}s. "
        f"Total articles written: {total_written:,}"
    )
    jsonl_size_mb = output_jsonl.stat().st_size / (1024 * 1024)

    # 5. Generate Manifest
    manifest_path = out_path / "corpus_100k_manifest.json"
    hasher = hashlib.sha256()
    with open(output_jsonl, "rb") as f:
        while chunk_data := f.read(65536):
            hasher.update(chunk_data)
    file_sha256 = hasher.hexdigest()

    manifest_data = {
        "dataset_name": "wikimedia/wikipedia",
        "snapshot": "20231101.ar",
        "sample_size": total_written,
        "sample_seed": seed,
        "min_words": min_words,
        "max_words": max_words,
        "filter_statistics": filter_stats,
        "output_file": str(output_jsonl.name),
        "file_size_mb": round(jsonl_size_mb, 2),
        "sha256_checksum": file_sha256,
        "first_5_doc_ids": [r[0] for r in sampled_records[:5]],
        "last_5_doc_ids": [r[0] for r in sampled_records[-5:]],
        "metrics": {
            "avg_words_per_doc": round(sum(word_lengths) / len(word_lengths), 2),
            "min_words": min(word_lengths),
            "max_words": max(word_lengths),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"Manifest created: {manifest_path}")

    # 6. Automatic Disk Cleanup
    if cleanup_raw:
        print("\n[Disk Safety] Cleaning up raw parquet chunks to preserve storage...")
        for p in downloaded_paths:
            if p.exists():
                p.unlink()
        if raw_path.exists() and not any(raw_path.iterdir()):
            raw_path.rmdir()
        print("  ✓ Raw parquet directory pruned. Storage preserved.")

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"PIPELINE COMPLETE in {total_time:.1f}s. Final corpus size: {jsonl_size_mb:.1f} MB.")
    print("=" * 70)

    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire and sample Arabic Wikipedia corpus")
    parser.add_argument("--output-dir", default="data/corpus", help="Destination corpus directory")
    parser.add_argument("--raw-dir", default="data/raw_parquet", help="Staging raw parquet directory")
    parser.add_argument("--sample-size", type=int, default=100000, help="Number of sampled articles")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum words per article")
    parser.add_argument("--max-words", type=int, default=25000, help="Maximum words per article")
    parser.add_argument("--keep-raw", action="store_true", help="Preserve raw parquet chunks")

    args = parser.parse_args()
    run_acquisition_pipeline(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        min_words=args.min_words,
        max_words=args.max_words,
        cleanup_raw=not args.keep_raw,
    )
