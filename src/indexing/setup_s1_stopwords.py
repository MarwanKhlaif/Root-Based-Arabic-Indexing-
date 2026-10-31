#!/usr/bin/env python3
"""
setup_s1_stopwords.py - Generates Arabic stopwords and configures PostgreSQL text search.
Milestone 3 - Algorithmic Indexing Implementation (S1).

Responsibilities:
1. Compiles an exhaustive list of Arabic stopwords from NLTK corpus (~750 words)
   and standard classical Arabic particles (~150 words).
2. Generates dual representations (raw orthographic + normalized forms via normalize_arabic)
   to guarantee matching against both raw user queries and normalized corpus tokens.
3. Deploys the stopwords file into PostgreSQL container at:
   `/usr/share/postgresql/16/tsearch_data/arabic.stop`
4. Creates PostgreSQL Text Search Dictionary `arabic_stopwords` (TEMPLATE = simple)
   and Text Search Configuration `arabic_simple`.
5. Verifies configuration with canonical verification query:
   `SELECT to_tsvector('arabic_simple', 'في المكتبة');` -> `'المكتبة':2`
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Set

sys.path.insert(0, ".")

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
except ImportError:
    print("[ERROR] nltk package is required. Install via uv sync or uv add nltk.", file=sys.stderr)
    sys.exit(1)

from src.db.connection import get_connection
from src.preprocessing.normalize import normalize_arabic

CONTAINER_NAME = "arabic_ir_db"
PG_TSEARCH_DIR = "/usr/share/postgresql/16/tsearch_data"
STOPWORD_FILE_NAME = "arabic.stop"

# Core Arabic grammatical particles, pronouns, and prepositions
CUSTOM_ARABIC_STOPWORDS = {
    "في", "من", "على", "علي", "عن", "إلى", "الي", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللاتي", "اللواتي", "هو", "هي", "هم", "هن", "أنت", "انت",
    "أنتم", "انتم", "نحن", "أنا", "انا", "كان", "كانت", "كانوا", "يكون", "تكون",
    "أن", "ان", "إن", "بان", "بأن", "كما", "قد", "وقد", "ثم", "حيث", "حتى", "بين",
    "إذا", "اذا", "إذ", "اذ", "كل", "بعض", "غير", "فقط", "لكن", "ولكن", "أي", "اي",
    "أو", "او", "أم", "ام", "لا", "ما", "لم", "لن", "ليس", "ليست", "ليسوا", "فيه", "فيها",
    "منه", "منها", "عنه", "عنها", "إليه", "اليه", "إليها", "اليها", "به", "بها", "كثيرا", "جدا"
}


def build_stopword_list() -> Set[str]:
    """Compile and normalize full stopword set."""
    raw_stops = set(stopwords.words("arabic")).union(CUSTOM_ARABIC_STOPWORDS)
    combined: Set[str] = set()

    for word in raw_stops:
        w = word.strip()
        if not w:
            continue
        combined.add(w)
        # Add normalized version to match docs.normalized_body
        norm = normalize_arabic(w)
        if norm:
            combined.add(norm)

    return combined


def write_stopword_file(output_path: Path, words: Set[str]) -> None:
    """Write words in PostgreSQL tsearch dictionary format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("| Arabic Stopword List for PostgreSQL 16 arabic_simple\n")
        f.write(f"| Total Stopwords: {len(words)}\n")
        for w in sorted(words):
            f.write(f"{w}\n")


def deploy_to_container(local_file: Path) -> None:
    """Deploy stopword file into the running PostgreSQL container."""
    dest_path = f"{PG_TSEARCH_DIR}/{STOPWORD_FILE_NAME}"
    print(f"[INFO] Copying {local_file} -> {CONTAINER_NAME}:{dest_path}...")
    cp_cmd = ["docker", "cp", str(local_file), f"{CONTAINER_NAME}:{dest_path}"]
    subprocess.run(cp_cmd, check=True)

    chmod_cmd = ["docker", "exec", CONTAINER_NAME, "chmod", "644", dest_path]
    subprocess.run(chmod_cmd, check=True)
    print("[SUCCESS] Stopword file deployed and permissions set (644).")


def configure_postgresql_text_search() -> None:
    """Execute DDL in PostgreSQL to register dictionary and configuration."""
    ddl = """
    -- 1. Create or Replace Text Search Dictionary
    DROP TEXT SEARCH CONFIGURATION IF EXISTS arabic_simple CASCADE;
    DROP TEXT SEARCH DICTIONARY IF EXISTS arabic_stopwords CASCADE;

    CREATE TEXT SEARCH DICTIONARY arabic_stopwords (
        TEMPLATE = pg_catalog.simple,
        STOPWORDS = 'arabic'
    );

    -- 2. Create Text Search Configuration copying 'simple'
    CREATE TEXT SEARCH CONFIGURATION arabic_simple (COPY = simple);

    -- 3. Alter token mappings to route words through arabic_stopwords
    ALTER TEXT SEARCH CONFIGURATION arabic_simple
        ALTER MAPPING FOR word, asciiword, hword, asciihword
        WITH arabic_stopwords, simple;
    """
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            print("[SUCCESS] DDL executed: arabic_stopwords dictionary and arabic_simple config active.")

            # Verification query
            cur.execute("SELECT to_tsvector('arabic_simple', 'في المكتبة')::text;")
            res = cur.fetchone()[0]
            print(f"[VERIFY] to_tsvector('arabic_simple', 'في المكتبة') => {res}")
            assert "'المكتبة':2" in res or "'المكتبه':2" in res, f"Unexpected result: {res}"
            print("[PASS] Verification passed: 'في' stripped, 'المكتبة' preserved.")


def main() -> None:
    print("[STEP 1] Generating combined Arabic stopwords...")
    stops = build_stopword_list()
    print(f"[INFO] Generated {len(stops)} distinct stopwords.")

    local_path = Path("tsearch_data") / STOPWORD_FILE_NAME
    print(f"[STEP 2] Writing local stopword file to {local_path}...")
    write_stopword_file(local_path, stops)

    print(f"[STEP 3] Deploying to Docker container {CONTAINER_NAME}...")
    deploy_to_container(local_path)

    print("[STEP 4] Configuring PostgreSQL 16 Text Search...")
    configure_postgresql_text_search()

    print("[FINISHED] S1 arabic_simple setup complete.")


if __name__ == "__main__":
    main()
