"""
Search Service and Morphological Processing Engine for Arabic IR Web Interface.
Provides unified query transformations (S0–S5), live PostgreSQL execution,
intelligent snippet generation with keyword highlighting, and cached benchmark access.
"""

import os
import re
import html
import json
import time
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg
from psycopg.errors import QueryCanceled

logger = logging.getLogger(__name__)

from src.db.connection import get_connection, get_db_config
from src.preprocessing.normalize import normalize_arabic
from src.indexing.stemmers.larkey import stem_text as stem_larkey
from src.indexing.stemmers.isri import stem_text as stem_isri
from src.indexing.stemmers.khoja import stem_text as stem_khoja


# Core Arabic stopwords for S1
CUSTOM_ARABIC_STOPWORDS: Set[str] = {
    "في", "من", "على", "علي", "عن", "إلى", "الي", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللاتي", "اللواتي", "هو", "هي", "هم", "هن", "أنت", "انت",
    "أنتم", "انتم", "نحن", "أنا", "انا", "كان", "كانت", "كانوا", "يكون", "تكون",
    "أن", "ان", "إن", "بان", "بأن", "كما", "قد", "وقد", "ثم", "حيث", "حتى", "بين",
    "إذا", "اذا", "إذ", "اذ", "كل", "بعض", "غير", "فقط", "لكن", "ولكن", "أي", "اي",
    "أو", "او", "أم", "ام", "لا", "ما", "لم", "لن", "ليس", "ليست", "ليسوا", "فيه", "فيها",
    "منه", "منها", "عنه", "عنها", "إليه", "اليه", "إليها", "اليها", "به", "بها", "كثيرا", "جدا"
}

STOPWORDS_CACHE: Optional[Set[str]] = None


def get_arabic_stopwords() -> Set[str]:
    """Load normalized Arabic stopwords from tsearch_data or fallback."""
    global STOPWORDS_CACHE
    if STOPWORDS_CACHE is not None:
        return STOPWORDS_CACHE

    stopwords: Set[str] = set()
    # Normalize built-in custom list
    for w in CUSTOM_ARABIC_STOPWORDS:
        stopwords.add(w)
        stopwords.add(normalize_arabic(w, normalize_teh=True))

    # Try loading from tsearch_data/arabic.stop
    stop_file = os.path.join(os.path.dirname(__file__), "../../tsearch_data/arabic.stop")
    if os.path.exists(stop_file):
        try:
            with open(stop_file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        stopwords.add(word)
                        stopwords.add(normalize_arabic(word, normalize_teh=True))
        except Exception:
            pass

    STOPWORDS_CACHE = stopwords
    return STOPWORDS_CACHE


def generate_trigrams(text: str) -> List[str]:
    """
    Generate character trigrams equivalent to PostgreSQL pg_trgm.
    Each word is padded with two leading spaces and one trailing space.
    """
    if not text:
        return []
    words = text.split()
    trigrams: List[str] = []
    seen = set()
    for w in words:
        padded = f"  {w} "
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            if tri not in seen:
                seen.add(tri)
                trigrams.append(tri)
    return trigrams


SYSTEM_METADATA = {
    "S0": {
        "id": "S0",
        "name": "Surface Simple (Baseline)",
        "level": "Surface / Lexical",
        "badge_color": "neutral",
        "color_hex": "#171717",
        "description": "Exact surface form matching without morphological reduction or affix stripping.",
        "index_type": "GIN (simple tsvector)",
        "index_size_mb": 255.0,
        "index_size_pretty": "255 MB",
        "relative_size": "100%",
        "p_at_10": 0.350,
        "recall_at_100": 0.443,
    },
    "S1": {
        "id": "S1",
        "name": "Surface + Stopwords",
        "level": "Stopword Filtered",
        "badge_color": "neutral",
        "color_hex": "#262626",
        "description": "Surface matching with Arabic functional particles and stopwords removed.",
        "index_type": "GIN (arabic_simple tsvector)",
        "index_size_mb": 178.2,
        "index_size_pretty": "178 MB",
        "relative_size": "69.9%",
        "p_at_10": 0.265,
        "recall_at_100": 0.3851,
    },
    "S2": {
        "id": "S2",
        "name": "Light Stemming (Larkey)",
        "level": "Inflectional Morphology",
        "badge_color": "neutral",
        "color_hex": "#404040",
        "description": "Affix stripping removing regular prefixes and suffixes while preserving word stems.",
        "index_type": "GIN (light10 tsvector)",
        "index_size_mb": 142.7,
        "index_size_pretty": "143 MB",
        "relative_size": "56.0%",
        "p_at_10": 0.460,
        "recall_at_100": 0.6743,
    },
    "S3": {
        "id": "S3",
        "name": "Morphological Stemmer (ISRI)",
        "level": "Heuristic Root Extraction",
        "badge_color": "neutral",
        "color_hex": "#525252",
        "description": "Dictionary-free root extraction based on Arabic morphological pattern templates (Awzan).",
        "index_type": "GIN (isri root tsvector)",
        "index_size_mb": 96.8,
        "index_size_pretty": "97 MB",
        "relative_size": "38.0%",
        "p_at_10": 0.445,
        "recall_at_100": 0.7042,
    },
    "S4": {
        "id": "S4",
        "name": "Dictionary Root Stemmer (Khoja)",
        "level": "Dictionary-Based Root Extraction",
        "badge_color": "neutral",
        "color_hex": "#737373",
        "description": "Morphological root extraction validated against a canonical 4,748 Arabic root lexicon.",
        "index_type": "GIN (khoja root tsvector)",
        "index_size_mb": 96.0,
        "index_size_pretty": "96 MB",
        "relative_size": "37.6%",
        "p_at_10": 0.455,
        "recall_at_100": 0.7042,
    },
    "S5": {
        "id": "S5",
        "name": "Character Trigrams (pg_trgm)",
        "level": "Sub-word Character 3-gram",
        "badge_color": "neutral",
        "color_hex": "#0a0a0a",
        "description": "Character 3-gram decomposition resilient against typographical and surface variations.",
        "index_type": "GIN (gin_trgm_ops)",
        "index_size_mb": 207.1,
        "index_size_pretty": "207 MB",
        "relative_size": "81.2%",
        "p_at_10": 0.050,
        "recall_at_100": 0.1129,
    },
}



def transform_query(raw_query: str) -> Dict[str, Any]:
    """
    Transform input query across all 6 systems (S0–S5) and return structured representations.
    """
    clean_raw = raw_query.strip() if raw_query else ""
    normalized = normalize_arabic(clean_raw, normalize_teh=True)
    stopwords = get_arabic_stopwords()

    # S0: Surface Simple
    s0_tokens = normalized.split() if normalized else []
    s0_query = normalized

    # S1: Surface + Stopwords
    s1_tokens: List[str] = []
    removed_stops: List[str] = []
    for tok in s0_tokens:
        if tok in stopwords:
            removed_stops.append(tok)
        else:
            s1_tokens.append(tok)
    s1_query = " ".join(s1_tokens)

    # S2: Larkey Light10 Stemmer
    s2_query = stem_larkey(normalized) if normalized else ""
    s2_tokens = s2_query.split() if s2_query else []

    # S3: ISRI Stemmer
    s3_query = stem_isri(normalized) if normalized else ""
    s3_tokens = s3_query.split() if s3_query else []
    s3_unique = list(dict.fromkeys(s3_tokens))

    # S4: Khoja Stemmer
    s4_query = stem_khoja(normalized) if normalized else ""
    s4_tokens = s4_query.split() if s4_query else []
    s4_unique = list(dict.fromkeys(s4_tokens))

    # S5: Character Trigrams
    s5_trigrams = generate_trigrams(normalized)
    s5_query = normalized

    return {
        "raw_query": clean_raw,
        "normalized_query": normalized,
        "systems": {
            "S0": {
                **SYSTEM_METADATA["S0"],
                "transformed_query": s0_query,
                "tokens": s0_tokens,
                "display_tokens": s0_tokens,
            },
            "S1": {
                **SYSTEM_METADATA["S1"],
                "transformed_query": s1_query,
                "tokens": s1_tokens,
                "display_tokens": s1_tokens,
                "removed_stopwords": removed_stops,
            },
            "S2": {
                **SYSTEM_METADATA["S2"],
                "transformed_query": s2_query,
                "tokens": s2_tokens,
                "display_tokens": s2_tokens,
            },
            "S3": {
                **SYSTEM_METADATA["S3"],
                "transformed_query": s3_query,
                "tokens": s3_tokens,
                "display_tokens": s3_unique if s3_unique else s3_tokens,
            },
            "S4": {
                **SYSTEM_METADATA["S4"],
                "transformed_query": s4_query,
                "tokens": s4_tokens,
                "display_tokens": s4_unique if s4_unique else s4_tokens,
            },
            "S5": {
                **SYSTEM_METADATA["S5"],
                "transformed_query": s5_query,
                "tokens": s0_tokens,
                "trigrams": s5_trigrams,
                "trigrams_sample": s5_trigrams[:12],
                "trigrams_count": len(s5_trigrams),
            },
        },
    }


def highlight_snippet(
    text: Optional[str],
    keywords: List[str],
    max_length: int = 240,
) -> str:
    """
    Locate query keywords in text and return an HTML snippet with <mark> tags.
    """
    if not text:
        return ""

    clean_text = " ".join(text.split())
    if not clean_text:
        return ""

    # Filter meaningful keywords, sorted by length descending so longer words match first
    valid_keywords = sorted([re.escape(k) for k in keywords if k and len(k) >= 2], key=len, reverse=True)
    if not valid_keywords:
        # Return first max_length chars escaped
        snippet = html.escape(clean_text[:max_length])
        return snippet + ("..." if len(clean_text) > max_length else "")

    pattern = re.compile(r"(" + "|".join(valid_keywords) + r")", re.IGNORECASE)

    # Find earliest match position
    first_match = pattern.search(clean_text)
    if first_match:
        start_idx = max(0, first_match.start() - 60)
        end_idx = min(len(clean_text), start_idx + max_length)
        if start_idx > 0:
            # Advance to space boundary
            space_pos = clean_text.find(" ", start_idx)
            if 0 < space_pos < first_match.start():
                start_idx = space_pos + 1
        prefix = "..." if start_idx > 0 else ""
        suffix = "..." if end_idx < len(clean_text) else ""
        window = clean_text[start_idx:end_idx]
    else:
        prefix = ""
        suffix = "..." if len(clean_text) > max_length else ""
        window = clean_text[:max_length]

    # Safe HTML escape before injecting <mark>
    escaped_window = html.escape(window)

    def replacer(m):
        return f'<mark class="bg-neutral-200 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100 font-semibold px-1 rounded">{m.group(0)}</mark>'

    highlighted = pattern.sub(replacer, escaped_window)
    return f"{prefix}{highlighted}{suffix}"


def check_db_health() -> Dict[str, Any]:
    """
    Check PostgreSQL connectivity, total documents count, and index status.
    """
    config = get_db_config()
    info: Dict[str, Any] = {
        "connected": False,
        "host": config["host"],
        "port": config["port"],
        "database": config["dbname"],
        "total_documents": 0,
        "pg_version": None,
        "indexes_active": [],
        "message": "لم يتم الاتصال بقاعدة البيانات",
        "error": None,
    }

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                info["connected"] = True
                cur.execute("SELECT version();")
                info["pg_version"] = cur.fetchone()[0]

                # Check if docs table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'docs'
                    );
                """)
                table_exists = cur.fetchone()[0]

                if table_exists:
                    cur.execute("SELECT COUNT(*) FROM docs;")
                    info["total_documents"] = cur.fetchone()[0]

                    # Check indexes
                    cur.execute("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = 'docs' AND indexname LIKE 'idx_docs_%';
                    """)
                    info["indexes_active"] = [r[0] for r in cur.fetchall()]
                    info["message"] = f"متصل بنجاح - تم العثور على {info['total_documents']:,} مقال في قاعدة البيانات"
                else:
                    info["message"] = "متصل بقاعدة البيانات ولكن جدول docs غير موجود بعد"
    except Exception as exc:
        info["connected"] = False
        info["error"] = str(exc)
        info["message"] = f"تعذر الاتصال بـ PostgreSQL على {config['host']}:{config['port']}. تأكد من تشغيل حاوية arabic_ir_db."

    return info


def execute_system_search(
    system_id: str,
    raw_query: str,
    transformed_query: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Execute robust two-tier search for a single system against PostgreSQL.
    Tier 1 executes strict phrase/websearch matching; Tier 2 falls back to flexible
    OR / partial matching to ensure multi-word queries never return 0 results.
    """
    meta = SYSTEM_METADATA.get(system_id, {})
    sys_name = meta.get("name", system_id)

    # Fallback to normalized raw query if transformed query is empty
    effective_query = transformed_query.strip() if transformed_query else normalize_arabic(raw_query, normalize_teh=True)
    if not effective_query:
        return {
            "system_id": system_id,
            "system_name": sys_name,
            "results": [],
            "total_found": 0,
            "latency_ms": 0.0,
            "error": "Query is empty after processing.",
        }

    stopwords = get_arabic_stopwords()
    t0 = time.perf_counter()
    rows: List[Tuple] = []

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET random_page_cost = 1.5;")
                cur.execute("SET statement_timeout = '12000';")  # 12s ceiling

                norm_q = normalize_arabic(raw_query.strip(), normalize_teh=True)
                raw_tokens = [t for t in norm_q.split() if t]
                content_tokens = [t for t in raw_tokens if t not in stopwords and len(t) >= 2]
                if not content_tokens:
                    content_tokens = raw_tokens

                if system_id in ["S0", "S1", "S2", "S3", "S4"]:
                    config_map = {
                        "S0": ("s0_tsvector", "simple"),
                        "S1": ("s1_tsvector", "arabic_simple"),
                        "S2": ("s2_tsvector", "simple"),
                        "S3": ("s3_tsvector", "simple"),
                        "S4": ("s4_tsvector", "simple"),
                    }
                    col_name, cfg_name = config_map[system_id]

                    # Derive stems based on system
                    if system_id in ["S0", "S1"]:
                        stems = content_tokens
                    elif system_id == "S2":
                        stems = list(dict.fromkeys([stem_larkey(t) for t in content_tokens if t]))
                    elif system_id == "S3":
                        stems = list(dict.fromkeys([stem_isri(t) for t in content_tokens if t]))
                    else:  # S4
                        stems = list(dict.fromkeys([stem_khoja(t) for t in content_tokens if t]))

                    stems = [re.sub(r"[^\w\u0600-\u06FF]", "", t) for t in stems]
                    stems = [t for t in stems if t]
                    if not stems:
                        stems = raw_tokens

                    stems_set = set(stems)
                    cfg_query = "simple" if (system_id == "S1" and not [t for t in raw_tokens if t not in stopwords]) else cfg_name

                    ts_and_query = " & ".join(stems[:5])
                    ts_or_query = " | ".join(stems[:6])

                    title_phrase = f"%{norm_q}%"
                    title_tokens = list(dict.fromkeys(raw_tokens + stems))
                    title_patterns = [f"%{w}%" for w in title_tokens if len(w) >= 2][:8]

                    sql_search = f"""
                        WITH phrase_matches AS (
                            SELECT id, doc_id, title, substring(body from 1 for 3000) AS body_sample, {col_name}, 10.0::float4 AS boost
                            FROM docs
                            WHERE title ILIKE %s
                            LIMIT 10
                        ),
                        title_matches AS (
                            SELECT id, doc_id, title, substring(body from 1 for 3000) AS body_sample, {col_name}, 5.0::float4 AS boost
                            FROM docs
                            WHERE title ILIKE ANY(%s)
                            LIMIT 25
                        ),
                        and_matches AS (
                            SELECT id, doc_id, title, substring(body from 1 for 3000) AS body_sample, {col_name}, 2.0::float4 AS boost
                            FROM docs
                            WHERE {col_name} @@ to_tsquery('{cfg_query}', %s)
                            LIMIT 30
                        ),
                        or_matches AS (
                            SELECT id, doc_id, title, substring(body from 1 for 3000) AS body_sample, {col_name}, 0.0::float4 AS boost
                            FROM docs
                            WHERE {col_name} @@ to_tsquery('{cfg_query}', %s)
                            LIMIT 40
                        ),
                        combined AS (
                            SELECT * FROM phrase_matches
                            UNION SELECT * FROM title_matches
                            UNION SELECT * FROM and_matches
                            UNION SELECT * FROM or_matches
                        )
                        SELECT id, doc_id, title, body_sample,
                               (boost + ts_rank({col_name}, to_tsquery('{cfg_query}', %s))) AS score
                        FROM combined
                        ORDER BY score DESC, id ASC
                        LIMIT %s;
                    """
                    cur.execute(sql_search, (title_phrase, title_patterns, ts_and_query, ts_or_query, ts_or_query, limit))
                    rows = list(cur.fetchall())

                elif system_id == "S5":
                    clean_phrase = re.sub(r"[%_\\]", " ", norm_q).strip()
                    clean_phrase = " ".join(clean_phrase.split())
                    s5_tokens = [re.sub(r"[^\w\u0600-\u06FF]", "", t) for t in clean_phrase.split()]
                    s5_tokens = [t for t in s5_tokens if t and len(t) >= 2]
                    if not s5_tokens:
                        s5_tokens = raw_tokens

                    title_phrase = f"%{clean_phrase}%"
                    title_pats = [f"%{t}%" for t in s5_tokens[:6]]
                    like_tokens = [f"%{t}%" for t in s5_tokens[:4]]

                    sql_s5 = """
                        WITH phrase_c AS (
                            SELECT id, doc_id, title, substring(body from 1 for 3000) AS body_sample, s5_trgm_text, 5.0::float4 AS boost
                            FROM docs
                            WHERE title ILIKE %s
                            LIMIT 10
                        ),
                        title_c AS (
                            SELECT id, doc_id, title, substring(body from 1 for 3000) AS body_sample, s5_trgm_text, 3.0::float4 AS boost
                            FROM docs
                            WHERE title ILIKE ANY(%s)
                            LIMIT 20
                        ),
                        trgm_c AS (
                            SELECT id, doc_id, title, substring(body from 1 for 3000) AS body_sample, s5_trgm_text, 0.0::float4 AS boost
                            FROM docs
                            WHERE s5_trgm_text LIKE ANY(%s)
                            LIMIT 40
                        ),
                        combined AS (
                            SELECT * FROM phrase_c
                            UNION SELECT * FROM title_c
                            UNION SELECT * FROM trgm_c
                        )
                        SELECT id, doc_id, title, body_sample,
                               (boost + word_similarity(%s, s5_trgm_text)) AS score
                        FROM combined
                        ORDER BY score DESC, id ASC
                        LIMIT %s;
                    """
                    cur.execute(sql_s5, (title_phrase, title_pats, like_tokens, clean_phrase or norm_q, limit))
                    rows = list(cur.fetchall())

    except (QueryCanceled, psycopg.OperationalError) as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        err_text = str(exc)
        if "statement timeout" in err_text.lower() or isinstance(exc, QueryCanceled):
            user_msg = f"Search timed out for {sys_name} ({latency_ms} ms)."
        else:
            user_msg = f"Database query error: {err_text}"
        return {
            "system_id": system_id,
            "system_name": sys_name,
            "results": [],
            "total_found": 0,
            "latency_ms": latency_ms,
            "error": user_msg,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "system_id": system_id,
            "system_name": sys_name,
            "results": [],
            "total_found": 0,
            "latency_ms": latency_ms,
            "error": f"Database error: {exc}",
        }

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Explicit deduplication and strictly monotonic score descending sort
    deduped_rows: Dict[Any, Tuple] = {}
    for r in rows:
        pk = r[0]
        score_val = float(r[4]) if (len(r) > 4 and r[4] is not None) else 0.0
        if pk not in deduped_rows:
            deduped_rows[pk] = r
        else:
            prev_score = float(deduped_rows[pk][4]) if (len(deduped_rows[pk]) > 4 and deduped_rows[pk][4] is not None) else 0.0
            if score_val > prev_score:
                deduped_rows[pk] = r

    unique_rows = list(deduped_rows.values())
    unique_rows.sort(
        key=lambda r: (
            float(r[4]) if (len(r) > 4 and r[4] is not None) else 0.0,
            -int(r[0]) if (len(r) > 0 and isinstance(r[0], (int, float))) else 0,
        ),
        reverse=True,
    )
    rows = unique_rows[:limit]

    # Build matched forms for each result and format snippet
    results: List[Dict[str, Any]] = []
    norm_q = normalize_arabic(raw_query.strip(), normalize_teh=True)
    raw_tokens = [t for t in norm_q.split() if t]
    content_tokens = [t for t in raw_tokens if t not in stopwords and len(t) >= 2] or raw_tokens

    if system_id in ["S0", "S1"]:
        stems_set = set(content_tokens)
    elif system_id == "S2":
        stems_set = set(stem_larkey(t) for t in content_tokens if t)
    elif system_id in ["S3", "S4"]:
        stems_set = set(stem_isri(t) for t in content_tokens if t)
    else:
        stems_set = set(content_tokens)

    for rank, row in enumerate(rows, start=1):
        doc_pk, doc_id, title, body_sample, score = row

        # Extract morphological forms matched in body_sample
        matched_forms: Set[str] = set()
        if body_sample:
            for word in body_sample.split():
                cw = word.strip(".,:;()[]\"'«»-")
                if not cw or len(cw) < 2:
                    continue
                if system_id in ["S3", "S4"]:
                    if stem_isri(cw) in stems_set:
                        matched_forms.add(cw)
                elif system_id == "S2":
                    if stem_larkey(cw) in stems_set:
                        matched_forms.add(cw)
                elif system_id == "S5":
                    if any(t in cw for t in raw_tokens if len(t) >= 2):
                        matched_forms.add(cw)
                else:
                    if cw in stems_set or cw in raw_tokens:
                        matched_forms.add(cw)

        # Highlight terms: raw tokens + stems + all detected morphological matches
        highlight_keywords = list(set(raw_tokens) | stems_set | matched_forms)
        snippet = highlight_snippet(body_sample, highlight_keywords)

        results.append({
            "rank": rank,
            "id": doc_pk,
            "doc_id": doc_id,
            "title": title,
            "score": round(float(score), 4) if score is not None else 0.0,
            "snippet": snippet,
            "matched_forms": sorted(matched_forms)[:6],
            "url": f"https://ar.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else None,
        })

    return {
        "system_id": system_id,
        "system_name": sys_name,
        "results": results,
        "total_found": len(results),
        "latency_ms": latency_ms,
        "error": None,
    }



# Cache for golden queries and benchmark metrics
GOLDEN_QUERIES_CACHE: Optional[List[Dict[str, Any]]] = None
QRELS_GRADED_CACHE: Optional[Dict[str, Any]] = None
IR_METRICS_CACHE: Optional[Dict[str, Any]] = None
INDEXING_BENCHMARKS_CACHE: Optional[Dict[str, Any]] = None
OVERSTEMMING_CACHE: Optional[Dict[str, Any]] = None


def load_golden_queries() -> List[Dict[str, Any]]:
    """Load the 20 golden queries with linguistic annotations."""
    global GOLDEN_QUERIES_CACHE
    if GOLDEN_QUERIES_CACHE is not None:
        return GOLDEN_QUERIES_CACHE

    path = os.path.join(os.path.dirname(__file__), "../../queries/golden_queries_20.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            GOLDEN_QUERIES_CACHE = data.get("queries", [])
    else:
        GOLDEN_QUERIES_CACHE = []

    return GOLDEN_QUERIES_CACHE


def load_qrels_graded() -> Dict[str, Any]:
    """Load ground-truth relevance grades for golden queries."""
    global QRELS_GRADED_CACHE
    if QRELS_GRADED_CACHE is not None:
        return QRELS_GRADED_CACHE

    path = os.path.join(os.path.dirname(__file__), "../../queries/qrels_graded.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            QRELS_GRADED_CACHE = json.load(f)
    else:
        QRELS_GRADED_CACHE = {}

    return QRELS_GRADED_CACHE


def load_benchmark_metrics() -> Dict[str, Any]:
    """Load formal IR evaluation metrics (P@10, MAP, nDCG, Recall, Latency)."""
    global IR_METRICS_CACHE
    if IR_METRICS_CACHE is not None:
        return IR_METRICS_CACHE

    path = os.path.join(os.path.dirname(__file__), "../../results/ir_evaluation_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            IR_METRICS_CACHE = json.load(f)
    else:
        IR_METRICS_CACHE = {}

    return IR_METRICS_CACHE


def load_indexing_benchmarks() -> Dict[str, Any]:
    """Load GIN index sizes and build speed metrics."""
    global INDEXING_BENCHMARKS_CACHE
    if INDEXING_BENCHMARKS_CACHE is not None:
        return INDEXING_BENCHMARKS_CACHE

    path = os.path.join(os.path.dirname(__file__), "../../results/indexing_benchmarks.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            INDEXING_BENCHMARKS_CACHE = json.load(f)
    else:
        INDEXING_BENCHMARKS_CACHE = {}

    return INDEXING_BENCHMARKS_CACHE


def load_overstemming_data() -> Dict[str, Any]:
    """Load over-stemming and semantic collision analysis."""
    global OVERSTEMMING_CACHE
    if OVERSTEMMING_CACHE is not None:
        return OVERSTEMMING_CACHE

    path = os.path.join(os.path.dirname(__file__), "../../results/overstemming_fallout.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            OVERSTEMMING_CACHE = json.load(f)
    else:
        OVERSTEMMING_CACHE = {}

    return OVERSTEMMING_CACHE


def get_demo_search_results(
    query_text: str,
    system_id: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Fallback demo results extracted from qrels_graded.json when PostgreSQL is offline or empty.
    Provides immediate interactive experience for the 20 golden queries or general terms.
    """
    meta = SYSTEM_METADATA.get(system_id, {})
    qrels = load_qrels_graded()
    golden_list = load_golden_queries()

    # Find matching golden query
    matched_qid: Optional[str] = None
    for gq in golden_list:
        term = gq.get("query_term", "")
        if query_text.strip() == term.strip() or query_text.strip() in gq.get("expected_surface_forms", []):
            matched_qid = gq.get("query_id")
            break

    # If no exact match, fallback to Q01 as illustrative example
    if not matched_qid:
        matched_qid = "Q01"

    candidates: List[Dict[str, Any]] = []
    if qrels and "queries" in qrels and matched_qid in qrels["queries"]:
        qdata = qrels["queries"][matched_qid]
        for c in qdata.get("candidates", []):
            # Check if this candidate was retrieved by this system
            retrieved_by = c.get("systems_retrieved", [])
            if system_id in retrieved_by:
                candidates.append(c)

    # Sort by grade descending
    candidates.sort(key=lambda x: x.get("grade", 0), reverse=True)
    selected = candidates[:limit]

    results: List[Dict[str, Any]] = []
    for rank, c in enumerate(selected, start=1):
        doc_id = str(c.get("doc_id", ""))
        title = c.get("title", f"مقال {doc_id}")
        grade = c.get("grade", 0)
        rationale = c.get("rationale", "")
        body_sample = f"{title}. {rationale}. مقتطف توضيحي من المقال حول موضوع البحث ومطابقة الكلمات الصرفية في ويكيبيديا العربية."

        results.append({
            "rank": rank,
            "id": rank,
            "doc_id": doc_id,
            "title": title,
            "score": round(0.5 + (grade * 0.15), 3),
            "grade": grade,
            "snippet": highlight_snippet(body_sample, [query_text, title]),
            "url": f"https://ar.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "is_demo": True,
            "rationale": rationale,
        })

    # Realistic simulated latency from benchmark
    ir_metrics = load_benchmark_metrics()
    macro_summary = ir_metrics.get("systems_macro_summary", {}).get(system_id, {})
    simulated_latency = macro_summary.get("macro_warm_latency_ms", 15.0)

    return {
        "system_id": system_id,
        "system_name": meta.get("name", system_id),
        "results": results,
        "total_found": len(results),
        "latency_ms": round(float(simulated_latency), 2),
        "is_demo": True,
        "matched_query_id": matched_qid,
        "error": None,
    }


def search_all_systems(
    raw_query: str,
    selected_systems: Optional[List[str]] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Run multi-system search across selected systems (defaults to all S0-S5).
    Gracefully falls back to demo mode if database is offline.
    """
    if not selected_systems:
        selected_systems = ["S0", "S1", "S2", "S3", "S4", "S5"]

    transformation = transform_query(raw_query)
    db_health = check_db_health()
    use_live_db = db_health["connected"] and db_health["total_documents"] > 0

    valid_systems = [s for s in selected_systems if s in SYSTEM_METADATA]

    def _execute_for_system(sys_id: str) -> Tuple[str, Dict[str, Any]]:
        trans_query = transformation["systems"][sys_id]["transformed_query"]
        if use_live_db:
            try:
                res = execute_system_search(
                    system_id=sys_id,
                    raw_query=raw_query,
                    transformed_query=trans_query,
                    limit=limit,
                )
                res["is_demo"] = False
                return sys_id, res
            except Exception as e:
                # If error, fallback to demo mode for this system
                res = get_demo_search_results(raw_query, sys_id, limit=limit)
                res["warning"] = f"تعذر البحث الحي: {e} - تم تفعيل وضع المعاينة"
                return sys_id, res
        else:
            # PostgreSQL offline or no docs: return high quality demo results
            res = get_demo_search_results(raw_query, sys_id, limit=limit)
            return sys_id, res

    systems_results: Dict[str, Any] = {}
    if valid_systems:
        with ThreadPoolExecutor(max_workers=min(len(valid_systems), 6)) as executor:
            future_to_sys = {executor.submit(_execute_for_system, s): s for s in valid_systems}
            for future in as_completed(future_to_sys):
                sys_id, res = future.result()
                systems_results[sys_id] = res

    # Preserve the requested systems order in output dictionary
    ordered_results = {s: systems_results[s] for s in valid_systems if s in systems_results}

    return {
        "query": raw_query,
        "transformation": transformation,
        "db_connected": db_health["connected"],
        "is_demo_mode": not use_live_db,
        "systems": ordered_results,
    }
