#!/usr/bin/env python3
"""
src/benchmarks/pool_and_score_relevance.py

Ground Truth Relevance Assessment & Pooling Engine (TREC-Style)
Collects Top-20 candidate documents across S0–S5 for each of the 20 golden queries in queries/golden_queries_20.json.
Assesses relevance grades (0, 1, 2, 3) using title, lead snippet, and criteria from golden_queries_20.json with fatal collision detection.
Computes IDCG@10 = sum_{i=1}^{min(10, |Rel|)} (2^{rel_{(i)}} - 1) / log2(i + 1).
Exports queries/qrels_graded.json and standard TREC queries/qrels.tsv.
"""

import json
import math
import os
import re
import sys
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.db.connection import get_cursor
from src.preprocessing.normalize import normalize_arabic
from src.indexing.stemmers.larkey import stem_text as stem_larkey
from src.indexing.stemmers.isri import stem_text as stem_isri
from src.indexing.stemmers.khoja import stem_text as stem_khoja


def calculate_idcg10(grades: List[int]) -> float:
    """
    Computes Ideal Discounted Cumulative Gain at Rank 10 (IDCG@10).
    IDCG@10 = sum_{i=1}^{min(10, |Rel|)} (2^{rel_{(i)}} - 1) / log2(i + 1)
    where rel_{(i)} are sorted descending.
    """
    sorted_grades = sorted([g for g in grades if g > 0], reverse=True)
    idcg = 0.0
    for i, grade in enumerate(sorted_grades[:10], start=1):
        gain = (2.0 ** grade) - 1.0
        discount = math.log2(i + 1.0)
        idcg += gain / discount
    return idcg


# Explicit Semantic Collision and Positive Patterns Registry
SEMANTIC_COLLISION_RULES = {
    "Q01": {
        "collision": r"\b(كتيبة|كتائب|مكتب عمل|مكتب بريد|مكتب تنسيق|اكتتاب)\b",
        "positive": ["كتاب", "كتب", "مكتبة", "مؤلف", "تأليف", "رواية", "كاتب", "مخطوط", "طبع ونشر"],
        "core": ["كتاب", "مكتبة", "مؤلف", "كاتب", "كتب"],
    },
    "Q02": {
        "collision": r"\b(محصول|حبوب|حصاد|درس المحصول|مدروس بعناية|مدروسة بعناية)\b",
        "positive": ["مدرسة", "مدارس", "دراسة", "تعليم", "تدريس", "جامعة", "طالب", "منهاج", "مناهج"],
        "core": ["مدرسة", "تعليم", "دراسة", "تدريس"],
    },
    "Q03": {
        "collision": r"\b(تحكيم رياضي|حكام المباريات|حكم المباراة|كرة القدم)\b",
        "positive": ["حكومة", "حكم", "محكمة", "قضاء", "دستور", "سياسة", "قاضي", "أحكام", "حكام"],
        "core": ["حكومة", "محكمة", "قضاء", "حكم"],
    },
    "Q04": {
        "collision": r"\b(مخرج سينمائي|إخراج سينمائي|تخرج الطلاب|حفل تخرج)\b",
        "positive": ["خارجية", "استخراج", "خروج", "وزارة الخارجية", "استخراج النفط", "علاقات خارجية"],
        "core": ["خارجية", "وزارة الخارجية", "استخراج", "خروج"],
    },
    "Q05": {
        "collision": r"\b(إعلام|الإعلام|وسائل الإعلام|عالم الرياضة|كأس العالم)\b",
        "positive": ["علم", "علماء", "تعليم", "معرفة", "فيزياء", "كيمياء", "أبحاث", "علوم"],
        "core": ["علم", "علماء", "علوم", "معرفة"],
    },
    "Q06": {
        "collision": r"\b(سفارة|السفارة|سفارات|سفير|أسفار العهد)\b",
        "positive": ["سفر", "مسافر", "سياحة", "رحلة", "طيران", "تنقل", "سياحي", "أسفار"],
        "core": ["سفر", "سياحة", "رحلة", "مسافر"],
    },
    "Q07": {
        "collision": r"\b(الناصرة|عبد الناصر|نصرانية|النصرانية|مدينة نصر)\b",
        "positive": ["انتصار", "نصر", "معركة", "حرب", "جيش", "ظفر", "انتصارات"],
        "core": ["نصر", "انتصار", "ظفر"],
    },
    "Q08": {
        "collision": r"\b(معمل كيميائي|معامل كيميائية|معاملات تجارية|معاملة)\b",
        "positive": ["عمل", "عمال", "صناعة", "وظيفة", "إنتاج", "عمالة", "سوق العمل"],
        "core": ["عمل", "عمال", "وظيفة", "صناعة"],
    },
    "Q09": {
        "collision": r"\b(استسلام|استسلم|سلم درج|سلالم|الفتح الإسلامي|تاريخ الإسلام|إسلامي)\b",
        "positive": ["سلام", "معاهدة سلام", "صلح", "هدنة", "حمامة السلام", "اتفاقية سلام"],
        "core": ["سلام", "معاهدة سلام", "صلح", "هدنة"],
    },
    "Q10": {
        "collision": r"$^",  # No collision for tree
        "positive": ["شجر", "شجرة", "أشجار", "غابات", "نبات", "تشجير", "أخشاب"],
        "core": ["شجر", "شجرة", "أشجار", "غابات"],
    },
    "Q11": {
        "collision": r"\b(انقلاب|انقلابات|انقلابي|تقلبات|تقلب)\b",
        "positive": ["قلب", "قلوب", "عضلة القلب", "شرايين", "أمراض القلب", "صمامات القلب", "قسطرة"],
        "core": ["قلب", "قلوب", "عضلة القلب", "شرايين"],
    },
    "Q12": {
        "collision": r"\b(مطرقة|مطارق|طارق بن زياد)\b",
        "positive": ["طريق", "طرق", "مواصلات", "نقل", "شوارع", "طريقة", "منهج", "طرق سريعة"],
        "core": ["طريق", "طرق", "مواصلات", "شوارع"],
    },
    "Q13": {
        "collision": r"\b(أصحاب الشركات|أصحاب العمل|أصحاب رؤوس الأموال)\b",
        "positive": ["صحابة", "أصحاب النبي", "الصحابي", "رفقة", "صحابة رسول الله", "أصحاب"],
        "core": ["صحابة", "الصحابي", "أصحاب النبي"],
    },
    "Q14": {
        "collision": r"\b(قال فلان|روى|حدثنا)\b",
        "positive": ["مقولة", "أقوال", "مقالة", "بلاغة", "فن المقالة", "حكم وأقوال", "قول"],
        "core": ["مقولة", "أقوال", "مقالة", "قول"],
    },
    "Q15": {
        "collision": r"\b(موقف سيارات|توقيف|موقوف|حظر)\b",
        "positive": ["وقف", "أوقاف", "وقف إسلامي", "حبس", "الأوقاف الإسلامية", "حجة وقفية"],
        "core": ["وقف", "أوقاف", "وقف إسلامي"],
    },
    "Q16": {
        "collision": r"\b(مرمى|حارس المرمى|في المرمى|ضربة مرمى)\b",
        "positive": ["رماية", "قوس", "سهم", "رمي", "رياضة الرماية", "رماة السهام"],
        "core": ["رماية", "رمي", "قوس وسهم"],
    },
    "Q17": {
        "collision": r"\b(وفيات|الوفيات|توفي|وفاة|معدل الوفيات|جائحة|فيروس كورونا)\b",
        "positive": ["وفاء", "إخلاص", "عهد", "الوفاء بالعهد", "صدق ووفاء"],
        "core": ["وفاء", "إخلاص", "الوفاء بالعهد"],
    },
    "Q18": {
        "collision": r"\b(تعيين|تعيينات|معاينة|عين ماء|ينابيع|جاسوس|أعيان|مجلس الأعيان|عين العرب)\b",
        "positive": ["عين", "عيون", "بصر", "رؤية", "طب العيون", "قرنية", "شبكية", "عين الإنسان"],
        "core": ["عين", "عيون", "طب العيون", "قرنية", "شبكية"],
    },
    "Q19": {
        "collision": r"\b(شعر الإنسان|العناية بالشعر|تساقط الشعر|بصيلات|مشاعر|شعور|أشعرية)\b",
        "positive": ["شعر", "شاعر", "شعراء", "قصيدة", "قصائد", "أشعار", "ديوان", "أوزان الشعر"],
        "core": ["شعر", "شاعر", "قصيدة", "قصائد", "ديوان"],
    },
    "Q20": {
        "collision": r"\b(مملكة البحرين|المنامة|بحريني|بحرينية|مجلس النواب البحريني|آل خليفة)\b",
        "positive": ["بحر", "بحار", "محيط", "محيطات", "أسماك", "ملاحة", "سفينة", "أحياء بحرية", "شاطئ"],
        "core": ["بحر", "بحار", "محيط", "أحياء بحرية"],
    },
}


def evaluate_document_relevance(
    doc: Dict[str, Any],
    query: Dict[str, Any]
) -> Tuple[int, str]:
    """
    Automated semantic assessment of document against query criteria.
    Assigns grades 0, 1, 2, 3 based on golden_queries_20.json rules.
    """
    qid = query["query_id"]
    term = query["query_term"]
    expected_forms = query.get("expected_surface_forms", [])
    title = doc.get("title", "")
    body_snippet = doc.get("body_preview", "")
    full_text = f"{title} {body_snippet}"

    # Normalize texts for robust matching
    norm_title = normalize_arabic(title, normalize_teh=True)
    norm_snippet = normalize_arabic(body_snippet, normalize_teh=True)
    norm_full = f"{norm_title} {norm_snippet}"

    rule = SEMANTIC_COLLISION_RULES.get(qid, {})
    col_pattern = rule.get("collision", "")
    pos_words = [normalize_arabic(w, True) for w in rule.get("positive", [])]
    core_words = [normalize_arabic(w, True) for w in rule.get("core", [])]
    norm_expected = [normalize_arabic(w, True) for w in expected_forms]

    # 1. Fatal Semantic Collision Checks (Grade 0 Overrides)
    if col_pattern and re.search(col_pattern, norm_title):
        # Unless the title explicitly has a core positive word
        if not any(w in norm_title for w in core_words):
            return 0, f"Severe semantic collision in title matching pattern '{col_pattern}'."

    if col_pattern and re.search(col_pattern, norm_snippet):
        col_count = len(re.findall(col_pattern, norm_snippet))
        pos_count = sum(1 for w in pos_words if w in norm_snippet)
        if col_count > 0 and col_count >= pos_count and not any(w in norm_title for w in core_words):
            return 0, f"Body snippet dominated by semantic collision terms ({col_count} collisions vs {pos_count} positive)."

    # 2. Grade 3: Highly Relevant / Core Encyclopedic Focus
    norm_term = normalize_arabic(term, True)
    if norm_title == norm_term or any(w == norm_title for w in core_words):
        return 3, f"Exact core title topic match on query '{term}'."
    if any(w in norm_title for w in core_words) and any(w in norm_snippet[:300] for w in pos_words):
        return 3, f"Core topic affirmed in title ('{title}') and opening definition."

    # 3. Grade 2: Substantially Relevant
    title_pos_matches = [w for w in pos_words if w in norm_title]
    snippet_pos_matches = [w for w in pos_words if w in norm_snippet]
    if len(title_pos_matches) >= 1 or len(snippet_pos_matches) >= 3 or (len(snippet_pos_matches) >= 2 and any(w in norm_full for w in norm_expected)):
        return 2, f"Substantive topical discussion with {len(snippet_pos_matches)} domain keywords."

    # 4. Grade 1: Marginally Relevant / Incidental Mention
    if norm_term in norm_full or any(w in norm_full for w in norm_expected) or len(snippet_pos_matches) >= 1:
        return 1, "Incidental or passing mention of query concept without core topical focus."

    # 5. Grade 0: Unrelated / Off-topic
    return 0, "Unrelated document with zero topical relevance."


def run_pooling_and_scoring(
    queries_path: str = "queries/golden_queries_20.json",
    output_graded_json: str = "queries/qrels_graded.json",
    output_qrels_tsv: str = "queries/qrels.tsv"
) -> Dict[str, Any]:
    """
    Executes top-20 pooling across S0-S5, scores all unique candidates,
    computes IDCG@10, and outputs standard qrels artifacts.
    """
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]

    qrels_data = {
        "metadata": {
            "title": "Arabic IR Ground Truth Graded Relevance Annotations (QRELS)",
            "benchmark": "TREC-Style Top-20 Pooling Across S0-S5 in PostgreSQL 16",
            "corpus": "100,000 Arabic Wikipedia Articles (Seed 42)",
            "total_queries": len(queries),
            "binary_relevance_threshold": 2,  # Grade >= 2 is relevant
            "idcg_10_per_query": {}
        },
        "qrels": {},      # Direct mapping {qid: {doc_id: grade}} for fast metric lookup
        "queries": {}     # Detailed candidate breakdown
    }

    tsv_rows = []

    with get_cursor(as_dict=True) as cur:
        cur.execute("SET random_page_cost = 1.5;")
        cur.execute("SET work_mem = '64MB';")

        for q in queries:
            qid = q["query_id"]
            term = q["query_term"]
            print(f"[*] Processing Pooling for {qid}: '{term}'...")

            pool: Dict[str, Dict[str, Any]] = {}

            # --- S0: Surface Simple ---
            cur.execute("""
                SELECT doc_id, title, substring(body from 1 for 600) AS body_preview
                FROM docs
                WHERE s0_tsvector @@ plainto_tsquery('simple', %s)
                ORDER BY ts_rank(s0_tsvector, plainto_tsquery('simple', %s)) DESC, id ASC
                LIMIT 20;
            """, (term, term))
            for r in cur.fetchall():
                d = pool.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "title": r["title"], "body_preview": r["body_preview"], "systems": []})
                d["systems"].append("S0")

            # --- S1: Surface + Stopwords ---
            cur.execute("""
                SELECT doc_id, title, substring(body from 1 for 600) AS body_preview
                FROM docs
                WHERE s1_tsvector @@ plainto_tsquery('arabic_simple', %s)
                ORDER BY ts_rank(s1_tsvector, plainto_tsquery('arabic_simple', %s)) DESC, id ASC
                LIMIT 20;
            """, (term, term))
            for r in cur.fetchall():
                d = pool.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "title": r["title"], "body_preview": r["body_preview"], "systems": []})
                if "S1" not in d["systems"]:
                    d["systems"].append("S1")

            # --- S2: Larkey light10 ---
            s2_q = stem_larkey(normalize_arabic(term, True))
            cur.execute("""
                SELECT doc_id, title, substring(body from 1 for 600) AS body_preview
                FROM docs
                WHERE s2_tsvector @@ plainto_tsquery('simple', %s)
                ORDER BY ts_rank(s2_tsvector, plainto_tsquery('simple', %s)) DESC, id ASC
                LIMIT 20;
            """, (s2_q, s2_q))
            for r in cur.fetchall():
                d = pool.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "title": r["title"], "body_preview": r["body_preview"], "systems": []})
                if "S2" not in d["systems"]:
                    d["systems"].append("S2")

            # --- S3: ISRI morphological pattern ---
            s3_q = stem_isri(normalize_arabic(term, True))
            cur.execute("""
                SELECT doc_id, title, substring(body from 1 for 600) AS body_preview
                FROM docs
                WHERE s3_tsvector @@ plainto_tsquery('simple', %s)
                ORDER BY ts_rank(s3_tsvector, plainto_tsquery('simple', %s)) DESC, id ASC
                LIMIT 20;
            """, (s3_q, s3_q))
            for r in cur.fetchall():
                d = pool.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "title": r["title"], "body_preview": r["body_preview"], "systems": []})
                if "S3" not in d["systems"]:
                    d["systems"].append("S3")

            # --- S4: Khoja dictionary root ---
            s4_q = stem_khoja(normalize_arabic(term, True))
            cur.execute("""
                SELECT doc_id, title, substring(body from 1 for 600) AS body_preview
                FROM docs
                WHERE s4_tsvector @@ plainto_tsquery('simple', %s)
                ORDER BY ts_rank(s4_tsvector, plainto_tsquery('simple', %s)) DESC, id ASC
                LIMIT 20;
            """, (s4_q, s4_q))
            for r in cur.fetchall():
                d = pool.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "title": r["title"], "body_preview": r["body_preview"], "systems": []})
                if "S4" not in d["systems"]:
                    d["systems"].append("S4")

            # --- S5: Trigram word similarity (escaped %%> for psycopg 3) ---
            s5_q = normalize_arabic(term, normalize_teh=True)
            cur.execute("""
                SELECT doc_id, title, substring(body from 1 for 600) AS body_preview
                FROM docs
                WHERE s5_trgm_text %%> %s
                ORDER BY word_similarity(%s, s5_trgm_text) DESC, id ASC
                LIMIT 20;
            """, (s5_q, s5_q))
            for r in cur.fetchall():
                d = pool.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "title": r["title"], "body_preview": r["body_preview"], "systems": []})
                if "S5" not in d["systems"]:
                    d["systems"].append("S5")

            # Assess relevance for each pooled candidate
            evaluated_docs = []
            assigned_grades = []
            qrels_data["qrels"][qid] = {}

            for doc_id, doc_info in pool.items():
                grade, rationale = evaluate_document_relevance(doc_info, q)
                assigned_grades.append(grade)
                qrels_data["qrels"][qid][str(doc_id)] = grade
                evaluated_docs.append({
                    "doc_id": doc_id,
                    "title": doc_info["title"],
                    "grade": grade,
                    "is_relevant": grade >= 2,
                    "systems_retrieved": doc_info["systems"],
                    "rationale": rationale
                })
                # TREC standard line: qid 0 doc_id grade
                tsv_rows.append(f"{qid}\t0\t{doc_id}\t{grade}")

            # Compute IDCG@10
            idcg10 = calculate_idcg10(assigned_grades)
            qrels_data["metadata"]["idcg_10_per_query"][qid] = round(idcg10, 4)

            qrels_data["queries"][qid] = {
                "query_term": term,
                "category_id": q["category_id"],
                "total_candidates_pooled": len(pool),
                "relevant_count_grade_ge_2": sum(1 for g in assigned_grades if g >= 2),
                "idcg_10": round(idcg10, 4),
                "candidates": evaluated_docs
            }

            print(f"    -> Pool: {len(pool)} docs | Rel (>=2): {sum(1 for g in assigned_grades if g >= 2)} | IDCG@10: {idcg10:.4f}")

    # Write output files
    os.makedirs(os.path.dirname(output_graded_json), exist_ok=True)
    with open(output_graded_json, "w", encoding="utf-8") as f:
        json.dump(qrels_data, f, ensure_ascii=False, indent=2)

    with open(output_qrels_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(tsv_rows) + "\n")

    print(f"\n[SUCCESS] Ground truth pooling complete.")
    print(f"Graded JSON: {output_graded_json}")
    print(f"TREC TSV:    {output_qrels_tsv}")
    return qrels_data


if __name__ == "__main__":
    run_pooling_and_scoring()
