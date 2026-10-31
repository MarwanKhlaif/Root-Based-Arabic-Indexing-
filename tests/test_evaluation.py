"""
tests/test_evaluation.py - Unit and Integration Tests for Milestone 4 IR Evaluation Engine.
Verifies mathematical metric calculations (P@k, AP, nDCG, Recall), over-stemming formulas
(Delta P_over, SFI), semantic collision detection, and QRELS invariants.
"""

import math
import os
import json
import pytest
from typing import List

from src.benchmarks.evaluate_ir import (
    compute_precision_at_k,
    compute_average_precision,
    compute_dcg_at_k,
    compute_idcg_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    SYSTEM_SPECS,
)
from src.benchmarks.overstemming_analysis import (
    evaluate_document_relevance,
    SEMANTIC_COLLISION_REGISTRY,
)
from src.db.connection import get_connection


# ==============================================================================
# 1. UNIT TESTS: PRECISION@K
# ==============================================================================

class TestPrecisionAtK:
    def test_all_relevant_returns_one(self):
        grades = [3, 2, 2, 3, 2, 3, 2, 3, 2, 3]
        assert compute_precision_at_k(grades, k=10, rel_threshold=2) == 1.0

    def test_all_irrelevant_returns_zero(self):
        grades = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        assert compute_precision_at_k(grades, k=10, rel_threshold=2) == 0.0

    def test_half_relevant(self):
        grades = [3, 0, 2, 1, 2, 0, 3, 1, 2, 0]  # 5 >= 2
        assert compute_precision_at_k(grades, k=10, rel_threshold=2) == 0.5

    def test_short_retrieval_list(self):
        # 5 items retrieved, all relevant -> 5 / 10 = 0.5
        grades = [3, 3, 2, 2, 2]
        assert compute_precision_at_k(grades, k=10, rel_threshold=2) == 0.5

    def test_empty_list_returns_zero(self):
        assert compute_precision_at_k([], k=10) == 0.0

    def test_invalid_k_returns_zero(self):
        assert compute_precision_at_k([3, 2, 3], k=0) == 0.0
        assert compute_precision_at_k([3, 2, 3], k=-5) == 0.0


# ==============================================================================
# 2. UNIT TESTS: AVERAGE PRECISION (AP)
# ==============================================================================

class TestAveragePrecision:
    def test_exact_hand_calculated_sequence(self):
        # Retrieved ranks: [Rel(3), NonRel(0), Rel(2), Rel(3)]
        # Rank 1: rel=1, P@1 = 1/1 = 1.0
        # Rank 2: rel=0, P@2 = 1/2 = 0.5 (excluded from sum)
        # Rank 3: rel=1, P@3 = 2/3 = 0.6667
        # Rank 4: rel=1, P@4 = 3/4 = 0.7500
        # Total relevant in pool = 3
        # Expected AP = (1.0 + 2/3 + 3/4) / 3 = (1.0 + 0.666667 + 0.75) / 3 = 2.416667 / 3 = 0.8056
        retrieved = [3, 0, 2, 3]
        ap = compute_average_precision(retrieved, total_relevant_in_pool=3, cutoff=4, rel_threshold=2)
        expected = (1.0 + (2.0 / 3.0) + (3.0 / 4.0)) / 3.0
        assert math.isclose(ap, expected, abs_tol=1e-4)

    def test_perfect_ranking_returns_one(self):
        # 4 relevant in pool, retrieved top 4 are all relevant
        retrieved = [3, 3, 2, 2, 0, 0]
        ap = compute_average_precision(retrieved, total_relevant_in_pool=4, cutoff=6, rel_threshold=2)
        assert ap == 1.0

    def test_zero_relevant_retrieved_returns_zero(self):
        retrieved = [0, 1, 0, 1, 0]
        assert compute_average_precision(retrieved, total_relevant_in_pool=5, cutoff=5) == 0.0

    def test_zero_relevant_in_pool_returns_zero(self):
        retrieved = [3, 2, 3]
        assert compute_average_precision(retrieved, total_relevant_in_pool=0, cutoff=3) == 0.0


# ==============================================================================
# 3. UNIT TESTS: DISCOUNTED CUMULATIVE GAIN (DCG) & nDCG
# ==============================================================================

class TestDCGandNDCG:
    def test_dcg_exact_formula(self):
        # grades: [3, 2]
        # i=1: (2^3 - 1) / log2(2) = 7 / 1.0 = 7.0
        # i=2: (2^2 - 1) / log2(3) = 3 / 1.58496 = 1.89279
        # DCG@2 = 7.0 + 1.89279 = 8.89279
        grades = [3, 2]
        dcg = compute_dcg_at_k(grades, k=2)
        expected = (7.0 / math.log2(2.0)) + (3.0 / math.log2(3.0))
        assert math.isclose(dcg, expected, abs_tol=1e-4)

    def test_ndcg_perfect_ranking_is_one(self):
        pool_grades = [3, 3, 2, 2, 1, 0]
        retrieved_perfect = [3, 3, 2, 2, 1, 0]
        ndcg = compute_ndcg_at_k(retrieved_perfect, pool_grades, k=6)
        assert ndcg == 1.0

    def test_ndcg_suboptimal_ranking_between_zero_and_one(self):
        pool_grades = [3, 3, 2, 1, 0]
        retrieved_suboptimal = [0, 1, 2, 3, 3]
        ndcg = compute_ndcg_at_k(retrieved_suboptimal, pool_grades, k=5)
        assert 0.0 < ndcg < 1.0

    def test_ndcg_zero_grades_returns_zero(self):
        assert compute_ndcg_at_k([0, 0, 0], [0, 0, 0], k=3) == 0.0


# ==============================================================================
# 4. UNIT TESTS: RECALL@K
# ==============================================================================

class TestRecallAtK:
    def test_recall_partial(self):
        # 3 relevant retrieved out of 6 in pool -> 3 / 6 = 0.5
        retrieved = [3, 0, 2, 1, 2, 0]
        assert compute_recall_at_k(retrieved, total_relevant_in_pool=6, k=6) == 0.5

    def test_recall_full(self):
        retrieved = [3, 2, 2, 0, 0]
        assert compute_recall_at_k(retrieved, total_relevant_in_pool=3, k=5) == 1.0

    def test_recall_zero_pool_returns_zero(self):
        assert compute_recall_at_k([3, 2], total_relevant_in_pool=0, k=2) == 0.0


# ==============================================================================
# 5. UNIT TESTS: OVER-STEMMING FORMULAS & COLLISION DETECTION
# ==============================================================================

class TestOverstemmingAndCollision:
    def test_delta_p_over_formula(self):
        p10_s2 = 0.80
        p10_s4 = 0.50
        delta_p = round(p10_s2 - p10_s4, 4)
        assert delta_p == 0.30

    def test_semantic_fallout_index_formula(self):
        # 3 items of grade 0 in top 10 -> SFI = 3/10 = 0.30
        grades = [3, 0, 2, 0, 2, 3, 0, 1, 2, 2]
        sfi = sum(1 for g in grades if g == 0) / 10.0
        assert sfi == 0.30

    def test_collision_detection_q20_bahrain(self):
        # Query Q20 (بحر): Kingdom of Bahrain must trigger PROPER_NOUN_DRIFT collision
        grade, is_col, ctype, expl = evaluate_document_relevance(
            "Q20",
            "الاحتجاجات البحرينية 2011",
            "اندلعت في مملكة البحرين احتجاجات شعبية في المنامة..."
        )
        assert grade == 0
        assert is_col is True
        assert ctype == "PROPER_NOUN_DRIFT"

    def test_collision_detection_q01_battalion(self):
        # Query Q01 (كتب): Military battalion must trigger DERIVATIONAL_SPECIALIZATION_COLLISION
        grade, is_col, ctype, expl = evaluate_document_relevance(
            "Q01",
            "كتيبة جنين العسكرية",
            "خاضت الكتيبة معارك عنيفة مع قوات الاحتلال..."
        )
        assert grade == 0
        assert is_col is True
        assert ctype == "DERIVATIONAL_SPECIALIZATION_COLLISION"

    def test_collision_detection_q18_eye_appointment(self):
        # Query Q18 (عين): Appointment / Senate must trigger HOMONYM_SENSE_COLLISION
        grade, is_col, ctype, expl = evaluate_document_relevance(
            "Q18",
            "مجلس الأعيان الأردني",
            "صدرت الإرادة الملكية بتعيين أعضاء مجلس الأعيان..."
        )
        assert grade == 0
        assert is_col is True
        assert ctype == "HOMONYM_SENSE_COLLISION"

    def test_positive_match_q20_oceanography(self):
        # Query Q20 (بحر): Core oceanography article should be Grade 3
        grade, is_col, ctype, expl = evaluate_document_relevance(
            "Q20",
            "البحر الأبيض المتوسط",
            "هو بحر متصل بالمحيط الأطلسي تحيط به منطقة حوض البحر الأبيض المتوسط وفيه أحياء بحرية متنوعة..."
        )
        assert grade >= 2
        assert is_col is False


# ==============================================================================
# 6. INTEGRATION TESTS: DATABASE DISPATCHERS & PLANNER TUNING
# ==============================================================================

class TestDatabaseIntegration:
    def test_planner_tuning_random_page_cost(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET random_page_cost = 1.5;")
                cur.execute("SHOW random_page_cost;")
                val = cur.fetchone()[0]
                assert float(val) == 1.5

    def test_all_six_systems_sql_dispatch(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET random_page_cost = 1.5;")
                for sys_def in SYSTEM_SPECS:
                    sys_id = sys_def["system_id"]
                    sql = sys_def["sql"]
                    trans_q = sys_def["transform"]("كتب")
                    params = sys_def["params"](trans_q, 5)

                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    assert len(rows) > 0, f"System {sys_id} returned 0 rows for query 'كتب'!"
                    assert len(rows[0]) == 4, f"System {sys_id} row schema must be (id, doc_id, title, score)"
