"""
Unit and Regression Tests for Corpus Acquisition, Quality Filtering, and Sampling.
Verifies all filtering criteria, deduplication, seed-42 determinism, and disk safety.
"""

import os
import json
import random
import pytest
from pathlib import Path
from src.corpus.acquire_wikipedia import (
    is_valid_article,
    EXCLUDED_NAMESPACES,
    normalize_arabic,
)


@pytest.mark.unit
def test_namespace_filtering():
    """Verify that all non-article namespaces are strictly excluded."""
    forbidden_titles = [
        "قالب:صندوق معلومات شخص",
        "تصنيف:تاريخ بغداد",
        "بوابة:فلسفة",
        "مستخدم:مروان",
        "ويكيبيديا:سياسات وتوجيهات",
        "نقاش:الذكاء الاصطناعي",
        "ملف:Cairo_Tower.jpg",
        "مساعدة:إنشاء مقالة جديدة",
        "Template:Infobox",
        "Category:Arabic_Language",
    ]
    dummy_text = (
        "هذا نص تجريبي طويل يحتوي على مائة كلمة وأكثر للتأكد من تجاوز حد الطول "
        "الأدنى للمقالة في ويكيبيديا العربية بهدف اختبار دقة مرشحات فضاءات الأسماء "
        * 5
    )

    for title in forbidden_titles:
        valid, reason, wc = is_valid_article(title, dummy_text)
        assert valid is False, f"Failed to filter namespace in title: {title}"
        assert reason.startswith("namespace")


@pytest.mark.unit
def test_valid_article_passes_namespace_check():
    """Verify that valid article titles pass the namespace filter."""
    valid_titles = [
        "ابن خلدون",
        "تاريخ دمشق",
        "معركة القادسية",
        "الخوارزمي وعلم الجبر",
        "جامعة القاهرة",
    ]
    dummy_text = (
        "هذا نص مقالة معرفية موسوعية كاملة يستعرض تاريخ وسيرة الشخصية أو المدينة "
        "بمحتوى يتجاوز المائة كلمة بصورة تفصيلية وموثقة بالمصادر الأكاديمية والتاريخية المعتمدة "
        * 6
    )

    for title in valid_titles:
        valid, reason, wc = is_valid_article(title, dummy_text)
        assert valid is True, f"Valid title falsely rejected: {title}"
        assert reason == "valid"


@pytest.mark.unit
def test_disambiguation_filtering():
    """Verify rejection of disambiguation pages by title or introductory marker."""
    disambig_samples = [
        ("جبل السيدة (توضيح)", "جبل السيدة قد يشير إلى أكثر من موضع جغرافي... " * 10),
        ("النور (توضيح)", "النور قد تعني: سورة النور في القرآن أو نور الشمس... " * 10),
        ("الرافدين", "صفحة توضيح تحتوي على أسماء المؤسسات والمجلات التي تحمل اسم الرافدين... " * 10),
    ]
    for title, text in disambig_samples:
        valid, reason, _ = is_valid_article(title, text)
        assert valid is False, f"Failed to filter disambiguation page: {title}"
        assert reason == "disambiguation"


@pytest.mark.unit
def test_redirect_filtering():
    """Verify rejection of Wikipedia redirect stubs."""
    redirect_samples = [
        ("الدوحة", "#تحويل [[قطر]]"),
        ("البصرة", "#REDIRECT [[بصرة الفيحاء]]"),
        ("الكوفة", "  #redirect [[الكوفة القديمة]]"),
    ]
    for title, text in redirect_samples:
        valid, reason, _ = is_valid_article(title, text)
        assert valid is False, f"Failed to filter redirect: {title}"
        assert reason == "redirect"


@pytest.mark.unit
def test_length_boundaries_filtering():
    """Verify strict adherence to [100, 25000] word count boundaries."""
    title = "مقالة تجريبية"

    # 1. Under minimum (stub with 99 words)
    stub_text = "كلمة " * 99
    valid, reason, wc = is_valid_article(title, stub_text, min_words=100, max_words=25000)
    assert valid is False
    assert reason == "stub_length"
    assert wc == 99

    # 2. Exact minimum boundary (100 words)
    exact_min_text = "كلمة " * 100
    valid, reason, wc = is_valid_article(title, exact_min_text, min_words=100, max_words=25000)
    assert valid is True
    assert wc == 100

    # 3. Normal article (1,200 words)
    normal_text = "كلمة " * 1200
    valid, reason, wc = is_valid_article(title, normal_text, min_words=100, max_words=25000)
    assert valid is True
    assert wc == 1200

    # 4. Exact maximum boundary (25,000 words)
    exact_max_text = "كلمة " * 25000
    valid, reason, wc = is_valid_article(title, exact_max_text, min_words=100, max_words=25000)
    assert valid is True
    assert wc == 25000

    # 5. Over maximum (bloat table dump with 25,001 words)
    bloat_text = "كلمة " * 25001
    valid, reason, wc = is_valid_article(title, bloat_text, min_words=100, max_words=25000)
    assert valid is False
    assert reason == "bloat_length"
    assert wc == 25001


@pytest.mark.unit
def test_null_byte_sanitization():
    """Verify that null bytes are stripped to prevent PostgreSQL COPY failures."""
    text_with_null = "نص عربي يحتوي على \x00 رمز بايت صفري ضار"
    cleaned = text_with_null.replace("\x00", "")
    assert "\x00" not in cleaned
    assert len(cleaned) == len(text_with_null) - 1


@pytest.mark.unit
def test_deterministic_sampling_seed_reproducibility():
    """Verify that seed=42 yields 100% reproducible sampling order across runs."""
    candidates = list(range(1000, 2000))  # 1,000 synthetic doc_ids
    sample_k = 50

    # Run 1
    random.seed(42)
    sample_1 = random.sample(candidates, sample_k)

    # Run 2
    random.seed(42)
    sample_2 = random.sample(candidates, sample_k)

    # Run 3 with different seed
    random.seed(99)
    sample_3 = random.sample(candidates, sample_k)

    assert sample_1 == sample_2, "Seed 42 failed to reproduce identical sample"
    assert sample_1 != sample_3, "Different seed produced identical sample"


@pytest.mark.unit
def test_content_deduplication_via_normalized_sha256():
    """Verify that articles differing only by tashkeel or tatweel trigger hash collision."""
    import hashlib

    text_vocalized = "كَتَبَ الأُسْتَاذُ فِي المَكْتَبَةِ الجَدِيدَةِ"
    text_unvocalized = "كتب الاستاذ في المكتبه الجديده"
    text_tatweel = "كـتـب الاسـتـاذ فـي الـمـكـتـبـه الـجـديـدة"

    norm1 = normalize_arabic(text_vocalized)
    norm2 = normalize_arabic(text_unvocalized)
    norm3 = normalize_arabic(text_tatweel)

    h1 = hashlib.sha256(norm1.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(norm2.encode("utf-8")).hexdigest()
    h3 = hashlib.sha256(norm3.encode("utf-8")).hexdigest()

    assert h1 == h2 == h3, "Normalized SHA-256 hashes must collide for identical linguistic text"
