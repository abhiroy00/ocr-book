"""
Tests for `app.utils.page_normalize` -- A4 content-aware crop/scale/center
normalization. The "no content loss" property (spec step 39) is verified
directly: `compute_content_bbox` is built to always contain every OCR
word by construction, so these tests check that guarantee holds, not just
that it happens to hold for one example.
"""
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.utils.page_normalize import A4_LANDSCAPE_PT, A4_PORTRAIT_PT, compute_content_bbox, compute_page_transform


def _word(text, x1, y1, x2, y2) -> OCRWordResult:
    return OCRWordResult(
        text=text, confidence=0.9, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=Polygon.from_xy_list([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
        page_number=1, block_id="b", line_id="l",
    )


def test_compute_content_bbox_contains_every_word():
    words = [_word("a", 100, 100, 150, 130), _word("b", 900, 1200, 950, 1230), _word("c", 400, 50, 440, 80)]
    x1, y1, x2, y2 = compute_content_bbox(1000, 1400, words, margin_px=10)
    for w in words:
        assert x1 <= w.bbox.x1 and w.bbox.x2 <= x2
        assert y1 <= w.bbox.y1 and w.bbox.y2 <= y2


def test_compute_content_bbox_falls_back_to_full_page_when_no_words():
    x1, y1, x2, y2 = compute_content_bbox(800, 1200, [], margin_px=10)
    assert (x1, y1, x2, y2) == (0, 0, 800, 1200)


def test_compute_content_bbox_falls_back_to_full_page_for_degenerate_detection():
    """A single tiny stray word (e.g. a misdetection) must not produce an
    aggressive, untrustworthy crop -- the full page is kept instead."""
    words = [_word("x", 400, 600, 410, 610)]
    x1, y1, x2, y2 = compute_content_bbox(800, 1200, words, margin_px=10)
    assert (x1, y1, x2, y2) == (0, 0, 800, 1200)


def test_page_transform_produces_exact_a4_portrait_dimensions():
    words = [_word("text", 100, 100, 700, 1000)]
    transform = compute_page_transform(823, 1288, words, dpi=150)
    assert (transform.page_width_pt, transform.page_height_pt) == A4_PORTRAIT_PT


def test_page_transform_detects_genuine_landscape_content():
    # A wide table spanning almost the full width but only a small height band.
    words = [_word(f"c{i}", 50 + i * 200, 500, 50 + i * 200 + 150, 560) for i in range(8)]
    transform = compute_page_transform(2000, 900, words, dpi=150)
    assert (transform.page_width_pt, transform.page_height_pt) == A4_LANDSCAPE_PT
    assert transform.is_landscape


def test_page_transform_keeps_portrait_for_mild_width_bias():
    """Content only slightly wider than tall must NOT flip to landscape
    (spec: robust threshold, not "slightly wider than tall")."""
    words = [_word("text", 50, 50, 1050, 900)]  # ~1000x850, ratio 1.18
    transform = compute_page_transform(1100, 1000, words, dpi=150)
    assert not transform.is_landscape


def test_page_transform_never_upscales_sparse_content():
    """A title page with a small amount of content must not be blown up
    to fill the entire A4 safe area (spec step 26)."""
    words = [_word("Title", 100, 100, 300, 140)]
    transform = compute_page_transform(2000, 3000, words, dpi=300)
    assert transform.scale <= 1.0


def test_page_transform_shrinks_oversized_content_to_fit():
    words = [_word("big", 0, 0, 4000, 5000)]
    transform = compute_page_transform(4000, 5000, words, dpi=300)
    assert transform.scale < 1.0
    scaled_w = (transform.crop_x2 - transform.crop_x1) * (72.0 / 300) * transform.scale
    scaled_h = (transform.crop_y2 - transform.crop_y1) * (72.0 / 300) * transform.scale
    assert scaled_w <= transform.page_width_pt + 1e-6
    assert scaled_h <= transform.page_height_pt + 1e-6


def test_transform_bbox_keeps_every_word_within_the_a4_page_bounds():
    """Direct check of the "no content loss / no out-of-page placement"
    property across a realistic multi-word page."""
    words = [_word(f"w{i}", 50 + i * 60, 100 + (i % 5) * 200, 50 + i * 60 + 50, 100 + (i % 5) * 200 + 30) for i in range(20)]
    transform = compute_page_transform(1200, 1600, words, dpi=150)
    for w in words:
        placed = transform.transform_bbox(w.bbox)
        assert -0.01 <= placed.x1 <= transform.page_width_pt + 0.01
        assert -0.01 <= placed.x2 <= transform.page_width_pt + 0.01
        assert -0.01 <= placed.y1 <= transform.page_height_pt + 0.01
        assert -0.01 <= placed.y2 <= transform.page_height_pt + 0.01


def test_transform_bbox_preserves_relative_word_order():
    """Two words side by side in the original must remain side by side
    (in the same left-to-right order) after transformation -- normalization
    must not reorder or flip content."""
    left = _word("left", 100, 500, 200, 540)
    right = _word("right", 600, 500, 700, 540)
    transform = compute_page_transform(1200, 1600, [left, right], dpi=150)
    placed_left = transform.transform_bbox(left.bbox)
    placed_right = transform.transform_bbox(right.bbox)
    assert placed_left.x1 < placed_right.x1
