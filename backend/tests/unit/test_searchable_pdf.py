"""
Unit tests for `app.reconstruction.searchable_pdf` -- the invisible OCR
text layer that makes the "cleaned searchable PDF" download (spec section
14: "This is CRITICAL") actually searchable.
"""
import fitz
import numpy as np

from app.reconstruction.searchable_pdf import add_searchable_page
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult


def _word(text: str, x1: float, y1: float, x2: float, y2: float) -> OCRWordResult:
    return OCRWordResult(
        text=text, confidence=0.9, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=Polygon.from_xy_list([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
        page_number=1, block_id="b0", line_id="l0",
    )


def _blank_image(width=800, height=1200):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_searchable_page_text_is_extractable_for_tight_real_world_word_boxes():
    """Regression test for a confirmed real bug: real OCR word bounding
    boxes are tight around the glyphs, and the previous `insert_textbox`-
    based implementation's line-height reservation needed slightly more
    room than that -- on real production data every single word (37/37)
    silently failed to insert (a small negative "did not fit" remainder,
    no exception), producing a "searchable" PDF with a completely empty,
    unsearchable text layer despite correct underlying OCR."""
    words = [
        _word("and", 217.92 * 150 / 72, 136.32 * 150 / 72, 255.84 * 150 / 72, 156.96 * 150 / 72),
        _word("Statistics", 223.2 * 150 / 72, 6.24 * 150 / 72, 264.48 * 150 / 72, 18.24 * 150 / 72),
    ]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(), 150, words, None)

    text = doc[0].get_text()
    assert "and" in text
    assert "Statistics" in text


def test_searchable_page_hindi_text_is_extractable():
    from app.reconstruction.fonts import resolve_body_font_path

    words = [_word("कृषि", 100, 100, 200, 130)]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(), 150, words, resolve_body_font_path())

    text = doc[0].get_text()
    assert "कृषि" in text


def test_searchable_page_search_for_returns_a_hit_near_the_word_bbox():
    words = [_word("Rohtak", 200, 300, 320, 330)]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(), 150, words, None)

    hits = doc[0].search_for("Rohtak")
    assert len(hits) == 1


def test_searchable_page_skips_blank_words_without_error():
    words = [_word("   ", 10, 10, 50, 30), _word("Real", 60, 10, 100, 30)]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(), 150, words, None)  # must not raise
    assert "Real" in doc[0].get_text()


def test_searchable_page_embeds_the_page_image():
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(width=400, height=600), 150, [], None)
    assert len(doc[0].get_images()) == 1


def test_searchable_page_output_is_a4_portrait():
    """A4 normalization (spec: FINAL PDF PAGE NORMALIZATION) -- the output
    page must be true A4, not the original scan's dimensions."""
    words = [_word("text", 100, 100, 700, 1000)]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(width=823, height=1288), 150, words, None)

    page = doc[0]
    assert abs(page.rect.width - 595.2756) < 0.01
    assert abs(page.rect.height - 841.8898) < 0.01


def test_searchable_page_output_is_a4_landscape_for_wide_table_content():
    words = [_word(f"c{i}", 50 + i * 200, 500, 50 + i * 200 + 150, 560) for i in range(8)]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(width=2000, height=900), 150, words, None)

    page = doc[0]
    assert abs(page.rect.width - 841.8898) < 0.01
    assert abs(page.rect.height - 595.2756) < 0.01


def test_searchable_page_invisible_text_stays_aligned_with_repositioned_image():
    """After crop/scale/center, the invisible OCR text for a word must
    land within the A4 page bounds (never off-page) -- spec step 36's
    alignment validation, applied to the actual PDF-building function
    (not just the transform math in isolation)."""
    words = [_word("Rohtak", 300, 400, 420, 440), _word("45189", 300, 460, 400, 500)]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(width=1200, height=1600), 150, words, None)

    page = doc[0]
    for term in ("Rohtak", "45189"):
        hits = page.search_for(term)
        assert len(hits) == 1
        rect = hits[0]
        assert 0 <= rect.x0 <= page.rect.width
        assert 0 <= rect.y0 <= page.rect.height


def test_searchable_page_sparse_title_page_fills_the_page():
    """A title page with a valid (non-degenerate) but small content region
    must be scaled up to fill the A4 safe area, not sit tiny in the middle
    of mostly blank space -- confirmed as the wanted behavior against a
    real title page (2026-09-13): content should fill the page like the
    rest of the book's pages."""
    # Two normal-height (60px) single-line words, far apart vertically --
    # like a real title page's title line and a date/publisher line far
    # below it. Their UNION spans enough of the page to clear the 10%
    # min-content-fraction in both dimensions (a real, non-degenerate small
    # crop, not the separate "degenerate detection -> keep full page"
    # fallback path) while each individual word stays a realistic size.
    words = [_word("BOOK", 900, 1200, 1200, 1260), _word("TITLE", 900, 1790, 1200, 1850)]
    doc = fitz.open()
    add_searchable_page(doc, _blank_image(width=2400, height=3400), 300, words, None)

    hits = doc[0].search_for("BOOK")
    assert len(hits) == 1
    # Original physical size would be ~14.4pt (60px at 300dpi); the
    # content-aware crop here scales by several times to fill the page, so
    # the placed text must land meaningfully larger than that old,
    # un-scaled size.
    assert hits[0].height > 40
