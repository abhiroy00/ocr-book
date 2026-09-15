"""
Tests for the library accession register's heuristic bibliographic-
metadata extraction (`app.services.accession_extractor`). Never invents a
value it can't support from the document's own content -- every "no
usable X found" path must leave that field blank/null and set
`needs_review`, not guess.
"""
from app.schemas.document_json import DocumentBlockJSON, PageJSON, TextRunJSON
from app.schemas.geometry import BBox, NormBBox
from app.schemas.ocr import OCRWordResult
from app.services.accession_extractor import extract_book_metadata

_BBOX = BBox(x1=0, y1=0, x2=100, y2=20)
_NORM_BBOX = NormBBox(x1=0.0, y1=0.0, x2=1.0, y2=0.1)


def _block(block_type, text, z_order=0) -> DocumentBlockJSON:
    return DocumentBlockJSON(
        id=f"blk-{z_order}-{text[:6]}", type=block_type, bbox=_BBOX, bbox_norm=_NORM_BBOX,
        confidence=0.9, z_order=z_order, content=[TextRunJSON(text=text)],
    )


def _page(page_number: int, blocks: list[DocumentBlockJSON]) -> PageJSON:
    return PageJSON(document_id="doc-1", page_id=f"pg-{page_number}", page_number=page_number, page_width=1000, page_height=1400, dpi=150, blocks=blocks)


def _word(text: str, language: str) -> OCRWordResult:
    from app.schemas.geometry import Polygon

    return OCRWordResult(
        text=text, confidence=0.9, bbox=_BBOX, polygon=Polygon.from_xy_list([[0, 0], [10, 0], [10, 10], [0, 10]]),
        page_number=1, block_id="b", line_id="l", language=language,
    )


def test_extracts_title_from_a_title_block():
    page = _page(1, [_block("title", "STATISTICAL ABSTRACT OF PUNJAB 2003", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [page])
    assert result.book_name == "STATISTICAL ABSTRACT OF PUNJAB 2003"


def test_falls_back_to_filename_when_no_text_blocks_at_all():
    page = _page(1, [])
    result = extract_book_metadata("STATISTICAL_ABSTRACT_1995.pdf", [page])
    assert result.book_name == "STATISTICAL_ABSTRACT_1995.pdf"
    assert result.needs_review is True
    assert any("book_name" in n for n in result.notes)


def test_extracts_creator_from_a_second_heading_block_distinct_from_the_title():
    page = _page(1, [
        _block("title", "STATISTICAL ABSTRACT OF PUNJAB 2003", z_order=0),
        _block("heading", "ECONOMIC AND STATISTICAL ORGANISATION PUNJAB", z_order=1),
    ])
    result = extract_book_metadata("fallback.pdf", [page])
    assert result.book_name == "STATISTICAL ABSTRACT OF PUNJAB 2003"
    assert result.creator == "ECONOMIC AND STATISTICAL ORGANISATION PUNJAB"


def test_creator_is_none_when_nothing_plausible_is_present():
    page = _page(1, [_block("title", "A Book With Only A Title", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [page])
    assert result.creator is None
    assert result.needs_review is True
    assert any("creator" in n for n in result.notes)


def test_extracts_a_plain_four_digit_year():
    page = _page(1, [_block("paragraph", "Published in 1998 by the Government of India", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [page])
    assert result.year_of_publication == "1998"


def test_extracts_a_year_range_verbatim():
    page = _page(1, [_block("paragraph", "AGRICULTURAL SITUATION IN INDIA APRIL 1995 MARCH 1996", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [page])
    # A bare "1995" and a bare "1996" both appear; the regex captures the
    # first plausible year token it finds, verbatim, never reformatted.
    assert result.year_of_publication is not None
    assert result.year_of_publication.startswith("1995")


def test_year_is_none_when_no_plausible_year_present():
    page = _page(1, [_block("title", "A Book With No Date Anywhere", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [page])
    assert result.year_of_publication is None
    assert result.needs_review is True


def test_never_hallucinates_a_year_from_an_implausible_number():
    """A page number or a large statistic must not be mistaken for a
    publication year."""
    page = _page(1, [_block("paragraph", "Total pages 9999, table reference 88888", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [page])
    assert result.year_of_publication is None


def test_majority_language_english():
    page = _page(1, [])
    words_by_page = {1: [_word("Statistical", "en"), _word("Abstract", "en"), _word("India", "en")]}
    result = extract_book_metadata("fallback.pdf", [page], words_by_page)
    assert result.language == "ENGLISH"


def test_majority_language_hindi():
    page = _page(1, [])
    words_by_page = {1: [_word("सांख्यिकी", "hi"), _word("रिपोर्ट", "hi")]}
    result = extract_book_metadata("fallback.pdf", [page], words_by_page)
    assert result.language == "HINDI"


def test_majority_language_mixed_hindi_english():
    page = _page(1, [])
    words_by_page = {1: [_word("Statistical", "en"), _word("Report", "en"), _word("सांख्यिकी", "hi"), _word("रिपोर्ट", "hi")]}
    result = extract_book_metadata("fallback.pdf", [page], words_by_page)
    assert result.language == "HINDI/ENGLISH"


def test_language_is_none_without_any_ocr_words():
    page = _page(1, [_block("title", "Some Title", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [page], {})
    assert result.language is None
    assert result.needs_review is True


def test_only_scans_the_first_few_pages_not_the_whole_document():
    """A year/title appearing deep in the body text (page 50) must not be
    picked up as the book's own title/year."""
    title_page = _page(1, [_block("title", "The Real Title", z_order=0)])
    deep_page = _page(50, [_block("title", "A Random Heading On Page 50 1975", z_order=0)])
    result = extract_book_metadata("fallback.pdf", [title_page, deep_page])
    assert result.book_name == "The Real Title"
