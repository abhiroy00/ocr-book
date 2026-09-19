"""
Tests for the geometry-based cover reader
(`app.services.cover_reader`, used by `app.services.accession_extractor`).

The three fixtures under `tests/fixtures/covers/` are the real OCR word
boxes of the front covers of three books scanned on the live server (old
government statistical reports -- the corpus this project targets). They
pin the behavior that motivated the reader: the layout detector shattered
these titles into one-word blocks, so the register used to get fragments
like "Studies in" / "the" / "Production" for Book Name / Creator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.document_json import PageJSON
from app.schemas.geometry import BBox, Polygon
from app.schemas.ocr import OCRWordResult
from app.services.accession_extractor import extract_book_metadata
from app.services.cover_reader import analyze_cover

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "covers"


def _load(name: str):
    """(pages, words_by_page) for a fixture -- earlier pages are blank, as
    they are for the two books whose page 1 has no text."""
    data = json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    cover_page = data["page_number"]
    pages = [
        PageJSON(
            document_id="d", page_id=f"p{n}", page_number=n, page_width=data["page_width"],
            page_height=data["page_height"], dpi=150, blocks=[],
        )
        for n in range(1, cover_page + 1)
    ]
    words = [
        OCRWordResult(
            text=text, confidence=0.9, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            polygon=Polygon.from_xy_list([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
            page_number=cover_page, block_id="b", line_id="l", language=lang,
        )
        for text, x1, y1, x2, y2, lang in data["boxes"]
    ]
    return pages, {cover_page: words}


def test_title_shattered_across_word_blocks_is_read_whole():
    pages, words = _load("farm_management_pali_1973")
    info = analyze_cover(pages, words)
    # Previously "Studies in" -- the OCR misses "in Pali (", but the line is otherwise whole.
    assert info.title == "Studies in the Economics of Farm Management Rajasthan COMBINED REPORT FOR THE YEAR 1962-53 TO 1964-55"


def test_personal_authors_under_by_are_the_creator_without_degrees():
    pages, words = _load("farm_management_pali_1973")
    # Previously "the".
    assert analyze_cover(pages, words).creator == "Mrs. Kusum Rathore / Bhupal Singh Rathore / Dr. Ram K. Patel"


def test_imprint_year_at_the_bottom_beats_the_period_printed_in_the_title():
    pages, words = _load("farm_management_pali_1973")
    # The cover prints "1962-63 to 1964-65" mid-page and "1973" at the foot.
    assert analyze_cover(pages, words).year == "1973"


def test_multiline_title_with_period_line_and_imprint_year():
    pages, words = _load("area_production_yield_1967")
    info = analyze_cover(pages, words)
    assert info.title == "Area, Production and Yield Principal Crops In India 1949-50 to 1956-57"
    assert info.year == "1967"


def test_issuing_body_is_the_creator_when_there_are_no_personal_authors():
    pages, words = _load("area_production_yield_1967")
    creator = analyze_cover(pages, words).creator
    assert creator is not None
    assert creator.startswith("DIRECTORATE")
    assert "MINISTRY" in creator and creator.endswith("GOVERNMENT OF INDIA")


def test_cover_is_found_on_page_two_when_page_one_has_no_text():
    pages, words = _load("statistical_abstract_mizoram_1995")
    info = analyze_cover(pages, words)
    assert info.title == "STATISTICAL ABSTRACT"  # not the small stamp text or the department line
    assert info.creator == "DEP ARTMEN'T OF AGRICULTURR MINOR IRRIGATION MIZORAM"  # OCR's own spelling, stray "&" glyph dropped
    assert info.year == "1995-1996"


def test_extract_book_metadata_uses_the_cover_reader_end_to_end():
    pages, words = _load("farm_management_pali_1973")
    metadata = extract_book_metadata("Studies_in_the_Eco.pdf", pages, words)
    assert metadata.book_name.startswith("Studies in the Economics of Farm Management")
    assert metadata.creator == "Mrs. Kusum Rathore / Bhupal Singh Rathore / Dr. Ram K. Patel"
    assert metadata.year_of_publication == "1973"
    assert metadata.needs_review is False


def test_nothing_readable_leaves_fields_empty_for_the_fallbacks():
    page = PageJSON(document_id="d", page_id="p1", page_number=1, page_width=1000, page_height=1400, dpi=150, blocks=[])
    info = analyze_cover([page], {1: []})
    assert (info.title, info.creator, info.year) == (None, None, None)


@pytest.mark.parametrize("text,expected", [
    ("1973", "1973"),
    ("1995 1996", "1995-1996"),
    ("1949-50 to 1966-67", "1949-50 to 1966-67"),
    ("PRICE 1973", None),
    ("Rs. 25", None),
])
def test_year_only_lines(text, expected):
    from app.services.cover_reader import _year_only

    assert _year_only(text) == expected
