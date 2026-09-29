"""
Heuristic bibliographic-metadata extraction for the library accession
register (spec: PDF -> extraction -> DB -> cumulative Master Excel).

Reads ONLY the already-built Document JSON (title-page layout blocks) and
OCR word language tags -- never re-runs OCR/layout/AI, and never invents a
value it can't support from the document's own content (spec section 14:
"do not hallucinate missing values"). A field that can't be confidently
read is left blank/null with `needs_review=True` and a human-readable note
explaining why, rather than guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.models.enums import LayoutBlockType
from app.schemas.document_json import DocumentBlockJSON, PageJSON
from app.schemas.ocr import OCRWordResult
from app.services.cover_reader import analyze_cover, page_lines_text

# Government/statistical scan titles in this project's real corpus land on
# page 1-3 (title page, sometimes a blank/publisher page in between) --
# scanning further risks picking up body-text headings unrelated to the
# book's own title/author/year.
_TITLE_PAGE_LIMIT = 3

# A plausible printed-publication year: 1800-2039. Wide enough to cover
# this project's real documents (1929-2015 seen so far) without accepting
# an arbitrary 4-digit number (a page number, a statistic) as a year.
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b(?:\s*[-–]\s*(\d{2,4}))?")

_LANGUAGE_LABELS = {"en": "ENGLISH", "hi": "HINDI", "num": "NUMERICAL"}

# A stamped/printed accession number: "ACC.NO. D-1087", "ACC NO-1087",
# "ACCESSION NO: 1087", etc. Checked against the FILENAME first (this
# project's real corpus consistently names files
# "<TITLE>..._ACC_NO_<N>.pdf" or similar, which is far more reliable than
# OCR of a small, sometimes faint/skewed library stamp), then against the
# cover's own OCR text as a fallback.
_ACCESSION_TOKEN_RE = re.compile(r"ACC(?:ESSION)?[._\s-]*NO\.?[._\s:-]*([A-Za-z]?[-\s]?\d{2,6})", re.IGNORECASE)


@dataclass
class ExtractedMetadata:
    book_name: str
    author: str | None
    publisher: str | None
    language: str | None
    year_of_publication: str | None
    accession_number: str | None = None
    needs_review: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def notes_text(self) -> str | None:
        return "; ".join(self.notes) if self.notes else None

    @property
    def creator(self) -> str | None:
        """Backward-compatible combined value: the DB-backed register's
        `AccessionRecord.creator` column and the "Organisation / Author
        Name" column in `master_register_exporter`'s detailed export both
        predate the author/publisher split and want a single string --
        personal author preferred, falling back to the publisher/issuing
        body."""
        return self.author or self.publisher


def extract_book_metadata(
    original_filename: str,
    pages: list[PageJSON],
    words_by_page: dict[int, list[OCRWordResult]] | None = None,
) -> ExtractedMetadata:
    title_pages = sorted(
        (p for p in pages if p.page_number <= _TITLE_PAGE_LIMIT), key=lambda p: p.page_number
    )

    notes: list[str] = []
    needs_review = False

    # Geometry-based cover reading first (rebuilds text lines from OCR
    # word boxes, so a title the layout detector shattered into one-word
    # blocks still comes out whole); the older block-type heuristics below
    # remain as the fallback for anything it can't confidently read.
    cover = analyze_cover(title_pages, words_by_page)

    book_name, title_block = _extract_book_name(title_pages)
    if cover.title:
        book_name = cover.title
    if book_name is None:
        book_name = original_filename
        needs_review = True
        notes.append("book_name: no TITLE/HEADING/paragraph text found on the first pages; used the filename")

    author = cover.author
    publisher = cover.publisher
    if author is None and publisher is None:
        # Older, block-type-based heuristic (not geometry-aware) -- its
        # own docstring shows it mainly targets publisher/issuing-body
        # style lines, so it fills the publisher slot when the cover
        # reader found neither.
        publisher = _extract_creator(title_pages, title_block)
    if author is None and publisher is None:
        needs_review = True
        notes.append("author/publisher: could not confidently identify an author or issuing-body line")

    year = cover.year or _extract_year(title_pages)
    if year is None:
        needs_review = True
        notes.append("year_of_publication: no plausible year found on the first pages")

    language = _majority_language(words_by_page) if words_by_page else None
    if language is None:
        needs_review = True
        notes.append("language: not enough OCR text to classify")

    accession_number = _extract_accession_number(original_filename, title_pages, words_by_page)
    if accession_number is None:
        # Not flagged for review -- an auto-generated sequential number is
        # an accepted, expected fallback here, not a data-quality problem;
        # this note just makes that traceable after the fact.
        notes.append("accession_number: not found in the filename or on the cover; auto-generated sequentially")

    return ExtractedMetadata(
        book_name=book_name, author=author, publisher=publisher, language=language, year_of_publication=year,
        accession_number=accession_number, needs_review=needs_review, notes=notes,
    )


def _extract_accession_number(original_filename: str, title_pages: list[PageJSON], words_by_page: dict[int, list[OCRWordResult]] | None) -> str | None:
    prefix = get_settings().accession_number_prefix

    match = _ACCESSION_TOKEN_RE.search(original_filename)
    if match:
        normalized = _normalize_accession_match(match.group(1), prefix)
        if normalized:
            return normalized

    combined = " ".join(page_lines_text(title_pages, words_by_page))
    match = _ACCESSION_TOKEN_RE.search(combined)
    if match:
        return _normalize_accession_match(match.group(1), prefix)

    return None


def _normalize_accession_match(raw: str, default_prefix: str) -> str | None:
    """"D-1087" / "D1087" -> "D-1087" (already has its own prefix letter);
    "1087" -> "{default_prefix}-1087" (bare number, needs the register's
    configured prefix)."""
    raw = raw.strip().upper().replace(" ", "")
    match = re.match(r"^([A-Z]+)-?(\d+)$", raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.match(r"^(\d+)$", raw)
    if match:
        return f"{default_prefix}-{match.group(1)}"
    return None


def _block_text(block: DocumentBlockJSON) -> str:
    return " ".join(run.text for run in block.content if run.text and run.text.strip()).strip()


def _extract_book_name(title_pages: list[PageJSON]) -> tuple[str | None, DocumentBlockJSON | None]:
    """Prefer an explicit TITLE block; fall back to the first HEADING; fall
    back to the longest paragraph on the first page that has any real
    text at all. Never fabricates a title -- returns None (caller
    supplies the filename fallback and flags for review) if nothing
    usable is found."""
    for wanted_type in (LayoutBlockType.TITLE, LayoutBlockType.HEADING):
        for page in title_pages:
            candidates = [b for b in page.sorted_blocks() if b.type == wanted_type and _block_text(b)]
            if candidates:
                best = max(candidates, key=lambda b: len(_block_text(b)))
                return _block_text(best), best

    for page in title_pages:
        paragraphs = [b for b in page.sorted_blocks() if b.type == LayoutBlockType.PARAGRAPH and _block_text(b)]
        if paragraphs:
            best = max(paragraphs, key=lambda b: len(_block_text(b)))
            text = _block_text(best)
            # A whole paragraph is not a title -- keep only its first line/
            # sentence-ish chunk rather than dumping a full block of body
            # text into the "book name" column.
            return text[:200], best

    return None, None


def _extract_creator(title_pages: list[PageJSON], title_block: DocumentBlockJSON | None) -> str | None:
    """Best-effort: a TITLE/HEADING block distinct from the chosen title,
    or a short, mostly-uppercase paragraph line on the title page (the
    "GOVERNMENT OF INDIA PLANNING COMMISSION" / "MINISTRY OF AGRICULTURE"
    style publisher/author line seen throughout this project's real
    corpus) -- never a guess dressed up as a name."""
    title_id = title_block.id if title_block else None

    for page in title_pages:
        for block in page.sorted_blocks():
            if block.id == title_id or block.type not in (LayoutBlockType.TITLE, LayoutBlockType.HEADING, LayoutBlockType.SUBHEADING):
                continue
            text = _block_text(block)
            if text:
                return text

    for page in title_pages:
        for block in page.sorted_blocks():
            if block.type != LayoutBlockType.PARAGRAPH:
                continue
            text = _block_text(block)
            if _looks_like_creator_line(text):
                return text

    return None


def _looks_like_creator_line(text: str) -> bool:
    if not text or not (3 <= len(text) <= 120):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    uppercase_fraction = sum(1 for c in letters if c.isupper()) / len(letters)
    # A short, mostly-uppercase line with no sentence-ending punctuation --
    # matches "GOVERNMENT OF INDIA PLANNING COMMISSION" style lines, not a
    # body-text sentence that happens to be short.
    return uppercase_fraction > 0.8 and not text.rstrip().endswith((".", ",", ";"))


def _extract_year(title_pages: list[PageJSON]) -> str | None:
    for page in title_pages:
        for block in page.sorted_blocks():
            text = _block_text(block)
            if not text:
                continue
            match = _YEAR_RE.search(text)
            if match:
                return match.group(0).strip()
    return None


def _majority_language(words_by_page: dict[int, list[OCRWordResult]]) -> str | None:
    counts = {"en": 0, "hi": 0, "num": 0}
    total_classifiable = 0
    for words in words_by_page.values():
        for w in words:
            lang = w.language
            if lang == "hi-en":
                counts["en"] += 1
                counts["hi"] += 1
                total_classifiable += 1
            elif lang in counts:
                counts[lang] += 1
                total_classifiable += 1

    if total_classifiable == 0:
        return None

    en_frac = counts["en"] / total_classifiable
    hi_frac = counts["hi"] / total_classifiable
    num_frac = counts["num"] / total_classifiable

    # Both scripts genuinely present in meaningful proportion -> mixed
    # document, matching the reference register's "HINDI/ENGLISH" value.
    if en_frac > 0.15 and hi_frac > 0.15:
        return "HINDI/ENGLISH"
    dominant = max(counts, key=lambda k: counts[k])
    if counts[dominant] == 0:
        return None
    return _LANGUAGE_LABELS[dominant]
