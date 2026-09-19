"""
Geometry-based reading of a book's front cover (title / creator / imprint
year) for the library accession register.

Why this exists: on old government-report scans the layout detector often
shatters one printed line into several one/two-word blocks ("Studies in" /
"the" / "Economics" / "of Farm" / "Management"), so "the longest single
block" yields fragments like "Studies in". The raw OCR word boxes, though,
are clean single-line boxes -- so text lines are rebuilt from their
positions, and the cover is then read the way a person reads it:

  - the biggest lettering is the title (plus the subtitle / period lines
    directly under it, e.g. "COMBINED REPORT", "1949-50 to 1966-67");
  - the lines under a lone "BY" are the personal authors; failing that, a
    run of issuing-body lines (Directorate / Ministry / Government ...)
    below the title is the creator;
  - a lone year near the bottom of the cover is the imprint year.

Everything returned is text that is actually printed on the cover -- when
nothing can be read confidently the field is None and the caller falls back
(and flags the record for review) rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from statistics import median

from app.models.enums import LayoutBlockType
from app.schemas.document_json import DocumentBlockJSON, PageJSON
from app.schemas.ocr import OCRWordResult

_TEXT_BLOCK_TYPES = {
    LayoutBlockType.TITLE, LayoutBlockType.HEADING, LayoutBlockType.SUBHEADING,
    LayoutBlockType.PARAGRAPH, LayoutBlockType.HEADER, LayoutBlockType.FOOTER,
}

# Words that mark an issuing body / publisher line rather than a title.
_ORG_KEYWORDS = (
    "DIRECTORATE", "MINISTRY", "DEPARTMENT", "DEPTT", "GOVERNMENT", "GOVT", "COMMISSION",
    "INSTITUTE", "ORGANISATION", "ORGANIZATION", "BUREAU", "COUNCIL", "BOARD", "SOCIETY",
    "UNIVERSITY", "ASSOCIATION", "OFFICE", "DIVISION", "CORPORATION", "PUBLISHED",
    "PUBLICATIONS", "CONTROLLER", "AUTHORITY", "COMMITTEE", "CENTRE", "FEDERATION",
)
_MIN_LEN_FOR_FUZZY = 6
_YEAR_TOKEN_RE = re.compile(r"^(1[89]\d{2}|20[0-3]\d)(?:[-–/](\d{2,4}))?$")
_DEVANAGARI_DIGITS = "०१२३४५६७८९"
_CONNECTORS = {"to", "and", "&", "-", "–", "—", "/"}

_TITLE_BODY_RATIO = 0.75   # a line this fraction of the biggest lettering is still "the title"
_SUBTITLE_RATIO = 0.5      # ...and this fraction is still a subtitle / period line
_MAX_SUBTITLE_LINES = 3
_MAX_AUTHORS = 6


@dataclass
class CoverInfo:
    """What could be read off the cover; any field may be None."""

    title: str | None = None
    creator: str | None = None
    year: str | None = None


@dataclass
class _Box:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def yc(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class _Line:
    boxes: list[_Box]

    @property
    def text(self) -> str:
        return _clean_tokens(" ".join(b.text for b in sorted(self.boxes, key=lambda b: b.x1)))

    @property
    def y1(self) -> float:
        return min(b.y1 for b in self.boxes)

    @property
    def y2(self) -> float:
        return max(b.y2 for b in self.boxes)

    @property
    def yc(self) -> float:
        return sum(b.yc for b in self.boxes) / len(self.boxes)

    @property
    def height(self) -> float:
        heights = [b.height for b in self.boxes if any(c.isalnum() for c in b.text)]
        return float(median(heights)) if heights else 0.0


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean_tokens(text: str) -> str:
    """Collapses whitespace and repairs/drops OCR junk tokens. The OCR
    reads a Latin "O" as the Devanagari zero "०" ("T०" for "TO"), so that
    one is mapped back; any other Devanagari digit, alone or mixed into
    Latin text, is a misread symbol (a "&" read as "८"), not content."""
    kept: list[str] = []
    for token in text.split():
        if any(c in _DEVANAGARI_DIGITS for c in token):
            letters = [c for c in token if c.isascii() and c.isalpha()]
            if letters and all(c in _DEVANAGARI_DIGITS or c.isascii() for c in token):
                replacement = "O" if all(c.isupper() for c in letters) else "o"
                token = token.replace("०", replacement)
                if any(c in _DEVANAGARI_DIGITS for c in token):
                    continue  # some other Devanagari digit mixed in -- unrecoverable
            elif len(token) == 1:
                continue  # a lone stray digit symbol; multi-digit Devanagari numbers ("१९९५") are real content
        kept.append(token)
    return " ".join(kept)


def _tidy(text: str, limit: int = 250) -> str:
    return text.strip(" \t,;:.-–—")[:limit].strip()


def _letters(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def _substantial(line: _Line) -> bool:
    tokens = line.text.split()
    return _letters(line.text) >= 4 and any(_letters(t) >= 3 for t in tokens)


def _clean_enough(line: _Line) -> bool:
    """False for OCR mush like "NO a ry Ta ble s" (letter-spaced italics
    read as fragments): most alphabetic tokens must be real-length words."""
    alpha_tokens = [
        t for t in line.text.split()
        if _letters(t) > 0 and not _YEAR_TOKEN_RE.match(t) and t.lower() not in _CONNECTORS
    ]
    if not alpha_tokens:
        return True
    return sum(1 for t in alpha_tokens if _letters(t) >= 3) / len(alpha_tokens) >= 0.6


def _is_org(line: _Line) -> bool:
    """Issuing-body line? Checks the space-stripped letters (OCR splits
    words: "DEP ARTMEN'T") and fuzzy-matches each token (OCR typos:
    "Direotorate")."""
    text = line.text.upper()
    squashed = "".join(c for c in text if c.isalpha())
    if any(k in squashed for k in _ORG_KEYWORDS):
        return True
    for token in text.split():
        word = "".join(c for c in token if c.isalpha())
        if len(word) >= _MIN_LEN_FOR_FUZZY and get_close_matches(word, _ORG_KEYWORDS, n=1, cutoff=0.8):
            return True
    return False


def _year_only(text: str) -> str | None:
    """A line that is nothing but year(s) ("1973", "1995 1996", "1949-50
    to 1966-67") -> the year text; otherwise None."""
    years: list[str] = []
    for token in text.split():
        if _YEAR_TOKEN_RE.match(token):
            years.append(token)
        elif token.lower() in _CONNECTORS or _letters(token) <= 1:
            continue
        else:
            return None
    if not years:
        return None
    if len(years) == 1:
        return years[0]
    if all("-" in y or "–" in y or "/" in y for y in years):
        return f"{years[0]} to {years[1]}"
    return f"{years[0]}-{years[1]}"


def _block_text(block: DocumentBlockJSON) -> str:
    return " ".join(run.text for run in block.content if run.text and run.text.strip()).strip()


# ---------------------------------------------------------------------------
# Lines from boxes
# ---------------------------------------------------------------------------

def _page_boxes(page: PageJSON, words: list[OCRWordResult] | None) -> list[_Box]:
    """OCR word boxes when available (clean single-line boxes); otherwise
    the layout's text blocks."""
    if words:
        return [
            _Box(w.text.strip(), w.bbox.x1, w.bbox.y1, w.bbox.x2, w.bbox.y2)
            for w in words if w.text and w.text.strip()
        ]
    boxes: list[_Box] = []
    for block in page.sorted_blocks():
        if block.type not in _TEXT_BLOCK_TYPES:
            continue
        text = _block_text(block)
        if text:
            boxes.append(_Box(text, block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2))
    return boxes


def _overlaps_horizontally(a: _Box, b: _Box) -> bool:
    """Genuinely-adjacent fragments of one printed line sit side by side;
    two boxes sharing most of their x-range are different lines (or
    duplicates), whatever their y-values say."""
    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    return overlap > 0.25 * max(1.0, min(a.width, b.width))


def _build_lines(boxes: list[_Box]) -> list[_Line]:
    """Clusters boxes into text lines by vertical centre (tolerance: about
    half the lettering height) and returns them top to bottom."""
    grouped: list[list[_Box]] = []
    for box in sorted(boxes, key=lambda b: (b.yc, b.x1)):
        best: list[_Box] | None = None
        best_delta = float("inf")
        for group in grouped[-4:]:
            line_yc = sum(b.yc for b in group) / len(group)
            line_h = float(median([b.height for b in group]))
            delta = abs(box.yc - line_yc)
            if delta <= 0.55 * max(1.0, min(box.height, line_h)) and delta < best_delta:
                if not any(_overlaps_horizontally(box, other) for other in group):
                    best, best_delta = group, delta
        if best is None:
            grouped.append([box])
        else:
            best.append(box)
    lines = [_Line(g) for g in grouped]
    lines.sort(key=lambda ln: ln.yc)
    return lines


# ---------------------------------------------------------------------------
# Title / creator / year
# ---------------------------------------------------------------------------

def _find_title(lines: list[_Line], page_height: float) -> tuple[str | None, int]:
    """Biggest lettering = the title. Returns (title text, index of its
    last line), or (None, -1)."""
    candidates = [
        i for i, ln in enumerate(lines)
        if _substantial(ln) and not _is_org(ln) and ln.y1 < 0.85 * page_height
    ]
    if not candidates:
        return None, -1

    biggest = max(lines[i].height for i in candidates)
    if biggest <= 0:
        return None, -1
    anchor = min(i for i in candidates if lines[i].height == biggest)

    def is_body(i: int) -> bool:
        ln = lines[i]
        return _substantial(ln) and not _is_org(ln) and ln.height >= _TITLE_BODY_RATIO * biggest

    def gap(above: int, below: int) -> float:
        return lines[below].y1 - lines[above].y2

    start = end = anchor
    while start - 1 >= 0 and is_body(start - 1) and gap(start - 1, start) <= 2 * biggest:
        start -= 1
    while end + 1 < len(lines) and is_body(end + 1) and gap(end, end + 1) <= 2 * biggest:
        end += 1

    # Subtitle / period lines directly under the title ("COMBINED REPORT",
    # "1949-50 to 1966-67") -- part of how the register names a book.
    added = 0
    while end + 1 < len(lines) and added < _MAX_SUBTITLE_LINES:
        nxt = lines[end + 1]
        if _is_org(nxt) or gap(end, end + 1) > 2.5 * biggest or nxt.height < _SUBTITLE_RATIO * biggest:
            break
        if not (_substantial(nxt) or _year_only(nxt.text)) or not _clean_enough(nxt):
            break
        end += 1
        added += 1

    title = _tidy(" ".join(lines[i].text for i in range(start, end + 1)))
    return (title or None), end


def _author_name(text: str) -> str | None:
    """"Mrs. Kusum Rathore, M.S. (IOWA)," -> "Mrs. Kusum Rathore" (degrees
    and affiliations follow the first comma)."""
    name = _tidy(text.split(",")[0])
    return name if _letters(name) >= 3 else None


def _find_creator(lines: list[_Line], title_end: int) -> str | None:
    # 1. Personal authors: the lines under a lone "BY".
    for i, ln in enumerate(lines):
        if not re.fullmatch(r"(?i)by[:.]?", ln.text.strip()):
            continue
        names: list[str] = []
        prev = ln
        for nxt in lines[i + 1:i + 9]:
            if nxt.y1 - prev.y2 > 3 * max(prev.height, nxt.height, 1.0):
                break
            text = nxt.text.strip()
            if re.fullmatch(r"(?i)and|&", text):
                prev = nxt
                continue
            if not _substantial(nxt) or _is_org(nxt):
                break
            name = _author_name(text)
            if name and name not in names:
                names.append(name)
            prev = nxt
        if names:
            return " / ".join(names[:_MAX_AUTHORS])

    # 2. Issuing body: the first run of org-like lines below the title (or
    # the first anywhere if no title was found), plus its obvious
    # continuation lines.
    i = title_end + 1
    while i < len(lines):
        if _is_org(lines[i]) and _substantial(lines[i]):
            group = [lines[i]]
            j = i + 1
            while j < len(lines):
                nxt, prev = lines[j], group[-1]
                if _year_only(nxt.text) or not _substantial(nxt):
                    break
                if nxt.y1 - prev.y2 > 1.5 * max(prev.height, nxt.height, 1.0):
                    break
                similar_size = 0.7 <= nxt.height / max(prev.height, 1.0) <= 1.4
                if not (_is_org(nxt) or similar_size):
                    break
                group.append(nxt)
                j += 1
            text = re.sub(r"(?i)^\s*published\s+by\s+", "", " ".join(ln.text for ln in group))
            return _tidy(text) or None
        i += 1
    return None


def _find_year(lines: list[_Line], title_end: int, page_height: float) -> str | None:
    """Imprint year: a lone year in the bottom of the cover, else a lone
    year line under the title (e.g. a "1995 - 1996" period line)."""
    bottom = [
        year for ln in lines
        if ln.y1 >= 0.6 * page_height and (year := _year_only(ln.text)) is not None
    ]
    if bottom:
        return bottom[-1]
    for ln in lines[title_end + 1:]:
        year = _year_only(ln.text)
        if year:
            return year
    return None


def analyze_cover(title_pages: list[PageJSON], words_by_page: dict[int, list[OCRWordResult]] | None = None) -> CoverInfo:
    """Reads title / creator / year off whichever of `title_pages` looks
    most like a front cover: the one whose biggest non-org lettering is
    largest relative to its page. Returns empty fields when nothing
    confident is found, leaving the caller's fallbacks in charge."""
    best: tuple[float, list[_Line], float] | None = None
    for page in title_pages:
        boxes = _page_boxes(page, (words_by_page or {}).get(page.page_number))
        if not boxes:
            continue
        lines = _build_lines(boxes)
        heights = [ln.height for ln in lines if _substantial(ln) and not _is_org(ln)]
        if not heights:
            continue
        page_height = float(page.page_height or 0) or max(b.y2 for b in boxes) or 1.0
        score = max(heights) / page_height
        if best is None or score > best[0] * 1.0001:
            best = (score, lines, page_height)

    if best is None:
        return CoverInfo()

    _, lines, page_height = best
    title, title_end = _find_title(lines, page_height)
    return CoverInfo(
        title=title,
        creator=_find_creator(lines, title_end),
        year=_find_year(lines, title_end, page_height),
    )
