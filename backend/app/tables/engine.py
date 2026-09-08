"""
Pluggable TableStructureEngine (spec section 12). Tries, in order:

  1. PP-Structure (PaddleOCR's layout/table structure model) if the
     `paddleocr` package's PPStructure is importable and initializes
     successfully — highest-quality structure recovery.
  2. OpenCV grid-line detection — reliable for bordered tables, works fully
     offline with zero extra models.
  3. OCR-alignment fallback — for borderless tables, or if both above find
     nothing, so tabular data is still reconstructed as a real table rather
     than falling back to a plain image (spec section 38: never convert a
     table to an image if a real one can be built).

Regions already claimed by a higher-priority method are excluded from the
lower-priority passes (by bbox overlap) so tables are not double-detected.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult
from app.tables.models import DetectedTable
from app.tables.ocr_fallback_engine import detect_borderless_tables
from app.tables.opencv_engine import extract_table_structure, find_candidate_table_regions

logger = get_logger(__name__)

_pp_structure_engine = None
_pp_structure_checked = False


def _get_pp_structure():
    global _pp_structure_engine, _pp_structure_checked
    if _pp_structure_checked:
        return _pp_structure_engine
    _pp_structure_checked = True
    if not get_settings().enable_pp_structure:
        logger.info("pp_structure_disabled", message="ENABLE_PP_STRUCTURE is off; using OpenCV + OCR-alignment table engines only.")
        return None
    try:
        from paddleocr import PPStructure

        _pp_structure_engine = PPStructure(table=True, ocr=False, show_log=False)
    except Exception as exc:  # noqa: BLE001 - optional dependency, must not break the pipeline
        logger.info("pp_structure_unavailable", error=str(exc))
        _pp_structure_engine = None
    return _pp_structure_engine


def detect_tables(image: np.ndarray, words: list[OCRWordResult], page_width: int, page_height: int) -> list[DetectedTable]:
    tables: list[DetectedTable] = []
    claimed: list[BBox] = []

    pp_tables = _detect_with_pp_structure(image)
    for t in pp_tables:
        _fill_cell_text_from_words(t, words)
        tables.append(t)
        claimed.append(t.bbox)

    candidate_regions = find_candidate_table_regions(image)
    for region in candidate_regions:
        if any(_overlaps(region, c) for c in claimed):
            continue
        region_words = [w for w in words if _overlaps(w.bbox, region)]
        detected = extract_table_structure(image, region, region_words)
        if detected is not None and detected.n_rows >= 1 and detected.n_cols >= 1:
            tables.append(detected)
            claimed.append(region)

    remaining_words = [w for w in words if not any(_center_in(w.bbox, c) for c in claimed)]
    fallback_tables = detect_borderless_tables(remaining_words, page_width, page_height)
    for t in fallback_tables:
        if any(_overlaps(t.bbox, c) for c in claimed):
            continue
        tables.append(t)
        claimed.append(t.bbox)

    return tables


def _detect_with_pp_structure(image: np.ndarray) -> list[DetectedTable]:
    engine = _get_pp_structure()
    if engine is None:
        return []
    try:
        result = engine(image)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pp_structure_inference_failed", error=str(exc))
        return []

    from app.models.enums import TableDetectionMethod, TextAlign
    from app.tables.models import DetectedCell

    tables: list[DetectedTable] = []
    for region in result:
        if region.get("type") != "table":
            continue
        x1, y1, x2, y2 = region["bbox"]
        res = region.get("res", {}) if isinstance(region.get("res"), dict) else {}
        html = res.get("html", "")
        cell_bboxes = res.get("cell_bbox", [])
        if not cell_bboxes:
            continue

        grid_positions = _parse_html_table_grid(html, expected=len(cell_bboxes))
        max_row = 0
        max_col = 0
        cells = []
        for i, cb in enumerate(cell_bboxes):
            xs = cb[0::2]
            ys = cb[1::2]
            row, col, rowspan, colspan = grid_positions[i] if i < len(grid_positions) else (i, 0, 1, 1)
            max_row = max(max_row, row + rowspan - 1)
            max_col = max(max_col, col + colspan - 1)
            cells.append(
                DetectedCell(
                    row=row,
                    column=col,
                    rowspan=rowspan,
                    colspan=colspan,
                    bbox=BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
                    text="",  # filled from OCR words by the caller/pipeline via bbox containment
                    confidence=0.8,
                    align_h=TextAlign.CENTER if row == 0 else TextAlign.LEFT,
                    is_header=(row == 0),
                )
            )
        tables.append(
            DetectedTable(
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                n_rows=max_row + 1,
                n_cols=max_col + 1,
                confidence=0.8,
                detection_method=TableDetectionMethod.PADDLE_STRUCTURE,
                cells=cells,
                border_style={"style": "grid", "source": "pp_structure", "html": html},
            )
        )
    return tables


def _parse_html_table_grid(html: str, expected: int) -> list[tuple[int, int, int, int]]:
    """Parses PP-Structure's predicted `<table>` HTML to recover (row, col,
    rowspan, colspan) per `<td>`/`<th>` in document order, accounting for
    spans occupying cells in following rows/columns."""
    from html.parser import HTMLParser

    class _TableGridParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[list[tuple[int, int, int, int]]] = []
            self._cur_row: list[tuple[int, int, int, int]] | None = None
            self._occupied: set[tuple[int, int]] = set()
            self._attrs: dict = {}
            self._row_idx = -1

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self._row_idx += 1
                self._cur_row = []
            elif tag in ("td", "th"):
                self._attrs = dict(attrs)

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self._cur_row is not None:
                rowspan = int(self._attrs.get("rowspan", 1) or 1)
                colspan = int(self._attrs.get("colspan", 1) or 1)
                col = 0
                while (self._row_idx, col) in self._occupied:
                    col += 1
                for r in range(self._row_idx, self._row_idx + rowspan):
                    for c in range(col, col + colspan):
                        self._occupied.add((r, c))
                self._cur_row.append((self._row_idx, col, rowspan, colspan))
            elif tag == "tr" and self._cur_row is not None:
                self.rows.append(self._cur_row)
                self._cur_row = None

    parser = _TableGridParser()
    try:
        parser.feed(html or "")
    except Exception:  # noqa: BLE001
        return []

    flat = [cell for row in parser.rows for cell in row]
    return flat if len(flat) == expected else flat


def _fill_cell_text_from_words(table: DetectedTable, words: list[OCRWordResult]) -> None:
    for cell in table.cells:
        if cell.text:
            continue
        cell_words = [w for w in words if _center_in(w.bbox, cell.bbox)]
        cell_words.sort(key=lambda w: (round(w.bbox.y1 / 8), w.bbox.x1))
        cell.text = " ".join(w.text for w in cell_words).strip()
        if cell_words:
            cell.confidence = float(np.mean([w.confidence for w in cell_words]))


def _overlaps(a: BBox, b: BBox, min_iou: float = 0.3) -> bool:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return False
    a_area = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    return (inter / a_area) >= min_iou


def _center_in(bbox: BBox, container: BBox) -> bool:
    cx, cy = (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2
    return container.x1 <= cx <= container.x2 and container.y1 <= cy <= container.y2
