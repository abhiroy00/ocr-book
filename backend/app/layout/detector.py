"""
Layout detection orchestrator (spec section 11).

Classifies every region of a page into: title / heading / subheading /
paragraph / table / image / chart / signature / stamp / handwritten /
header / footer / page_number / footnote / horizontal_line / vertical_line
— with a stacking `z_order` that preserves reading order for reconstruction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.layout.graphics import detect_graphic_regions
from app.layout.lines import detect_lines
from app.models.enums import LayoutBlockType
from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult
from app.tables.models import DetectedTable

_PAGE_NUMBER_RE = re.compile(r"^[\-–—]?\s*(\d{1,4}|[०-९]{1,4})\s*[\-–—]?$|^(page|पृष्ठ|पृ\.?)\s*\.?\s*\d{1,4}$", re.IGNORECASE)

TOP_MARGIN_RATIO = 0.08
BOTTOM_MARGIN_RATIO = 0.90
FOOTNOTE_BAND_RATIO = 0.82


@dataclass
class TextBlockCandidate:
    words: list[OCRWordResult]
    bbox: BBox
    font_size: float


@dataclass
class LayoutBlockResult:
    block_type: LayoutBlockType
    bbox: BBox
    confidence: float
    z_order: int = 0
    word_ids: list[str] = field(default_factory=list)  # "block_id:line_id" refs into OCRBlock rows
    text: str = ""
    style: dict = field(default_factory=dict)
    table_ref: int | None = None  # index into the `tables` list passed to detect()
    image_ref: str | None = None  # storage path, set by the pipeline after cropping graphic blocks
    db_id: str | None = None  # set by document_service once the LayoutBlock row exists, so Document JSON block ids match the DB row


class LayoutDetector:
    def detect(
        self,
        processed_image: np.ndarray,
        original_image: np.ndarray,
        words: list[OCRWordResult],
        tables: list[DetectedTable],
        page_width: int,
        page_height: int,
    ) -> list[LayoutBlockResult]:
        table_boxes = [t.bbox for t in tables]
        free_words = [w for w in words if not any(_center_in(w.bbox, tb) for tb in table_boxes)]

        line_groups = _group_into_lines(free_words)
        text_blocks = _group_lines_into_blocks(line_groups)

        median_font = _median_font_size(free_words)

        results: list[LayoutBlockResult] = []
        for block in text_blocks:
            block_type, confidence = _classify_text_block(block, median_font, page_width, page_height)
            text = " ".join(w.text for w in block.words if w.text).strip()
            # Reconstruct approximate line breaks for multi-line paragraphs.
            text = _join_with_linebreaks(block.words)
            results.append(
                LayoutBlockResult(
                    block_type=block_type,
                    bbox=block.bbox,
                    confidence=confidence,
                    word_ids=[f"{w.block_id}:{w.line_id}" for w in block.words],
                    text=text,
                    style={
                        "font_size": round(block.font_size, 1),
                        "alignment": _infer_alignment(block, page_width),
                    },
                )
            )

        for idx, table in enumerate(tables):
            results.append(
                LayoutBlockResult(
                    block_type=LayoutBlockType.TABLE,
                    bbox=table.bbox,
                    confidence=table.confidence,
                    table_ref=idx,
                )
            )

        excluded = table_boxes + [b.bbox for b in results if b.block_type != LayoutBlockType.TABLE]
        for line in detect_lines(processed_image):
            block_type = LayoutBlockType.HLINE if line.orientation == "horizontal" else LayoutBlockType.VLINE
            if any(_overlap_ratio(line.bbox, tb) > 0.3 for tb in table_boxes):
                continue  # part of a table's own grid, not a standalone rule
            results.append(LayoutBlockResult(block_type=block_type, bbox=line.bbox, confidence=0.7))

        graphic_regions = detect_graphic_regions(original_image, words, excluded, page_width, page_height)
        for region in graphic_regions:
            results.append(
                LayoutBlockResult(block_type=region.block_type, bbox=region.bbox, confidence=region.confidence)
            )

        results.sort(key=lambda r: (r.bbox.y1, r.bbox.x1))
        for i, r in enumerate(results):
            r.z_order = i

        return results


# ----------------------------------------------------------------------------
# Text grouping
# ----------------------------------------------------------------------------

def _group_into_lines(words: list[OCRWordResult]) -> list[list[OCRWordResult]]:
    by_line: dict[tuple[str, str], list[OCRWordResult]] = {}
    for w in words:
        by_line.setdefault((w.block_id, w.line_id), []).append(w)
    lines = list(by_line.values())
    for line in lines:
        line.sort(key=lambda w: w.bbox.x1)
    lines.sort(key=lambda line: min(w.bbox.y1 for w in line))
    return lines


def _group_lines_into_blocks(lines: list[list[OCRWordResult]]) -> list[TextBlockCandidate]:
    blocks: list[list[list[OCRWordResult]]] = []
    for line in lines:
        line_bbox = _line_bbox(line)
        line_height = line_bbox.height or 12.0
        placed = False
        if blocks:
            last_block = blocks[-1]
            last_line_bbox = _line_bbox(last_block[-1])
            gap = line_bbox.y1 - last_line_bbox.y2
            x_overlap = _x_overlap_ratio(line_bbox, last_line_bbox)
            if gap <= line_height * 0.9 and x_overlap > 0.15:
                last_block.append(line)
                placed = True
        if not placed:
            blocks.append([line])

    candidates: list[TextBlockCandidate] = []
    for block_lines in blocks:
        words = [w for line in block_lines for w in line]
        xs1 = [w.bbox.x1 for w in words]
        ys1 = [w.bbox.y1 for w in words]
        xs2 = [w.bbox.x2 for w in words]
        ys2 = [w.bbox.y2 for w in words]
        bbox = BBox(x1=min(xs1), y1=min(ys1), x2=max(xs2), y2=max(ys2))
        sizes = [w.font_size_estimate for w in words if w.font_size_estimate]
        font_size = float(np.median(sizes)) if sizes else 10.0
        candidates.append(TextBlockCandidate(words=words, bbox=bbox, font_size=font_size))
    return candidates


def _join_with_linebreaks(words: list[OCRWordResult]) -> str:
    by_line: dict[str, list[OCRWordResult]] = {}
    for w in words:
        by_line.setdefault(w.line_id, []).append(w)
    ordered_lines = sorted(by_line.values(), key=lambda ws: min(w.bbox.y1 for w in ws))
    return "\n".join(" ".join(w.text for w in sorted(ws, key=lambda w: w.bbox.x1)) for ws in ordered_lines)


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

def _median_font_size(words: list[OCRWordResult]) -> float:
    sizes = [w.font_size_estimate for w in words if w.font_size_estimate]
    return float(np.median(sizes)) if sizes else 10.0


def _classify_text_block(
    block: TextBlockCandidate, median_font: float, page_width: int, page_height: int
) -> tuple[LayoutBlockType, float]:
    text = " ".join(w.text for w in block.words).strip()
    n_lines = len({w.line_id for w in block.words})
    rel_y1 = block.bbox.y1 / max(1, page_height)
    rel_y2 = block.bbox.y2 / max(1, page_height)
    rel_width = block.bbox.width / max(1, page_width)

    is_mostly_numeric = bool(re.sub(r"[\s\-–—.]", "", text)) and _digit_ratio(text) > 0.7

    if n_lines == 1 and (_PAGE_NUMBER_RE.match(text.strip()) or (is_mostly_numeric and len(text) <= 6)):
        if rel_y1 < TOP_MARGIN_RATIO or rel_y2 > BOTTOM_MARGIN_RATIO:
            return LayoutBlockType.PAGE_NUMBER, 0.85

    # Font-size-driven classification (title/heading/subheading) takes
    # priority over the top/bottom-margin position heuristic: a large,
    # short, top-of-page block is almost always the document title, not a
    # running header (which normally sits in body-sized type).
    if n_lines <= 2 and block.font_size >= median_font * 1.35 and rel_y1 < 0.25:
        return LayoutBlockType.TITLE, 0.75

    if n_lines <= 2 and block.font_size >= median_font * 1.15:
        return LayoutBlockType.HEADING, 0.7

    if rel_y2 < TOP_MARGIN_RATIO and rel_width > 0.4:
        return LayoutBlockType.HEADER, 0.7

    if rel_y1 > BOTTOM_MARGIN_RATIO and rel_width > 0.4:
        return LayoutBlockType.FOOTER, 0.7

    if rel_y1 > FOOTNOTE_BAND_RATIO and block.font_size < median_font * 0.85:
        return LayoutBlockType.FOOTNOTE, 0.6

    if n_lines <= 2 and block.font_size >= median_font * 1.05:
        return LayoutBlockType.SUBHEADING, 0.6

    return LayoutBlockType.PARAGRAPH, 0.65


def _digit_ratio(text: str) -> float:
    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return 0.0
    digits = sum(1 for c in stripped if c.isdigit() or c in "०१२३४५६७८९")
    return digits / len(stripped)


def _infer_alignment(block: TextBlockCandidate, page_width: int) -> str:
    center = (block.bbox.x1 + block.bbox.x2) / 2
    page_center = page_width / 2
    if abs(center - page_center) < page_width * 0.06:
        return "CENTER"
    if block.bbox.x2 > page_width * 0.85:
        return "RIGHT" if block.bbox.x1 > page_width * 0.5 else "LEFT"
    return "LEFT"


def _line_bbox(line: list[OCRWordResult]) -> BBox:
    return BBox(
        x1=min(w.bbox.x1 for w in line),
        y1=min(w.bbox.y1 for w in line),
        x2=max(w.bbox.x2 for w in line),
        y2=max(w.bbox.y2 for w in line),
    )


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    ix1, ix2 = max(a.x1, b.x1), min(a.x2, b.x2)
    inter = max(0.0, ix2 - ix1)
    union_w = max(a.x2, b.x2) - min(a.x1, b.x1)
    return inter / union_w if union_w > 0 else 0.0


def _overlap_ratio(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    return inter / a_area


def _center_in(bbox: BBox, container: BBox) -> bool:
    cx, cy = (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2
    return container.x1 <= cx <= container.x2 and container.y1 <= cy <= container.y2
