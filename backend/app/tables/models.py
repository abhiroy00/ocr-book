from __future__ import annotations

from dataclasses import dataclass, field

from app.models.enums import TableDetectionMethod, TextAlign, VerticalAlign
from app.schemas.geometry import BBox


@dataclass
class DetectedCell:
    row: int
    column: int
    rowspan: int
    colspan: int
    bbox: BBox
    text: str = ""
    confidence: float = 0.0
    align_h: TextAlign = TextAlign.LEFT
    align_v: VerticalAlign = VerticalAlign.MIDDLE
    is_header: bool = False


@dataclass
class DetectedTable:
    bbox: BBox
    n_rows: int
    n_cols: int
    confidence: float
    detection_method: TableDetectionMethod
    cells: list[DetectedCell] = field(default_factory=list)
    column_widths: list[float] = field(default_factory=list)
    row_heights: list[float] = field(default_factory=list)
    border_style: dict = field(default_factory=dict)
