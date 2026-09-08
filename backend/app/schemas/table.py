from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.enums import TableDetectionMethod, TextAlign, VerticalAlign
from app.schemas.geometry import BBox


class TableCellRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    table_id: str
    row: int
    column: int
    rowspan: int
    colspan: int
    text: str
    bbox: BBox
    align_h: TextAlign
    align_v: VerticalAlign
    confidence: float
    is_header: bool
    is_edited: bool
    edited_text: Optional[str] = None


class TableCellUpdate(BaseModel):
    text: Optional[str] = None
    align_h: Optional[TextAlign] = None
    align_v: Optional[VerticalAlign] = None
    rowspan: Optional[int] = None
    colspan: Optional[int] = None


class TableRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    page_id: str
    bbox: BBox
    n_rows: int
    n_cols: int
    confidence: float
    detection_method: TableDetectionMethod
    column_widths: list[float]
    row_heights: list[float]
    border_style: dict
    cells: list[TableCellRead] = []


class TableUpdate(BaseModel):
    """Structural edits: add/remove row or column, merge/split cells."""

    op: str  # "add_row" | "delete_row" | "add_column" | "delete_column" | "merge_cells" | "split_cell"
    row: Optional[int] = None
    column: Optional[int] = None
    cell_ids: Optional[list[str]] = None
