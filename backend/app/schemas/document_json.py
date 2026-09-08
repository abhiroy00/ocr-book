"""
The canonical Document JSON intermediate representation (IR).

This is the backbone described in spec section 14/42: every page is
represented as TEXT + TABLE + IMAGE + LINES + POSITION + SIZE + STYLE +
STRUCTURE, and reconstruction (PDF/DOCX) as well as the editor always read
and write *this* structure — never raw OCR text directly.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.enums import LayoutBlockType, TextAlign, VerticalAlign
from app.schemas.geometry import BBox, NormBBox


class TextRunJSON(BaseModel):
    """A styled run of text traceable back to the OCR block(s) it came from."""

    text: str
    bold: bool = False
    italic: bool = False
    font_size: Optional[float] = None
    language: str = "und"
    confidence: float = 1.0
    source_ocr_block_ids: list[str] = Field(default_factory=list)


class TableCellJSON(BaseModel):
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    bbox: BBox
    align_h: TextAlign = TextAlign.LEFT
    align_v: VerticalAlign = VerticalAlign.MIDDLE
    confidence: float = 0.0
    is_header: bool = False
    is_edited: bool = False


class TableRowJSON(BaseModel):
    cells: list[TableCellJSON] = Field(default_factory=list)


class TableBlockJSON(BaseModel):
    rows: list[TableRowJSON] = Field(default_factory=list)
    column_widths: list[float] = Field(default_factory=list)
    row_heights: list[float] = Field(default_factory=list)
    border_style: dict = Field(default_factory=dict)
    detection_method: str = "opencv_lines"


class DocumentBlockJSON(BaseModel):
    """One layout element on the page, positioned absolutely."""

    id: str
    type: LayoutBlockType
    bbox: BBox
    bbox_norm: NormBBox
    confidence: float
    z_order: int = 0

    # Populated for text-like blocks (heading/paragraph/footnote/...).
    content: list[TextRunJSON] = Field(default_factory=list)

    # Populated only for type == "table".
    table: Optional[TableBlockJSON] = None

    # Populated for image/chart/signature/stamp/handwritten blocks that are
    # preserved as-is (cropped from the processed page image) rather than
    # reconstructed as text/vector content.
    image_ref: Optional[str] = None

    # font_size, alignment, line_spacing, letter_spacing, color, etc.
    style: dict = Field(default_factory=dict)

    is_edited: bool = False


class PageJSON(BaseModel):
    document_id: str
    page_id: str
    page_number: int
    page_width: int
    page_height: int
    dpi: int
    rotation: float = 0.0
    blocks: list[DocumentBlockJSON] = Field(default_factory=list)

    def sorted_blocks(self) -> list[DocumentBlockJSON]:
        return sorted(self.blocks, key=lambda b: b.z_order)


class DocumentJSON(BaseModel):
    """Whole-document container: ordered PageJSON list."""

    document_id: str
    page_count: int
    pages: list[PageJSON] = Field(default_factory=list)
    generated_by: Literal["pipeline", "user_edit"] = "pipeline"
