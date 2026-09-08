"""
Builds the canonical Document JSON (spec section 14) for one page from the
pipeline's intermediate outputs (OCR words, classified layout blocks,
detected tables). This is a pure function — no DB, no I/O — so it is easy
to unit test and to re-run after a user edit.
"""
from __future__ import annotations

import uuid

from app.layout.detector import LayoutBlockResult
from app.models.enums import LayoutBlockType
from app.schemas.document_json import DocumentBlockJSON, PageJSON, TableBlockJSON, TableCellJSON, TableRowJSON, TextRunJSON
from app.schemas.geometry import BBox
from app.tables.models import DetectedTable


def build_page_json(
    document_id: str,
    page_id: str,
    page_number: int,
    page_width: int,
    page_height: int,
    dpi: int,
    rotation: float,
    layout_results: list[LayoutBlockResult],
    tables: list[DetectedTable],
) -> PageJSON:
    blocks: list[DocumentBlockJSON] = []

    for result in layout_results:
        # Prefer the DB LayoutBlock id (set by document_service once the row
        # exists) so this JSON block's id matches the DB row it came from —
        # required for structural table edits to resync the right block.
        block_id = result.db_id or f"block_{uuid.uuid4().hex[:10]}"

        table_json = None
        if result.block_type == LayoutBlockType.TABLE and result.table_ref is not None:
            table_json = _table_to_json(tables[result.table_ref])

        content = []
        if result.text and result.block_type != LayoutBlockType.TABLE:
            content = [
                TextRunJSON(
                    text=result.text,
                    font_size=result.style.get("font_size"),
                    confidence=result.confidence,
                )
            ]

        blocks.append(
            DocumentBlockJSON(
                id=block_id,
                type=result.block_type,
                bbox=result.bbox,
                bbox_norm=result.bbox.to_norm(page_width, page_height),
                confidence=result.confidence,
                z_order=result.z_order,
                content=content,
                table=table_json,
                image_ref=result.image_ref,
                style=result.style,
            )
        )

    return PageJSON(
        document_id=document_id,
        page_id=page_id,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        dpi=dpi,
        rotation=rotation,
        blocks=blocks,
    )


def _table_to_json(table: DetectedTable) -> TableBlockJSON:
    rows: dict[int, list[TableCellJSON]] = {}
    for cell in table.cells:
        rows.setdefault(cell.row, []).append(
            TableCellJSON(
                row=cell.row,
                column=cell.column,
                rowspan=cell.rowspan,
                colspan=cell.colspan,
                text=cell.text,
                bbox=cell.bbox,
                align_h=cell.align_h,
                align_v=cell.align_v,
                confidence=cell.confidence,
                is_header=cell.is_header,
            )
        )
    ordered_rows = [TableRowJSON(cells=sorted(rows[r], key=lambda c: c.column)) for r in sorted(rows.keys())]
    return TableBlockJSON(
        rows=ordered_rows,
        column_widths=table.column_widths,
        row_heights=table.row_heights,
        border_style=table.border_style,
        detection_method=table.detection_method.value,
    )
