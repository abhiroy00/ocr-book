from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.enum_types import pg_enum
from app.models.enums import TableDetectionMethod


class Table(Base, TimestampMixin):
    __tablename__ = "tables"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    layout_block_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("layout_blocks.id", ondelete="SET NULL"), nullable=True
    )

    bbox: Mapped[dict] = mapped_column(JSON, nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_cols: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detection_method: Mapped[TableDetectionMethod] = mapped_column(
        pg_enum(TableDetectionMethod, "table_detection_method"), nullable=False
    )

    # column widths / row heights / border thickness, all in page px.
    column_widths: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    row_heights: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    border_style: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    page: Mapped["DocumentPage"] = relationship(back_populates="tables")
    layout_block: Mapped[Optional["LayoutBlock"]] = relationship(back_populates="table")
    cells: Mapped[List["TableCell"]] = relationship(
        back_populates="table", cascade="all, delete-orphan", order_by="TableCell.row, TableCell.column"
    )
