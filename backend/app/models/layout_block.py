from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.enum_types import pg_enum
from app.models.enums import LayoutBlockType


class LayoutBlock(Base, TimestampMixin):
    """
    A semantically classified region of a page (heading, paragraph, table,
    image, header/footer, page number, ...). `content` holds the ordered
    list of OCRBlock ids (+ merged text/style) that make up this block.
    `z_order` preserves stacking/reading order for reconstruction.
    """

    __tablename__ = "layout_blocks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )

    block_type: Mapped[LayoutBlockType] = mapped_column(pg_enum(LayoutBlockType, "layout_block_type"), nullable=False)
    bbox: Mapped[dict] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    z_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Ordered OCR block ids + derived text runs belonging to this layout block.
    content: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # font_size, bold, italic, alignment, line_spacing ...
    style: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    is_edited: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    edited_text: Mapped[Optional[str]] = mapped_column(String(8192), nullable=True)

    page: Mapped["DocumentPage"] = relationship(back_populates="layout_blocks")
    table: Mapped[Optional["Table"]] = relationship(back_populates="layout_block", uselist=False)
