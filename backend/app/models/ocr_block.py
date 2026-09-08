from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid


class OCRBlock(Base, TimestampMixin):
    """
    One OCR recognition result (typically a "line" from the OCR engine),
    with coordinates in the page's pixel coordinate system.

    bbox / polygon schema (see app.schemas.ocr.BBox / Polygon):
        bbox:    {"x1": .., "y1": .., "x2": .., "y2": ..}
        polygon: [[x, y], [x, y], [x, y], [x, y]]
    """

    __tablename__ = "ocr_blocks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )

    block_id: Mapped[str] = mapped_column(String(64), nullable=False)
    line_id: Mapped[str] = mapped_column(String(64), nullable=False)
    word_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    text: Mapped[str] = mapped_column(String(2048), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    bbox: Mapped[dict] = mapped_column(JSON, nullable=False)
    polygon: Mapped[list] = mapped_column(JSON, nullable=False)

    language: Mapped[str] = mapped_column(String(16), default="und", nullable=False)
    font_size_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    reading_order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # User correction (never overwrites `text` silently — see acceptance
    # criterion #14: source content must never be silently rewritten).
    corrected_text: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    is_reviewed: Mapped[bool] = mapped_column(default=False, nullable=False)

    page: Mapped["DocumentPage"] = relationship(back_populates="ocr_blocks")
