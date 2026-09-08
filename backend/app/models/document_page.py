from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid


class DocumentPage(TimestampMixin, Base):
    """One rendered page of a Document, its images, and its Document JSON (IR)."""

    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Pixel dimensions at render DPI (the coordinate system OCR/layout/table
    # bboxes are stored in — see app.utils.coordinates for conversions).
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    dpi: Mapped[int] = mapped_column(Integer, nullable=False)
    rotation: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    original_image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    processed_image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    ocr_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    layout_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    table_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)

    ocr_confidence_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    layout_confidence_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    table_confidence_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    visual_similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # SSIM vs original

    # Canonical intermediate representation for this page (see
    # app.schemas.document_json.PageJSON) — the single source of truth that
    # reconstruction/export read from.
    document_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="pages")
    ocr_blocks: Mapped[List["OCRBlock"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    layout_blocks: Mapped[List["LayoutBlock"]] = relationship(
        back_populates="page", cascade="all, delete-orphan", order_by="LayoutBlock.z_order"
    )
    tables: Mapped[List["Table"]] = relationship(back_populates="page", cascade="all, delete-orphan")
