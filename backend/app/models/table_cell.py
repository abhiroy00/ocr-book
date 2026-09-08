from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.enum_types import pg_enum
from app.models.enums import TextAlign, VerticalAlign


class TableCell(Base, TimestampMixin):
    __tablename__ = "table_cells"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    table_id: Mapped[str] = mapped_column(String(36), ForeignKey("tables.id", ondelete="CASCADE"), nullable=False, index=True)

    row: Mapped[int] = mapped_column(Integer, nullable=False)
    column: Mapped[int] = mapped_column(Integer, nullable=False)
    rowspan: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    colspan: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    text: Mapped[str] = mapped_column(String(2048), default="", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    bbox: Mapped[dict] = mapped_column(JSON, nullable=False)

    align_h: Mapped[TextAlign] = mapped_column(pg_enum(TextAlign, "cell_align_h"), default=TextAlign.LEFT, nullable=False)
    align_v: Mapped[VerticalAlign] = mapped_column(
        pg_enum(VerticalAlign, "cell_align_v"), default=VerticalAlign.MIDDLE, nullable=False
    )
    is_header: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    is_edited: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    edited_text: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)

    table: Mapped["Table"] = relationship(back_populates="cells")
