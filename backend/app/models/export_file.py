from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.enum_types import pg_enum
from app.models.enums import ExportType


class ExportFile(Base, TimestampMixin):
    __tablename__ = "export_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    export_type: Mapped[ExportType] = mapped_column(pg_enum(ExportType, "export_type"), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="exports")
