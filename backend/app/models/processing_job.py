from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.enum_types import pg_enum
from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum, ProcessingStage


class ProcessingJob(Base, TimestampMixin):
    """One run of the pipeline for a Document (re-processing creates a new job)."""

    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    status: Mapped[DocumentStatus] = mapped_column(
        pg_enum(DocumentStatus, "document_status", create_constraint=False),
        default=DocumentStatus.QUEUED,
        nullable=False,
    )
    stage: Mapped[ProcessingStage] = mapped_column(
        pg_enum(ProcessingStage, "processing_stage"), default=ProcessingStage.UPLOAD, nullable=False
    )
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    celery_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    ocr_provider: Mapped[OCRProviderEnum] = mapped_column(
        pg_enum(OCRProviderEnum, "ocr_provider_enum", create_constraint=False), nullable=False
    )
    dpi: Mapped[int] = mapped_column(Integer, nullable=False)
    preprocess_profile: Mapped[PreprocessProfileEnum] = mapped_column(
        pg_enum(PreprocessProfileEnum, "preprocess_profile_enum", create_constraint=False), nullable=False
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="jobs")
