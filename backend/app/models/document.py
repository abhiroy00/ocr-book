from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.enum_types import pg_enum
from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum


class Document(TimestampMixin, Base):
    """A single uploaded scanned document (PDF or image) and its processing state."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)

    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(16), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    status: Mapped[DocumentStatus] = mapped_column(
        pg_enum(DocumentStatus, "document_status"), default=DocumentStatus.UPLOADED, nullable=False, index=True
    )

    ocr_provider: Mapped[OCRProviderEnum] = mapped_column(
        pg_enum(OCRProviderEnum, "ocr_provider_enum"), default=OCRProviderEnum.PADDLEOCR, nullable=False
    )
    dpi: Mapped[int] = mapped_column(Integer, default=300, nullable=False)
    preprocess_profile: Mapped[PreprocessProfileEnum] = mapped_column(
        pg_enum(PreprocessProfileEnum, "preprocess_profile_enum"),
        default=PreprocessProfileEnum.BALANCED,
        nullable=False,
    )

    storage_original_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    # sha256 of the uploaded file's raw bytes -- duplicate-upload detection
    # (spec section 9). Nullable (not unique-constrained at the DB level)
    # so pre-existing rows from before this column existed stay valid;
    # the actual "is this a duplicate" check is an application-level query
    # (`document_service.find_document_by_hash`) filtered to
    # `is_deleted=False`, not a DB constraint, since a duplicate is a
    # soft, overridable warning ("reprocess anyway?"), not a hard
    # invariant the database should enforce.
    document_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    uploaded_by: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    pages: Mapped[List["DocumentPage"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="DocumentPage.page_number"
    )
    jobs: Mapped[List["ProcessingJob"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    exports: Mapped[List["ExportFile"]] = relationship(back_populates="document", cascade="all, delete-orphan")
