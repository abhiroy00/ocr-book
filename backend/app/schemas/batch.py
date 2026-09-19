"""API schemas for multiple-file (batch) OCR. `job_id` is the stable per-file
id (it survives retries); `document_id` / `processing_job_id` are the same
identifiers the existing single-file API already exposes, so every file in a
batch can still be opened, edited and exported through the normal document
pages. Filesystem paths are never included."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum, ProcessingStage


class BatchCreateRequest(BaseModel):
    ocr_provider: Optional[OCRProviderEnum] = None
    dpi: Optional[int] = None
    preprocess_profile: Optional[PreprocessProfileEnum] = None


class BatchItemRead(BaseModel):
    job_id: str
    document_id: str
    processing_job_id: Optional[str] = None
    filename: str
    file_size_bytes: int
    position: int
    # pending | queued | running | completed | failed | cancelled | duplicate
    state: str
    # The underlying pipeline's own status/stage (None for a file that has no
    # run yet or a duplicate that never ran in this batch).
    status: Optional[DocumentStatus] = None
    stage: Optional[ProcessingStage] = None
    progress_percent: int = 0
    processed_pages: int = 0
    total_pages: int = 0
    priority_score: float = 0.0
    attempts: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class ConcurrencyInfo(BaseModel):
    provider: str
    limit: int
    limiting_factor: str
    cpu_count: int
    total_mem_mb: Optional[int] = None
    bounds: dict[str, int] = {}


class BatchRead(BaseModel):
    batch_id: str
    # uploading | processing | completed | completed_with_errors | cancelled
    status: str
    ocr_provider: OCRProviderEnum
    dpi: int
    preprocess_profile: PreprocessProfileEnum
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None

    total_files: int
    completed_files: int
    failed_files: int
    cancelled_files: int
    duplicate_files: int
    queued_files: int
    processing_files: int
    finished_files: int

    total_pages: int
    processed_pages: int
    overall_percent: int

    concurrency: Optional[ConcurrencyInfo] = None
    items: list[BatchItemRead]


class BatchCapacityRead(BaseModel):
    """What the upload form needs to enforce limits client-side and to tell
    the user how much parallelism to expect on this server."""

    concurrency: ConcurrencyInfo
    max_files: int
    max_total_mb: int
    max_file_mb: int
    allowed_extensions: list[str]


class BatchStartResponse(BaseModel):
    batch_id: str
    queued_files: int
    status: str
