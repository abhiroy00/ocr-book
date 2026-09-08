from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum, ProcessingStage


class DocumentUploadResponse(BaseModel):
    document_id: str
    job_id: str
    status: DocumentStatus
    original_filename: str
    page_count: int


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    file_extension: str
    mime_type: str
    file_size_bytes: int
    page_count: int
    status: DocumentStatus
    ocr_provider: OCRProviderEnum
    dpi: int
    preprocess_profile: PreprocessProfileEnum
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentRead]
    total: int
    page: int
    page_size: int


class DocumentStats(BaseModel):
    total: int
    processing: int
    completed: int
    failed: int


class ProcessRequest(BaseModel):
    ocr_provider: Optional[OCRProviderEnum] = None
    dpi: Optional[int] = None
    preprocess_profile: Optional[PreprocessProfileEnum] = None


class ProcessingJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    status: DocumentStatus
    stage: ProcessingStage
    progress_percent: int
    ocr_provider: OCRProviderEnum
    dpi: int
    preprocess_profile: PreprocessProfileEnum
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ProgressEvent(BaseModel):
    document_id: str
    job_id: str
    stage: ProcessingStage
    percent: int
    page: Optional[int] = None
    message: str = ""
    status: DocumentStatus


class PageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    page_number: int
    width: int
    height: int
    dpi: int
    rotation: float
    original_image_url: str
    processed_image_url: Optional[str] = None
    ocr_status: str
    layout_status: str
    table_status: str
    ocr_confidence_avg: Optional[float] = None
    layout_confidence_avg: Optional[float] = None
    table_confidence_avg: Optional[float] = None
    visual_similarity_score: Optional[float] = None


class QualityPageReport(BaseModel):
    page_number: int
    ocr_confidence_avg: Optional[float]
    layout_confidence_avg: Optional[float]
    table_confidence_avg: Optional[float]
    visual_similarity_score: Optional[float]
    needs_review: bool


class QualityReport(BaseModel):
    document_id: str
    pages: list[QualityPageReport]
    overall_visual_similarity: Optional[float]
