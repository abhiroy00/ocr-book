from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AccessionRecordRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    accession_number: str
    book_name: str
    creator: Optional[str] = None
    language: Optional[str] = None
    year_of_publication: Optional[str] = None
    total_pages: int
    record_type: Optional[str] = None
    record_date: date
    needs_review: bool
    extraction_notes: Optional[str] = None
    created_at: datetime


class AccessionRecordListResponse(BaseModel):
    items: list[AccessionRecordRead]
    total: int
    page: int
    page_size: int


class AccessionSummary(BaseModel):
    total_records: int
    needs_review_count: int
    latest_record_date: Optional[date] = None
    total_documents_processed: int
