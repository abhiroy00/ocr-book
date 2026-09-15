"""Library accession register API (spec sections 6/7/16) -- cumulative,
database-backed, PDF-extraction-derived book register and its Master
Excel export."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.document import Document
from app.exporters.master_register_exporter import render_master_register_excel
from app.schemas.accession import AccessionRecordListResponse, AccessionRecordRead, AccessionSummary
from app.services import accession_service

router = APIRouter(prefix="/accession-records", tags=["accession-register"])


@router.get("", response_model=AccessionRecordListResponse)
def list_accession_records(
    year: Optional[int] = None,
    month: Optional[int] = Query(default=None, ge=1, le=12),
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
):
    items, total = accession_service.list_accession_records(db, year=year, month=month, page=page, page_size=page_size)
    return AccessionRecordListResponse(
        items=[AccessionRecordRead.model_validate(r) for r in items], total=total, page=page, page_size=page_size
    )


@router.get("/summary", response_model=AccessionSummary)
def accession_summary(db: Session = Depends(get_db)):
    return AccessionSummary(**accession_service.get_accession_summary(db))


@router.get("/export/master-excel")
def export_master_excel(db: Session = Depends(get_db)):
    """Always generated fresh from the database (spec section 8) -- every
    call returns the complete, current cumulative dataset, including any
    document processed since the last download; nothing is cached or
    reset between downloads (spec section 7)."""
    records = accession_service.get_all_accession_records(db)
    documents = list(db.scalars(select(Document).where(Document.is_deleted.is_(False)).order_by(Document.created_at)).all())
    data = render_master_register_excel(records, documents)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="master_extracted_data.xlsx"'},
    )
