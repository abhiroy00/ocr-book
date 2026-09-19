"""
Automatic Master Accession Register: a single, persisted, ever-growing
Excel workbook (`storage/master_register/master_accession_register.xlsx`,
sheet "Common") that gets one row appended the moment each document
finishes processing -- distinct from, and in addition to, the existing
DB-backed accession register (`app.services.accession_service` +
`app.exporters.master_register_exporter`), which regenerates a workbook
fresh from the database on every download rather than persisting one on
disk.

Why both exist: the DB-backed register is the correct source of truth for
everything the app itself reads back (listings, summaries, the on-demand
export endpoint) -- it can never drift or corrupt because it is never
written to as a file. This module adds what that approach can't give you
by itself: one physical file on the server's disk that keeps growing in
place in real time as each document completes, e.g. for direct access
from outside the app (mounted drive, scheduled backup, opening it in
Excel while uploads are still running) without needing to hit the API.
It always sources its row data from that same DB (`Document` +
`AccessionRecord`), so the two registers never disagree about a given
document's accession number, title, or metadata -- only about *how* that
data is exposed.

Concurrency model (multiple Celery workers on one EC2 host, spec: cloud
safety): the workbook is a single shared file, so exactly one worker may
be in its load -> modify -> save critical section at a time. That is
enforced by an OS-level file lock (`app.core.excel_lock`), scoped as
tightly as possible -- acquired only around this module's own read-modify-
write section, never around the (slow) OCR pipeline itself.
"""
from __future__ import annotations

import os
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from openpyxl import Workbook
from openpyxl import load_workbook as _openpyxl_load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.excel_lock import FileLock, Timeout, new_file_lock
from app.core.logging import get_logger
from app.models.document import Document
from app.models.accession_record import AccessionRecord
from app.models.enums import DocumentStatus, ExportType
from app.models.export_file import ExportFile
from app.models.processing_job import ProcessingJob
from app.services.storage import get_storage

logger = get_logger(__name__)

REGISTER_DIR = "master_register"
REGISTER_FILENAME = "master_accession_register.xlsx"
REGISTER_RELATIVE_PATH = f"{REGISTER_DIR}/{REGISTER_FILENAME}"
SHEET_NAME = "Common"

# Exact column order requested for this register -- distinct from (a
# differently-shaped set of columns than) the DB-backed register's own
# sheet, which is why this lives in its own module/file rather than
# reusing `master_register_exporter`'s layout. No document-identifying
# column is kept here on purpose: this register's duplicate rule is
# defined purely on Accession Number (see `is_duplicate`), so once a row
# is written it is never located/edited again by anything other than a
# person opening the file -- an append-only ledger, not a synced mirror
# of the DB.
HEADERS: list[str] = [
    "Date",
    "Accession Number",
    "Book Name",
    "Page",
    "Language",
    "Year of Publication",
    "Creator",
    "Searchable PDF Filename",
    "PDF Path",
    "Processing Timestamp",
]
(
    COL_DATE, COL_ACCESSION, COL_BOOK_NAME, COL_PAGE, COL_LANGUAGE, COL_YEAR,
    COL_CREATOR, COL_PDF_FILENAME, COL_PDF_PATH, COL_TIMESTAMP,
) = range(1, len(HEADERS) + 1)

# Starting width per column -- `autosize_columns` only ever widens from
# here for a newly-written row's actual content, never rescans the whole
# sheet (this workbook may hold 20,000+ rows; an O(n) rescan on every
# single append would make appends progressively slower forever).
_DEFAULT_COLUMN_WIDTHS = [13, 18, 45, 8, 16, 20, 32, 30, 45, 20]
_MAX_COLUMN_WIDTH = 60

_HEADER_FILL = PatternFill(start_color="3A8E2D", end_color="3A8E2D", fill_type="solid")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrap_text=True)
_THIN_BORDER = Border(*(Side(style="thin", color="000000") for _ in range(4)))
# Data rows (as opposed to the header): wrapped text, top-aligned -- so a
# long book name or path doesn't force one giant row height for every
# other column sharing that row.
_DATA_ALIGNMENT = Alignment(wrap_text=True, vertical="top")

_ACCESSION_NUMBER_RE = re.compile(r"^(?P<prefix>.+)-(?P<n>\d+)$")


@dataclass
class RegisterRow:
    record_date: date
    accession_number: str
    book_name: str
    page_count: int
    language: Optional[str]
    year_of_publication: Optional[str]
    creator: Optional[str]
    pdf_filename: Optional[str]
    pdf_path: Optional[str]
    processing_timestamp: datetime

    def as_values(self) -> list:
        return [
            self.record_date, self.accession_number, self.book_name, self.page_count,
            self.language, self.year_of_publication, self.creator, self.pdf_filename,
            self.pdf_path, self.processing_timestamp,
        ]


# ---------------------------------------------------------------------------
# Workbook lifecycle
# ---------------------------------------------------------------------------

def ensure_master_excel() -> str:
    """Creates the workbook (header only, formatted) the first time it's
    needed; never touches it if it already exists. Returns the local
    filesystem path (directories created as needed). Safe to call from
    every worker on every job -- idempotent, and the actual creation is
    itself performed under the same file lock everything else uses, so
    two workers racing to create it for the very first time can't both
    "win" and stomp each other's `Workbook()`."""
    storage = get_storage()
    path = storage.local_path(REGISTER_RELATIVE_PATH)
    if os.path.exists(path):
        return path

    lock = acquire_lock()
    try:
        if os.path.exists(path):  # re-check: another worker may have created it while we waited for the lock
            return path
        wb = Workbook()
        sheet = wb.active
        sheet.title = SHEET_NAME
        create_header(sheet)
        format_header(sheet)
        autosize_columns(sheet)
        wb.save(path)
        storage.sync_from_local(REGISTER_RELATIVE_PATH)
        logger.info("Created Master Register", path=path)
    finally:
        release_lock(lock)
    return path


def load_workbook() -> Workbook:
    """Opens the (already-created) register workbook for reading/writing."""
    path = ensure_master_excel()
    return _openpyxl_load_workbook(path)


def create_header(sheet: Worksheet) -> None:
    for col_idx, label in enumerate(HEADERS, start=1):
        sheet.cell(row=1, column=col_idx, value=label)


def format_header(sheet: Worksheet) -> None:
    for col_idx in range(1, len(HEADERS) + 1):
        cell = sheet.cell(row=1, column=col_idx)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = _HEADER_ALIGNMENT
        cell.border = _THIN_BORDER
    sheet.freeze_panes = "A2"
    _apply_auto_filter(sheet)


def autosize_columns(sheet: Worksheet, row_values: list | None = None) -> None:
    """Widens (never shrinks below the sane default) each column to fit
    `row_values` -- the one row just written, not a rescan of every
    existing row. Called once with no `row_values` at creation (sets the
    defaults) and once per append/update with that row's actual values."""
    for col_idx, default_width in enumerate(_DEFAULT_COLUMN_WIDTHS, start=1):
        letter = get_column_letter(col_idx)
        current = sheet.column_dimensions[letter].width or default_width
        target = current
        if row_values is not None and col_idx - 1 < len(row_values):
            value = row_values[col_idx - 1]
            if value is not None:
                target = max(target, min(len(str(value)) + 2, _MAX_COLUMN_WIDTH))
        sheet.column_dimensions[letter].width = max(target, default_width)


def _apply_auto_filter(sheet: Worksheet) -> None:
    last_row = max(2, sheet.max_row)
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{last_row}"


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def acquire_lock() -> FileLock:
    settings = get_settings()
    path = get_storage().local_path(REGISTER_RELATIVE_PATH)
    lock = new_file_lock(path, timeout=settings.master_register_lock_timeout_seconds)
    logger.info("master_register_lock_waiting")
    lock.acquire()
    logger.info("master_register_lock_acquired")
    return lock


def release_lock(lock: FileLock) -> None:
    lock.release()
    logger.info("master_register_lock_released")


# ---------------------------------------------------------------------------
# Duplicate protection / accession-number generation
# ---------------------------------------------------------------------------

def _normalize_accession(value) -> str:
    return str(value).strip().lower() if value else ""


def _is_duplicate_in_sheet(sheet: Worksheet, accession_number: str, ignore_row: int | None = None) -> bool:
    target = _normalize_accession(accession_number)
    if not target:
        return False
    for row in range(2, sheet.max_row + 1):
        if row == ignore_row:
            continue
        if _normalize_accession(sheet.cell(row=row, column=COL_ACCESSION).value) == target:
            return True
    return False


def is_duplicate(accession_number: str) -> bool:
    """Standalone check against the current on-disk register. The
    authoritative check that actually guards a write happens inside
    `append_document_record`'s own locked critical section -- this is for
    callers outside that path (e.g. a pre-check before assigning a manual
    accession number) that don't need to hold the write lock."""
    wb = load_workbook()
    try:
        return _is_duplicate_in_sheet(wb[SHEET_NAME], accession_number)
    finally:
        wb.close()


def _generate_next_accession_in_sheet(sheet: Worksheet, prefix: str, start: int) -> str:
    max_n = start - 1
    for row in range(2, sheet.max_row + 1):
        value = sheet.cell(row=row, column=COL_ACCESSION).value
        if not value:
            continue
        match = _ACCESSION_NUMBER_RE.match(str(value).strip())
        if match and match.group("prefix") == prefix:
            max_n = max(max_n, int(match.group("n")))
    return f"{prefix}-{max_n + 1}"


def generate_next_accession() -> str:
    """`{prefix}-{n}` continuing from the highest existing number already
    present *in the workbook itself* (as opposed to
    `app.services.accession_service.generate_next_accession_number`, which
    reads the same sequence from the database) -- used only as a fallback
    for a row that somehow has no accession number yet; the normal
    pipeline path always already has one from the DB-backed
    `AccessionRecord` by the time it reaches this module."""
    settings = get_settings()
    wb = load_workbook()
    try:
        return _generate_next_accession_in_sheet(
            wb[SHEET_NAME], settings.accession_number_prefix, settings.accession_number_start
        )
    finally:
        wb.close()


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_register(sheet: Worksheet) -> None:
    """Keeps the sheet ordered by Date ascending, then Accession Number
    ascending within the same date -- re-run after every append/update so
    the workbook is always in this order on disk, not just at export
    time."""
    if sheet.max_row <= 2:
        return

    rows = [[sheet.cell(row=r, column=c).value for c in range(1, len(HEADERS) + 1)] for r in range(2, sheet.max_row + 1)]

    def sort_key(values: list):
        raw_date = values[COL_DATE - 1]
        if isinstance(raw_date, datetime):
            raw_date = raw_date.date()
        accession = str(values[COL_ACCESSION - 1] or "").strip().lower()
        return (raw_date or date.min, accession)

    rows.sort(key=sort_key)

    for row_idx, values in enumerate(rows, start=2):
        _write_row_values(sheet, row_idx, values)


def _write_row_values(sheet: Worksheet, row_idx: int, values: list) -> None:
    for col_idx, value in enumerate(values, start=1):
        cell = sheet.cell(row=row_idx, column=col_idx, value=value)
        cell.alignment = _DATA_ALIGNMENT
        if col_idx == COL_DATE:
            cell.number_format = "DD-MM-YYYY"
        elif col_idx == COL_TIMESTAMP:
            cell.number_format = "DD-MM-YYYY HH:MM:SS"


# ---------------------------------------------------------------------------
# Crash-safe save
# ---------------------------------------------------------------------------

def _tmp_path_for(path: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}.tmp{p.suffix}"))


def save_atomic(wb: Workbook) -> None:
    """Writes to a `.tmp.xlsx` sibling file, then atomically renames it
    over the real workbook. If `wb.save` raises partway through (disk
    full, process killed), the tmp file is discarded and the ORIGINAL
    workbook -- never touched -- is left exactly as it was."""
    storage = get_storage()
    path = storage.local_path(REGISTER_RELATIVE_PATH)
    tmp_path = _tmp_path_for(path)
    try:
        wb.save(tmp_path)
        Path(tmp_path).replace(path)  # atomic rename on the same filesystem (POSIX rename / Windows ReplaceFile)
    except Exception:
        with suppress(OSError):
            os.remove(tmp_path)
        logger.error("Failed writing register", path=path)
        raise
    storage.sync_from_local(REGISTER_RELATIVE_PATH)
    logger.info("master_register_saved", path=path)


# ---------------------------------------------------------------------------
# Row lookup / building
# ---------------------------------------------------------------------------

def _latest_completed_job(db: Session, document_id: str) -> Optional[ProcessingJob]:
    """The most recent processing run for this document (reprocessing
    creates a new `ProcessingJob` row each time, per
    `app.models.processing_job`) -- used to enforce "only ever append for
    a job that actually reached COMPLETED", independent of whatever the
    caller believes about its own state."""
    return db.scalar(
        select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc())
    )


def _latest_searchable_pdf(db: Session, document_id: str) -> tuple[Optional[str], Optional[str]]:
    export = db.scalar(
        select(ExportFile)
        .where(ExportFile.document_id == document_id, ExportFile.export_type == ExportType.SEARCHABLE_PDF)
        .order_by(ExportFile.created_at.desc())
    )
    if export is None:
        return None, None
    return os.path.basename(export.storage_path), export.storage_path


def _as_naive_utc(moment: Optional[datetime]) -> datetime:
    """openpyxl can't store tz-aware datetimes -- normalize to naive UTC,
    falling back to "now" when the job has no recorded finish time."""
    if moment is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if moment.tzinfo is not None:
        return moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def _build_row(
    record: AccessionRecord, pdf_filename: Optional[str], pdf_path: Optional[str], processed_at: Optional[datetime]
) -> RegisterRow:
    return RegisterRow(
        record_date=record.record_date,
        accession_number=record.accession_number,
        book_name=record.book_name,
        page_count=record.total_pages,
        language=record.language,
        year_of_publication=record.year_of_publication,
        creator=record.creator,
        pdf_filename=pdf_filename,
        pdf_path=pdf_path,
        processing_timestamp=_as_naive_utc(processed_at),
    )


def _build_eligible_row(db: Session, document_id: str, log_skips: bool = True) -> Optional[RegisterRow]:
    """Builds the register row for `document_id`, or None (logging why, if
    `log_skips`) unless every precondition holds: the document exists, its
    most recent `ProcessingJob` is COMPLETED (never FAILED/CANCELLED), it
    has an `AccessionRecord`, and a `SEARCHABLE_PDF` export exists."""
    document = db.get(Document, document_id)
    if document is None:
        if log_skips:
            logger.warning("master_register_skipped_missing_document", document_id=document_id)
        return None

    job = _latest_completed_job(db, document_id)
    if job is None or job.status != DocumentStatus.COMPLETED:
        if log_skips:
            logger.info("master_register_skipped_not_completed", document_id=document_id, status=getattr(job, "status", None))
        return None

    record = db.scalar(select(AccessionRecord).where(AccessionRecord.document_id == document_id))
    if record is None:
        if log_skips:
            logger.warning("master_register_skipped_missing_accession_record", document_id=document_id)
        return None

    pdf_filename, pdf_path = _latest_searchable_pdf(db, document_id)
    if pdf_filename is None:
        if log_skips:
            logger.info("master_register_skipped_no_searchable_pdf", document_id=document_id)
        return None

    return _build_row(record, pdf_filename, pdf_path, job.finished_at)


def _read_accession_set() -> set[str]:
    """Normalized accession numbers already in the register -- read once
    (read-only, no lock needed: the file is only ever replaced atomically,
    so a reader sees either the old or the new version, never a torn one)."""
    path = ensure_master_excel()
    wb = _openpyxl_load_workbook(path, read_only=True)
    try:
        sheet = wb[SHEET_NAME]
        return {
            _normalize_accession(value)
            for (value,) in sheet.iter_rows(min_row=2, min_col=COL_ACCESSION, max_col=COL_ACCESSION, values_only=True)
            if value
        }
    finally:
        wb.close()


def _append_rows(rows: list[RegisterRow], log_duplicates: bool = True) -> list[RegisterRow]:
    """Under the register's file lock: load once, append every row whose
    accession number isn't already present, sort, save once (atomically).
    Returns the rows actually written (empty if all were duplicates, the
    lock timed out, or the save failed -- the original file is untouched
    in every one of those cases)."""
    if not rows:
        return []

    try:
        ensure_master_excel()
        lock = acquire_lock()
    except Timeout:
        logger.error("Failed writing register")
        return []

    try:
        wb = load_workbook()
        try:
            sheet = wb[SHEET_NAME]
            written: list[RegisterRow] = []
            for row in rows:
                if _is_duplicate_in_sheet(sheet, row.accession_number):
                    if log_duplicates:
                        logger.info(f"Skipped duplicate accession {row.accession_number}")
                    continue
                _write_row_values(sheet, sheet.max_row + 1, row.as_values())
                autosize_columns(sheet, row.as_values())
                written.append(row)

            if not written:
                return []

            sort_register(sheet)
            _apply_auto_filter(sheet)
            try:
                save_atomic(wb)
            except Exception:
                logger.error("Failed writing register")
                return []

            for row in written:
                logger.info(f"Appended accession {row.accession_number}")
            return written
        finally:
            wb.close()
    finally:
        release_lock(lock)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def append_document_record(db: Session, document_id: str) -> bool:
    """Appends one row for `document_id` to the live Master Register --
    called immediately after that ONE document finishes (no batch commit;
    whichever document finishes first gets appended first, regardless of
    upload order).

    Independently re-verifies, rather than trusting the caller, every
    precondition the spec requires before writing anything:
      - the document's most recent `ProcessingJob` reached `COMPLETED`
        (never for a FAILED or CANCELLED run);
      - a `SEARCHABLE_PDF` export actually exists for it;
      - its accession number isn't already in the register (an
        append-only ledger: once a row exists for an accession number, it
        is never edited or duplicated -- a reprocess of the same document
        reuses the same accession number, per `accession_service`, and so
        is itself just another duplicate that gets skipped, same as any
        other).

    Never raises for any of those expected conditions, nor for a lock
    timeout -- callers treat this as best-effort, exactly like the
    DB-backed accession record it reads from. Returns True if a row was
    written, False if skipped (and logged why).
    """
    row = _build_eligible_row(db, document_id)
    if row is None:
        return False
    return bool(_append_rows([row]))


def sync_register_from_db(db: Session) -> int:
    """Backfills the register with every already-COMPLETED document that
    isn't in it yet -- e.g. books processed before this feature was
    deployed, or a live append that failed at the time (lock timeout, disk
    hiccup). Same preconditions and duplicate rule as
    `append_document_record`; safe to call any number of times (a no-op
    once everything is present). One lock acquisition and one atomic save
    for the whole batch. Returns how many rows were appended."""
    present = _read_accession_set()
    candidates = db.execute(
        select(AccessionRecord.document_id, AccessionRecord.accession_number)
        .join(Document, Document.id == AccessionRecord.document_id)
        .where(Document.is_deleted.is_(False))
        .order_by(AccessionRecord.record_date, AccessionRecord.accession_number)
    ).all()

    rows = [
        row
        for document_id, accession_number in candidates
        if _normalize_accession(accession_number) not in present
        and (row := _build_eligible_row(db, document_id, log_skips=False)) is not None
    ]
    return len(_append_rows(rows, log_duplicates=False))


def rebuild_register_from_db(db: Session) -> int:
    """Rewrites every data row of the register from the database (header
    and formatting kept) -- a maintenance action for when the DB values
    behind existing rows changed (e.g. `refresh_register_details`
    re-read better metadata). Unlike the append-only live path this
    replaces rows, but nothing is lost: each row is regenerated from the
    same completed documents. One lock acquisition, one atomic save; on
    any failure the original file is untouched. Returns the row count."""
    candidates = db.execute(
        select(AccessionRecord.document_id)
        .join(Document, Document.id == AccessionRecord.document_id)
        .where(Document.is_deleted.is_(False))
        .order_by(AccessionRecord.record_date, AccessionRecord.accession_number)
    ).scalars().all()
    rows = [row for document_id in candidates if (row := _build_eligible_row(db, document_id, log_skips=False)) is not None]

    ensure_master_excel()
    lock = acquire_lock()
    try:
        wb = load_workbook()
        try:
            sheet = wb[SHEET_NAME]
            if sheet.max_row > 1:
                sheet.delete_rows(2, sheet.max_row - 1)
            for offset, row in enumerate(rows):
                _write_row_values(sheet, 2 + offset, row.as_values())
                autosize_columns(sheet, row.as_values())
            sort_register(sheet)
            _apply_auto_filter(sheet)
            save_atomic(wb)
        finally:
            wb.close()
    finally:
        release_lock(lock)

    logger.info(f"Rebuilt Master Register with {len(rows)} rows")
    return len(rows)


def refresh_register_details(db: Session) -> dict[str, int]:
    """Re-reads title / creator / year / language for every completed
    document from its stored OCR data (no re-OCR -- see
    `accession_service.refresh_metadata_from_stored_ocr`), then rebuilds
    the register so it shows the corrected values. Accession numbers are
    never changed. A document that can't be refreshed keeps its existing
    values and is counted under `failed`."""
    from app.services import accession_service

    documents = db.scalars(
        select(Document).where(Document.is_deleted.is_(False), Document.status == DocumentStatus.COMPLETED)
    ).all()

    refreshed = failed = 0
    for document in documents:
        try:
            if accession_service.refresh_metadata_from_stored_ocr(db, document) is not None:
                refreshed += 1
        except Exception as exc:  # noqa: BLE001 - one bad document must not block the rest
            db.rollback()
            failed += 1
            logger.error("master_register_refresh_failed", document_id=document.id, error=str(exc))

    return {"documents_refreshed": refreshed, "documents_failed": failed, "register_rows": rebuild_register_from_db(db)}


def get_master_register_bytes(db: Session) -> bytes:
    """The current Master Register file contents, for download. Syncs any
    missing completed documents in first (best-effort -- a sync failure
    still serves whatever the file already holds), so the download is
    always the full, up-to-date register."""
    try:
        appended = sync_register_from_db(db)
        if appended:
            logger.info("master_register_backfilled", rows=appended)
    except Exception as exc:  # noqa: BLE001 - never block the download over a backfill problem
        logger.error("master_register_sync_failed", error=str(exc))

    with open(ensure_master_excel(), "rb") as f:
        return f.read()
