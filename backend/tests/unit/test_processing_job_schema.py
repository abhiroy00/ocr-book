"""
Tests for `ProcessingJobRead.processing_duration_seconds` -- the
authoritative, backend-computed total pipeline duration the frontend
Process Timer freezes on once a job reaches a terminal state. No new DB
column/migration backs this: `started_at`/`finished_at` already exist on
`ProcessingJob` and are already set together on every terminal transition
(see `document_service.update_job_progress`), so this is pure computation
over fields the schema already carries.
"""
from datetime import datetime, timedelta, timezone

from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum, ProcessingStage
from app.schemas.document import ProcessingJobRead


def _job_read(started_at=None, finished_at=None, status=DocumentStatus.OCR_PROCESSING) -> ProcessingJobRead:
    return ProcessingJobRead(
        id="job-1", document_id="doc-1", status=status, stage=ProcessingStage.OCR, progress_percent=50,
        ocr_provider=OCRProviderEnum.PADDLEOCR, dpi=150, preprocess_profile=PreprocessProfileEnum.FAST,
        started_at=started_at, finished_at=finished_at, error_message=None,
    )


def test_duration_is_null_while_job_has_not_started():
    job = _job_read(started_at=None, finished_at=None, status=DocumentStatus.QUEUED)
    assert job.processing_duration_seconds is None


def test_duration_is_null_while_job_is_still_running():
    """The live ticking display is a frontend-only concern while active --
    the backend must not pretend to have a final duration yet."""
    started = datetime.now(timezone.utc) - timedelta(seconds=90)
    job = _job_read(started_at=started, finished_at=None, status=DocumentStatus.OCR_PROCESSING)
    assert job.processing_duration_seconds is None


def test_duration_is_computed_once_completed():
    started = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=155)
    job = _job_read(started_at=started, finished_at=finished, status=DocumentStatus.COMPLETED)
    assert job.processing_duration_seconds == 155.0


def test_duration_is_computed_when_failed():
    """Spec: don't hide the duration just because the job failed."""
    started = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=102)
    job = _job_read(started_at=started, finished_at=finished, status=DocumentStatus.FAILED)
    assert job.processing_duration_seconds == 102.0


def test_duration_is_computed_when_cancelled():
    started = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=47)
    job = _job_read(started_at=started, finished_at=finished, status=DocumentStatus.CANCELLED)
    assert job.processing_duration_seconds == 47.0


def test_duration_is_serialized_in_the_json_response():
    started = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=155)
    job = _job_read(started_at=started, finished_at=finished, status=DocumentStatus.COMPLETED)
    payload = job.model_dump(mode="json")
    assert payload["processing_duration_seconds"] == 155.0


def test_duration_never_negative_even_with_clock_skew():
    """Defensive: a `finished_at` that ends up slightly before
    `started_at` (clock skew across processes) must not surface as a
    confusing negative duration."""
    started = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    finished = started - timedelta(seconds=2)
    job = _job_read(started_at=started, finished_at=finished, status=DocumentStatus.COMPLETED)
    assert job.processing_duration_seconds == 0.0
