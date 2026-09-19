"""
Tests for the Celery batch-slot task (`app.workers.batch_tasks`): claim ->
run the existing pipeline -> mirror the outcome, with per-file failure
isolation and the worker-crash safety net. The pipeline itself is replaced
by a stub -- what is under test is the sequencing/isolation around it.
"""
import pytest
from celery.exceptions import Retry
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document import Document
from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum, ProcessingStage
from app.models.ocr_batch import BatchItemState, OCRBatch, OCRBatchItem
from app.models.processing_job import ProcessingJob
from app.services import document_service
from app.services.batch_scheduler import Claim, ClaimOutcome
from app.workers import batch_tasks


@pytest.fixture()
def Session(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(batch_tasks, "SessionLocal", maker)
    yield maker
    engine.dispose()


@pytest.fixture()
def running_item(Session):
    db = Session()
    batch = OCRBatch(ocr_provider="tesseract", dpi=150, preprocess_profile="FAST")
    db.add(batch)
    db.flush()
    doc = Document(
        original_filename="a.pdf", file_extension=".pdf", mime_type="application/pdf", file_size_bytes=1, page_count=2,
        status=DocumentStatus.QUEUED, ocr_provider=OCRProviderEnum.TESSERACT, dpi=150,
        preprocess_profile=PreprocessProfileEnum.FAST, storage_original_path="x",
    )
    db.add(doc)
    db.flush()
    job = ProcessingJob(
        document_id=doc.id, status=DocumentStatus.QUEUED, stage=ProcessingStage.UPLOAD, progress_percent=0,
        ocr_provider=OCRProviderEnum.TESSERACT, dpi=150, preprocess_profile=PreprocessProfileEnum.FAST,
    )
    db.add(job)
    db.flush()
    item = OCRBatchItem(
        batch_id=batch.id, document_id=doc.id, processing_job_id=job.id, position=0,
        state=BatchItemState.RUNNING.value, celery_task_id="task-1",
    )
    db.add(item)
    db.commit()
    ids = dict(batch=batch.id, item=item.id, doc=doc.id, job=job.id)
    db.close()
    return ids


def _claim_for(ids) -> Claim:
    return Claim(item_id=ids["item"], batch_id=ids["batch"], document_id=ids["doc"], job_id=ids["job"], provider="tesseract", limit=1, pool_cap=None)


def _state(Session, item_id):
    db = Session()
    try:
        item = db.get(OCRBatchItem, item_id)
        return item.state, item.error_message
    finally:
        db.close()


def test_a_successful_file_is_mirrored_as_completed(Session, running_item, monkeypatch):
    def pipeline(task_id, document_id, job_id, max_workers):
        db = Session()
        try:
            document_service.update_job_progress(db, db.get(ProcessingJob, job_id), ProcessingStage.DONE, 100, DocumentStatus.COMPLETED)
        finally:
            db.close()
        return "completed"

    monkeypatch.setattr(batch_tasks, "_claim", lambda b, t: (ClaimOutcome.CLAIMED, _claim_for(running_item)))
    monkeypatch.setattr(batch_tasks, "run_document_pipeline", pipeline)
    assert batch_tasks.run_batch_slot.run(running_item["batch"]) == "done"
    assert _state(Session, running_item["item"])[0] == "completed"


def test_a_failing_file_is_recorded_and_does_not_raise(Session, running_item, monkeypatch):
    """Error isolation: the exception must not escape the task (which would
    fail it in Celery) -- the file is marked failed and the batch carries on."""

    def pipeline(task_id, document_id, job_id, max_workers):
        raise RuntimeError("corrupt page 7")

    monkeypatch.setattr(batch_tasks, "_claim", lambda b, t: (ClaimOutcome.CLAIMED, _claim_for(running_item)))
    monkeypatch.setattr(batch_tasks, "run_document_pipeline", pipeline)
    assert batch_tasks.run_batch_slot.run(running_item["batch"]) == "failed"  # returned, not raised
    state, error = _state(Session, running_item["item"])
    assert state == "failed" and "corrupt page 7" in error


def test_a_pipeline_that_marked_the_job_failed_is_mirrored_with_its_message(Session, running_item, monkeypatch):
    def pipeline(task_id, document_id, job_id, max_workers):
        db = Session()
        try:
            document_service.update_job_progress(
                db, db.get(ProcessingJob, job_id), ProcessingStage.DONE, 10, DocumentStatus.FAILED, error="no OCR engine available"
            )
        finally:
            db.close()
        raise RuntimeError("no OCR engine available")

    monkeypatch.setattr(batch_tasks, "_claim", lambda b, t: (ClaimOutcome.CLAIMED, _claim_for(running_item)))
    monkeypatch.setattr(batch_tasks, "run_document_pipeline", pipeline)
    batch_tasks.run_batch_slot.run(running_item["batch"])
    assert _state(Session, running_item["item"]) == ("failed", "no OCR engine available")


def test_the_pool_cap_and_task_id_are_passed_to_the_shared_pipeline(Session, running_item, monkeypatch):
    seen = {}

    def pipeline(task_id, document_id, job_id, max_workers):
        seen.update(document_id=document_id, job_id=job_id, max_workers=max_workers)

    claim = Claim(**{**_claim_for(running_item).__dict__, "pool_cap": 3})
    monkeypatch.setattr(batch_tasks, "_claim", lambda b, t: (ClaimOutcome.CLAIMED, claim))
    monkeypatch.setattr(batch_tasks, "run_document_pipeline", pipeline)
    batch_tasks.run_batch_slot.run(running_item["batch"])
    assert seen == {"document_id": running_item["doc"], "job_id": running_item["job"], "max_workers": 3}


def test_no_capacity_requeues_itself_instead_of_holding_the_worker(monkeypatch):
    monkeypatch.setattr(batch_tasks, "_claim", lambda b, t: (ClaimOutcome.NO_CAPACITY, None))
    monkeypatch.setattr(batch_tasks, "run_document_pipeline", lambda *a: pytest.fail("must not run without a slot"))
    with pytest.raises(Retry):
        batch_tasks.run_batch_slot.run("b")


def test_an_exhausted_batch_is_a_noop(monkeypatch):
    monkeypatch.setattr(batch_tasks, "_claim", lambda b, t: (ClaimOutcome.EMPTY, None))
    monkeypatch.setattr(batch_tasks, "run_document_pipeline", lambda *a: pytest.fail("nothing to run"))
    assert batch_tasks.run_batch_slot.run("b") == "empty"


def test_a_killed_worker_marks_the_file_and_its_job_failed(Session, running_item):
    class FakeTask:
        name = "pipeline.run_batch_slot"

    batch_tasks._on_batch_task_failure(sender=FakeTask(), task_id="task-1", exception=RuntimeError("WorkerLostError"))
    state, error = _state(Session, running_item["item"])
    assert state == "failed" and "crashed" in error
    db = Session()
    try:
        assert db.get(ProcessingJob, running_item["job"]).status == DocumentStatus.FAILED
    finally:
        db.close()


def test_the_crash_handler_ignores_other_tasks_and_finished_files(Session, running_item):
    class Other:
        name = "pipeline.process_document"

    batch_tasks._on_batch_task_failure(sender=Other(), task_id="task-1", exception=RuntimeError("x"))
    assert _state(Session, running_item["item"])[0] == "running"

    db = Session()
    item = db.get(OCRBatchItem, running_item["item"])
    item.state = "completed"
    db.commit()
    db.close()

    class Mine:
        name = "pipeline.run_batch_slot"

    batch_tasks._on_batch_task_failure(sender=Mine(), task_id="task-1", exception=RuntimeError("x"))
    assert _state(Session, running_item["item"])[0] == "completed"  # untouched
