"""
End-to-end tests of the multiple-file OCR API (routes -> batch_service ->
existing document_service), against in-memory SQLite and local tmp storage.
The Celery broker is replaced by a recorder, so these verify what would be
queued, not the OCR itself (which is the unchanged single-file pipeline).
"""
import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db, get_storage_backend
from app.core.celery_app import celery_app
from app.db.base import Base
from app.models.document_page import DocumentPage
from app.models.enums import DocumentStatus, ProcessingStage
from app.models.ocr_batch import BatchItemState, OCRBatchItem
from app.models.processing_job import ProcessingJob
from app.services import document_service
from app.services.storage import LocalStorageBackend


def make_pdf(pages: int, tag: str = "x") -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{tag}-{i}")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from main import app

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    storage = LocalStorageBackend(str(tmp_path / "storage"))
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_backend] = lambda: storage

    sent: list = []
    monkeypatch.setattr(celery_app, "send_task", lambda name, args=None, **kw: sent.append((name, args)))

    with TestClient(app) as c:
        c.sent = sent
        c.Session = Session
        yield c
    app.dependency_overrides.clear()
    engine.dispose()


def new_batch(client, **body) -> str:
    payload = {"ocr_provider": "tesseract", "dpi": 150, "preprocess_profile": "FAST", **body}
    r = client.post("/api/batches", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["batch_id"]


def upload(client, batch_id, name, pages, tag=None, content=None, **form):
    # PyMuPDF stamps a random document ID into every PDF it writes, so two
    # calls to make_pdf never produce identical bytes -- a test that needs
    # byte-identical files must pass the same `content` to both uploads.
    return client.post(
        f"/api/batches/{batch_id}/files",
        files={"file": (name, content if content is not None else make_pdf(pages, tag or name), "application/pdf")},
        data=form,
    )


def set_job(client, job_id, status, error=None, stage=ProcessingStage.DONE, percent=100):
    db = client.Session()
    try:
        job = db.get(ProcessingJob, job_id)
        document_service.update_job_progress(db, job, stage, percent, status, error=error)
    finally:
        db.close()


def status(client, batch_id):
    r = client.get(f"/api/batches/{batch_id}")
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ creation

def test_create_batch_applies_settings_and_starts_in_uploading_state(client):
    bid = new_batch(client, dpi=200, preprocess_profile="BALANCED")
    s = status(client, bid)
    assert s["status"] == "uploading"
    assert s["dpi"] == 200 and s["preprocess_profile"] == "BALANCED" and s["ocr_provider"] == "tesseract"
    assert s["total_files"] == 0 and s["items"] == []


def test_create_batch_rejects_a_dpi_the_single_file_flow_also_rejects(client):
    r = client.post("/api/batches", json={"dpi": 999})
    assert r.status_code == 422


def test_unknown_batch_is_404_everywhere(client):
    assert client.get("/api/batches/nope").status_code == 404
    assert client.post("/api/batches/nope/start").status_code == 404
    assert client.post("/api/batches/nope/cancel").status_code == 404
    assert upload(client, "nope", "a.pdf", 1).status_code == 404


def test_capacity_endpoint_reports_limits_and_a_positive_concurrency(client):
    r = client.get("/api/batches/capacity", params={"ocr_provider": "paddleocr"})
    assert r.status_code == 200
    body = r.json()
    assert body["concurrency"]["limit"] >= 1
    assert body["max_files"] >= 1 and ".pdf" in body["allowed_extensions"]


# ------------------------------------------------------------------- uploads

def test_each_file_becomes_a_normal_document_with_a_page_count(client):
    bid = new_batch(client)
    r = upload(client, bid, "invoice.pdf", 3)
    assert r.status_code == 201, r.text
    item = r.json()
    assert item["filename"] == "invoice.pdf" and item["total_pages"] == 3 and item["state"] == "pending"
    doc = client.get(f"/api/documents/{item['document_id']}")
    assert doc.status_code == 200 and doc.json()["page_count"] == 3  # the existing documents API sees it


def test_response_never_exposes_filesystem_paths(client):
    bid = new_batch(client)
    upload(client, bid, "a.pdf", 1)
    text = client.get(f"/api/batches/{bid}").text
    assert "storage" not in text.lower() and "original/" not in text


def test_hostile_filenames_are_sanitized(client):
    bid = new_batch(client)
    r = client.post(
        f"/api/batches/{bid}/files", files={"file": ("../../etc/passwd.pdf", make_pdf(1), "application/pdf")}
    )
    assert r.status_code == 201
    assert "/" not in r.json()["filename"] and ".." not in r.json()["filename"]


def test_disallowed_extension_is_rejected(client):
    bid = new_batch(client)
    r = client.post(f"/api/batches/{bid}/files", files={"file": ("run.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422 and "Unsupported" in r.json()["detail"]


def test_a_pdf_that_is_not_really_a_pdf_is_rejected_by_content(client):
    bid = new_batch(client)
    r = client.post(f"/api/batches/{bid}/files", files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 422


def test_empty_file_is_rejected(client):
    bid = new_batch(client)
    assert client.post(f"/api/batches/{bid}/files", files={"file": ("e.pdf", b"", "application/pdf")}).status_code == 422


def test_file_count_limit(client, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "batch_max_files", 2)
    bid = new_batch(client)
    assert upload(client, bid, "a.pdf", 1).status_code == 201
    assert upload(client, bid, "b.pdf", 1).status_code == 201
    r = upload(client, bid, "c.pdf", 1)
    assert r.status_code == 422 and "at most 2" in r.json()["detail"]


def test_total_size_limit(client, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "batch_max_total_mb", 0)
    bid = new_batch(client)
    assert upload(client, bid, "a.pdf", 1).status_code == 413


def test_insufficient_disk_space_is_refused_with_507(client, monkeypatch):
    import collections

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr("app.services.batch_service.shutil.disk_usage", lambda p: usage(10**12, 10**12 - 10**6, 10**6))
    bid = new_batch(client)
    assert upload(client, bid, "a.pdf", 1).status_code == 507


def test_identical_content_is_reported_as_a_duplicate_not_reprocessed(client):
    bid = new_batch(client)
    same = make_pdf(2, "same")
    first = upload(client, bid, "a.pdf", 2, content=same).json()
    dup = upload(client, bid, "again.pdf", 2, content=same)
    assert dup.status_code == 201
    assert dup.json()["state"] == "duplicate" and dup.json()["document_id"] == first["document_id"]
    forced = upload(client, bid, "again.pdf", 2, content=same, force="true").json()
    assert forced["state"] == "pending" and forced["document_id"] != first["document_id"]


def test_cannot_add_files_after_start(client):
    bid = new_batch(client)
    upload(client, bid, "a.pdf", 1)
    assert client.post(f"/api/batches/{bid}/start").status_code == 200
    assert upload(client, bid, "late.pdf", 1).status_code == 409


# ------------------------------------------------------------------- start

def test_start_requires_at_least_one_file(client):
    bid = new_batch(client)
    assert client.post(f"/api/batches/{bid}/start").status_code == 422


def test_start_queues_one_slot_per_file_and_scores_small_files_lower(client):
    bid = new_batch(client)
    for name, pages in [("big.pdf", 150), ("tiny.pdf", 2), ("mid.pdf", 12), ("small.pdf", 5)]:
        assert upload(client, bid, name, pages).status_code == 201
    r = client.post(f"/api/batches/{bid}/start")
    assert r.status_code == 200 and r.json()["queued_files"] == 4
    assert client.sent == [("pipeline.run_batch_slot", [bid])] * 4

    items = status(client, bid)["items"]
    by_score = [i["filename"] for i in sorted(items, key=lambda i: i["priority_score"])]
    assert by_score == ["tiny.pdf", "small.pdf", "mid.pdf", "big.pdf"]  # regardless of upload order
    assert {i["state"] for i in items} == {"queued"}


def test_start_is_idempotent(client):
    bid = new_batch(client)
    upload(client, bid, "a.pdf", 1)
    client.post(f"/api/batches/{bid}/start")
    again = client.post(f"/api/batches/{bid}/start")
    assert again.status_code == 200 and again.json()["queued_files"] == 0
    assert len(client.sent) == 1  # no duplicate tasks


def test_a_broker_outage_at_start_is_reverted_so_start_can_be_retried(client, monkeypatch):
    bid = new_batch(client)
    upload(client, bid, "a.pdf", 1)

    def boom(*a, **k):
        raise ConnectionError("redis down")

    monkeypatch.setattr(celery_app, "send_task", boom)
    assert client.post(f"/api/batches/{bid}/start").status_code == 503
    s = status(client, bid)
    assert s["status"] == "uploading" and s["items"][0]["state"] == "pending"
    monkeypatch.setattr(celery_app, "send_task", lambda n, args=None, **k: client.sent.append((n, args)))
    assert client.post(f"/api/batches/{bid}/start").json()["queued_files"] == 1


# ------------------------------------------------- status, isolation, retry

def test_one_failed_file_does_not_affect_the_others(client):
    bid = new_batch(client)
    ids = {n: upload(client, bid, n, p).json() for n, p in [("a.pdf", 2), ("b.pdf", 3), ("c.pdf", 4)]}
    client.post(f"/api/batches/{bid}/start")

    set_job(client, ids["a.pdf"]["processing_job_id"], DocumentStatus.COMPLETED)
    set_job(client, ids["b.pdf"]["processing_job_id"], DocumentStatus.FAILED, error="ocr exploded")

    s = status(client, bid)
    states = {i["filename"]: i["state"] for i in s["items"]}
    assert states == {"a.pdf": "completed", "b.pdf": "failed", "c.pdf": "queued"}
    assert s["status"] == "processing"  # c still pending -> batch is not done
    failed = next(i for i in s["items"] if i["filename"] == "b.pdf")
    assert failed["error_message"] == "ocr exploded"
    assert (s["completed_files"], s["failed_files"], s["queued_files"], s["finished_files"]) == (1, 1, 1, 2)

    set_job(client, ids["c.pdf"]["processing_job_id"], DocumentStatus.COMPLETED)
    s = status(client, bid)
    assert s["status"] == "completed_with_errors" and s["finished_files"] == 3


def test_a_batch_of_only_duplicates_shows_100_percent_not_0(client):
    # An already-completed document from a prior batch...
    prior_bid = new_batch(client)
    same = make_pdf(2, "same")
    prior = upload(client, prior_bid, "a.pdf", 2, content=same).json()
    client.post(f"/api/batches/{prior_bid}/start")
    set_job(client, prior["processing_job_id"], DocumentStatus.COMPLETED)

    # ...re-uploaded twice in a brand new batch: every item in THIS batch is a duplicate.
    bid = new_batch(client)
    upload(client, bid, "b.pdf", 2, content=same)
    upload(client, bid, "c.pdf", 2, content=same)
    client.post(f"/api/batches/{bid}/start")
    s = status(client, bid)
    assert all(i["state"] == "duplicate" for i in s["items"])
    assert s["status"] == "completed" and s["overall_percent"] == 100 and s["total_pages"] == 0


def test_all_files_completed_marks_the_batch_completed(client):
    bid = new_batch(client)
    a = upload(client, bid, "a.pdf", 2).json()
    client.post(f"/api/batches/{bid}/start")
    set_job(client, a["processing_job_id"], DocumentStatus.COMPLETED)
    s = status(client, bid)
    assert s["status"] == "completed" and s["overall_percent"] == 100 and s["elapsed_seconds"] is not None
    assert s["items"][0]["processed_pages"] == 2 == s["items"][0]["total_pages"]


def test_processed_page_progress_for_a_running_file(client):
    bid = new_batch(client)
    item = upload(client, bid, "long.pdf", 12).json()
    client.post(f"/api/batches/{bid}/start")

    db = client.Session()
    try:
        row = db.get(OCRBatchItem, item["job_id"])
        row.state = BatchItemState.RUNNING.value
        db.commit()
        job = db.get(ProcessingJob, item["processing_job_id"])
        document_service.update_job_progress(db, job, ProcessingStage.OCR, 30, DocumentStatus.PROCESSING)
        for n in (1, 2, 3):
            db.add(DocumentPage(document_id=item["document_id"], page_number=n, width=1, height=1, dpi=150, original_image_path="p"))
        db.commit()
    finally:
        db.close()

    it = status(client, bid)["items"][0]
    assert (it["processed_pages"], it["total_pages"]) == (3, 12)
    assert it["state"] == "running" and it["stage"] == "ocr"


def test_a_retry_is_not_credited_with_pages_from_the_failed_attempt(client):
    bid = new_batch(client)
    item = upload(client, bid, "a.pdf", 6).json()
    client.post(f"/api/batches/{bid}/start")
    db = client.Session()
    try:
        job = db.get(ProcessingJob, item["processing_job_id"])
        document_service.update_job_progress(db, job, ProcessingStage.OCR, 30, DocumentStatus.PROCESSING)
        for n in (1, 2, 3, 4):
            db.add(DocumentPage(document_id=item["document_id"], page_number=n, width=1, height=1, dpi=150, original_image_path="p"))
        db.commit()
        document_service.update_job_progress(db, job, ProcessingStage.OCR, 30, DocumentStatus.FAILED, error="x")
    finally:
        db.close()
    assert status(client, bid)["items"][0]["state"] == "failed"

    r = client.post(f"/api/batches/{bid}/items/{item['job_id']}/retry")
    assert r.status_code == 200
    again = r.json()
    assert again["state"] == "queued" and again["processing_job_id"] != item["processing_job_id"]
    assert again["processed_pages"] == 0  # new job, nothing started yet
    assert client.sent[-1] == ("pipeline.run_batch_slot", [bid])


def test_only_failed_or_cancelled_files_can_be_retried(client):
    bid = new_batch(client)
    a = upload(client, bid, "a.pdf", 1).json()
    client.post(f"/api/batches/{bid}/start")
    assert client.post(f"/api/batches/{bid}/items/{a['job_id']}/retry").status_code == 409
    assert client.post(f"/api/batches/{bid}/items/nope/retry").status_code == 404


def test_retry_only_requeues_that_one_file(client):
    bid = new_batch(client)
    a = upload(client, bid, "a.pdf", 1).json()
    b = upload(client, bid, "b.pdf", 2).json()
    client.post(f"/api/batches/{bid}/start")
    set_job(client, a["processing_job_id"], DocumentStatus.COMPLETED)
    set_job(client, b["processing_job_id"], DocumentStatus.FAILED, error="x")
    status(client, bid)  # sync
    before = len(client.sent)
    client.post(f"/api/batches/{bid}/items/{b['job_id']}/retry")
    states = {i["filename"]: i["state"] for i in status(client, bid)["items"]}
    assert states == {"a.pdf": "completed", "b.pdf": "queued"} and len(client.sent) == before + 1


def test_item_status_endpoint(client):
    bid = new_batch(client)
    a = upload(client, bid, "a.pdf", 1).json()
    r = client.get(f"/api/batches/{bid}/items/{a['job_id']}")
    assert r.status_code == 200 and r.json()["filename"] == "a.pdf"
    assert client.get(f"/api/batches/{bid}/items/nope").status_code == 404


# -------------------------------------------------------------------- cancel

def test_cancel_drops_waiting_files_and_cancels_their_jobs(client, monkeypatch):
    calls = []
    monkeypatch.setattr("app.api.v1.batches.cancel_processing", lambda doc_id, db: calls.append(doc_id))
    bid = new_batch(client)
    ids = [upload(client, bid, f"{n}.pdf", 1).json() for n in "abc"]
    client.post(f"/api/batches/{bid}/start")

    # one file already in OCR
    db = client.Session()
    try:
        db.get(OCRBatchItem, ids[0]["job_id"]).state = BatchItemState.RUNNING.value
        db.commit()
    finally:
        db.close()

    r = client.post(f"/api/batches/{bid}/cancel")
    assert r.status_code == 200
    assert r.json()["cancelled_waiting"] == 2 and r.json()["cancelling_running"] == 1
    assert calls == [ids[0]["document_id"]]  # the running one goes through the existing cooperative cancel

    s = status(client, bid)
    waiting = [i for i in s["items"] if i["job_id"] != ids[0]["job_id"]]
    assert {i["state"] for i in waiting} == {"cancelled"}
    assert {i["status"] for i in waiting} == {"CANCELLED"}
