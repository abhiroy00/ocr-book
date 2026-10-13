"""
Regression tests for the queue-reliability fixes found by live testing
(2026-10-03, local docker stack):

  1. Cancelling a QUEUED document and then retrying it made the retry
     cancel itself after its first page (stale Redis cancel flag).
  2. Celery's Redis visibility_timeout (default 1h) was shorter than a
     long pipeline run, so a still-running job was redelivered and the copy
     marked it FAILED mid-run.
  3. Single-file uploads skipped the host's concurrency admission entirely
     (a second PaddleOCR document started its ~4.8GB worker with ~2.3GB
     free).
  4. On more than one OCR worker machine, each machine compared the GLOBAL
     running count against its OWN RAM-based limit, so extra machines never
     admitted batch work.

Redis is replaced by a small in-memory fake so these run without a broker.
"""
from __future__ import annotations

import fnmatch

import pytest
import redis

from app.core import locks
from app.core.config import Settings
from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum
from app.services import batch_scheduler, document_service
from app.services.batch_scheduler import HostResources
from app.workers import pipeline_tasks

EC2_SMALL = HostResources(cpu_count=2, total_mem_mb=7775)  # fits 1 PaddleOCR document
DEV_BIG = HostResources(cpu_count=14, total_mem_mb=11960)  # fits 2


class FakeRedis:
    """Just the commands the lock/cancel/scheduler code uses."""

    store: dict[str, str] = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def from_url(cls, *args, **kwargs) -> "FakeRedis":
        return cls()

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def get(self, key):
        v = self.store.get(key)
        return v.encode() if v is not None else None

    def mget(self, keys):
        return [self.get(k) for k in keys]

    def exists(self, key):
        return 1 if key in self.store else 0

    def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0

    def scan_iter(self, match="*", count=None):
        return [k.encode() for k in list(self.store) if fnmatch.fnmatch(k, match)]

    def eval(self, script, numkeys, key, token, *rest):
        if self.store.get(key) != token:
            return 0
        if "DEL" in script:
            del self.store[key]
        return 1

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    FakeRedis.store = {}
    monkeypatch.setattr(redis.Redis, "from_url", FakeRedis.from_url)
    return FakeRedis.store


def _settings(monkeypatch, **overrides) -> Settings:
    base = dict(ocr_reserved_cpu=1, ocr_reserved_memory_mb=1536, ocr_worker_est_memory_mb=4800, max_ai_concurrency=2)
    base.update(overrides)
    s = Settings(**base)
    for module in (batch_scheduler, pipeline_tasks):
        monkeypatch.setattr(module, "get_settings", lambda: s)
    return s


def _upload(db, storage, png, name="a.png"):
    return document_service.create_document_from_upload(
        db, storage, name, png, OCRProviderEnum.PADDLEOCR, 300, PreprocessProfileEnum.BALANCED
    )


# ------------------------------------------------- 1. stale cancel flag

def test_new_run_clears_a_cancel_flag_left_by_a_cancelled_queued_job(test_db_session, tmp_storage, sample_png_bytes, fake_redis):
    document, _ = _upload(test_db_session, tmp_storage, sample_png_bytes)
    # `/cancel` on a QUEUED job: flag set, task revoked, nothing ever consumes it.
    fake_redis[locks.DOCUMENT_CANCEL_PREFIX + document.id] = "1"

    document_service.create_processing_job(
        test_db_session, document, OCRProviderEnum.TESSERACT, 300, PreprocessProfileEnum.BALANCED
    )

    assert not pipeline_tasks._is_cancel_requested(document.id)


def test_new_run_keeps_the_flag_while_an_earlier_run_still_holds_the_lock(test_db_session, tmp_storage, sample_png_bytes, fake_redis):
    document, _ = _upload(test_db_session, tmp_storage, sample_png_bytes)
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + document.id] = "running-token"
    fake_redis[locks.DOCUMENT_CANCEL_PREFIX + document.id] = "1"

    document_service.create_processing_job(
        test_db_session, document, OCRProviderEnum.TESSERACT, 300, PreprocessProfileEnum.BALANCED
    )

    # A live run is still active -- the cancel is aimed at it and must survive.
    assert pipeline_tasks._is_cancel_requested(document.id)


def test_creating_a_job_survives_redis_being_down(test_db_session, tmp_storage, sample_png_bytes, monkeypatch):
    def unreachable(*a, **k):
        raise redis.exceptions.ConnectionError("down")

    monkeypatch.setattr(redis.Redis, "from_url", unreachable)
    document, job = _upload(test_db_session, tmp_storage, sample_png_bytes)
    assert job.status == DocumentStatus.QUEUED


# ------------------------------------------ 2. visibility timeout vs task length

def test_broker_visibility_timeout_outlasts_the_longest_task():
    from app.core.celery_app import celery_app

    visibility = celery_app.conf.broker_transport_options["visibility_timeout"]
    assert visibility > celery_app.conf.task_time_limit


# -------------------------------------------- 3. single-file admission

def test_single_file_waits_when_the_host_is_full(test_db_session, monkeypatch, fake_redis):
    _settings(monkeypatch)
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "already-running"] = "x"
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "new-doc"] = "y"  # the caller's own lock

    a = batch_scheduler.admit_single_document(test_db_session, "new-doc", "paddleocr", resources=EC2_SMALL)

    assert not a.admitted and a.in_use == 2 and a.limit == 1


def test_single_file_is_admitted_with_a_pool_cap_when_there_is_room(test_db_session, monkeypatch, fake_redis):
    _settings(monkeypatch)
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "already-running"] = "x"
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "new-doc"] = "y"

    a = batch_scheduler.admit_single_document(test_db_session, "new-doc", "paddleocr", resources=DEV_BIG)

    assert a.admitted and a.limit == 2
    assert a.pool_cap == 1  # two documents share the host's 2-worker budget


def test_pipeline_returns_no_capacity_releases_its_lock_and_leaves_the_job_queued(
    test_db_session, tmp_storage, sample_png_bytes, monkeypatch, fake_redis
):
    _settings(monkeypatch)
    document, job = _upload(test_db_session, tmp_storage, sample_png_bytes)
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "already-running"] = "x"
    monkeypatch.setattr(batch_scheduler, "detect_host_resources", lambda *a, **k: EC2_SMALL)
    monkeypatch.setattr(pipeline_tasks, "SessionLocal", lambda: test_db_session)
    monkeypatch.setattr(pipeline_tasks, "_run_pipeline", lambda *a, **k: pytest.fail("must not run without capacity"))

    outcome = pipeline_tasks.run_document_pipeline("task-1", document.id, job.id, admission_check=True)

    assert outcome == pipeline_tasks.NO_CAPACITY
    assert locks.DOCUMENT_LOCK_PREFIX + document.id not in fake_redis
    from app.models.processing_job import ProcessingJob

    fresh = test_db_session.get(ProcessingJob, job.id)  # the pipeline closed (detached) the session's objects
    assert fresh.status == DocumentStatus.QUEUED and fresh.error_message is None


def test_pipeline_runs_with_the_admitted_pool_cap(test_db_session, tmp_storage, sample_png_bytes, monkeypatch):
    _settings(monkeypatch)
    document, job = _upload(test_db_session, tmp_storage, sample_png_bytes)
    monkeypatch.setattr(batch_scheduler, "detect_host_resources", lambda *a, **k: DEV_BIG)
    monkeypatch.setattr(pipeline_tasks, "SessionLocal", lambda: test_db_session)
    seen = {}
    monkeypatch.setattr(pipeline_tasks, "_run_pipeline", lambda db, storage, d, j, token, cap: seen.setdefault("cap", cap) and False)

    assert pipeline_tasks.run_document_pipeline("task-1", document.id, job.id, admission_check=True) == "completed"
    assert "cap" in seen  # it ran (alone on the host: no cap needed beyond the default sizing)


def test_process_document_task_retries_instead_of_running(monkeypatch):
    _settings(monkeypatch, ocr_admission_retry_seconds=20)
    monkeypatch.setattr(pipeline_tasks, "run_document_pipeline", lambda *a, **k: pipeline_tasks.NO_CAPACITY)
    calls = {}

    def fake_retry(countdown):
        calls["countdown"] = countdown
        return RuntimeError("retry")

    monkeypatch.setattr(pipeline_tasks.process_document, "retry", fake_retry)
    with pytest.raises(RuntimeError):
        pipeline_tasks.process_document.run("doc", "job")
    assert 20 <= calls["countdown"] <= 40


# --------------------------------------- 4. per-machine counting (OCR_NODE_ID)

def test_lock_values_carry_the_node_id(monkeypatch, fake_redis):
    s = _settings(monkeypatch, ocr_node_id="i-aaa")
    monkeypatch.setattr(pipeline_tasks, "get_settings", lambda: s)

    token = pipeline_tasks._acquire_document_lock("doc-1")

    assert token.startswith("i-aaa|")
    assert fake_redis[locks.DOCUMENT_LOCK_PREFIX + "doc-1"] == token


def test_a_machine_only_counts_documents_running_on_itself(monkeypatch, fake_redis):
    _settings(monkeypatch, ocr_node_id="i-bbb")
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "on-a"] = "i-aaa|t1"
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "on-b"] = "i-bbb|t2"

    assert batch_scheduler.live_pipeline_document_ids() == {"on-b"}


def test_without_a_node_id_every_lock_counts_as_before(monkeypatch, fake_redis):
    _settings(monkeypatch)
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "on-a"] = "i-aaa|t1"
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "legacy"] = "plainhex"

    assert batch_scheduler.live_pipeline_document_ids() == {"on-a", "legacy"}


def test_an_idle_second_machine_admits_work_while_the_first_is_full(test_db_session, monkeypatch, fake_redis):
    _settings(monkeypatch, ocr_node_id="i-bbb")
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "busy-on-a"] = "i-aaa|t1"  # machine A is at its limit of 1
    fake_redis[locks.DOCUMENT_LOCK_PREFIX + "new-doc"] = "i-bbb|t2"

    a = batch_scheduler.admit_single_document(test_db_session, "new-doc", "paddleocr", resources=EC2_SMALL)

    assert a.admitted and a.in_use == 1
