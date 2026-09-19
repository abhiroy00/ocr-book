"""
Unit tests for multiple-file OCR scheduling (`app.services.batch_scheduler`):
workload estimate, ordering policy, cgroup-aware resource detection, the
engine-aware concurrency limit, and the atomic claim step.

Host sizes below are the two real machines this project runs on: the
production EC2 `m7i-flex.large` (2 vCPU, ~7.8GB, no GPU) and a 14-core/12GB
developer Docker VM -- the same OCR settings must resolve to a very
different concurrency on each, which is the point of resolving it at all.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.models.enums import DocumentStatus, OCRProviderEnum, PreprocessProfileEnum
from app.models.ocr_batch import BatchItemState, OCRBatch, OCRBatchItem
from app.services import batch_scheduler
from app.services.batch_scheduler import ClaimOutcome, HostResources

EC2_SMALL = HostResources(cpu_count=2, total_mem_mb=7775)
DEV_BIG = HostResources(cpu_count=14, total_mem_mb=11960)


def _settings(monkeypatch, **overrides) -> Settings:
    # Mirrors the production .env for the fields that matter here.
    base = dict(ocr_reserved_cpu=1, ocr_reserved_memory_mb=1536, ocr_worker_est_memory_mb=4800, max_ai_concurrency=2)
    base.update(overrides)
    s = Settings(**base)
    monkeypatch.setattr(batch_scheduler, "get_settings", lambda: s)
    return s


def _item(score: float, position: int, state: str = BatchItemState.QUEUED.value, doc: str | None = None) -> OCRBatchItem:
    return OCRBatchItem(
        id=f"item-{position}", batch_id="b", document_id=doc or f"doc-{position}", position=position,
        state=state, priority_score=score,
    )


# ---------------------------------------------------------------- workload

def test_estimate_workload_orders_by_pages_not_size():
    small_pages_big_file = batch_scheduler.estimate_workload(3, 400 * 1024 * 1024)
    more_pages_small_file = batch_scheduler.estimate_workload(4, 1024)
    assert small_pages_big_file < more_pages_small_file


def test_estimate_workload_size_is_a_tiebreak_for_equal_pages():
    assert batch_scheduler.estimate_workload(10, 1_000_000) < batch_scheduler.estimate_workload(10, 90_000_000)


def test_estimate_workload_falls_back_to_size_when_page_count_unknown():
    assert batch_scheduler.estimate_workload(None, 150 * 1024 * 20) == pytest.approx(20, abs=0.1)
    assert batch_scheduler.estimate_workload(0, 10) >= 1.0


# ----------------------------------------------------------------- ordering

def test_pick_next_single_lane_is_strictly_smallest_first():
    queued = [_item(150, 0), _item(2, 1), _item(80, 2), _item(5, 3)]
    order = []
    while queued:
        chosen = batch_scheduler.pick_next(queued, [], concurrency=1)
        order.append(chosen.priority_score)
        queued.remove(chosen)
    assert order == [2, 5, 80, 150]


def test_pick_next_ties_break_on_upload_order():
    queued = [_item(5, 2), _item(5, 0), _item(5, 1)]
    assert batch_scheduler.pick_next(queued, [], 1).position == 0


def test_pick_next_empty_returns_none():
    assert batch_scheduler.pick_next([], [], 2) is None


def test_pick_next_two_lanes_first_claim_is_still_smallest():
    queued = [_item(150, 0), _item(2, 1), _item(80, 2)]
    assert batch_scheduler.pick_next(queued, [], concurrency=2).priority_score == 2


def test_pick_next_two_lanes_starts_the_longest_file_early():
    """Once a short file is running and nothing long is, the free slot takes
    the LARGEST queued file so it is not left as a tail at the very end."""
    running = [_item(2, 1, BatchItemState.RUNNING.value)]
    queued = [_item(150, 0), _item(80, 2), _item(5, 3)]
    assert batch_scheduler.pick_next(queued, running, concurrency=2).priority_score == 150


def test_pick_next_two_lanes_keeps_small_files_flowing_while_a_long_one_runs():
    running = [_item(150, 0, BatchItemState.RUNNING.value)]
    queued = [_item(80, 2), _item(5, 3), _item(9, 4)]
    assert batch_scheduler.pick_next(queued, running, concurrency=2).priority_score == 5


# --------------------------------------------------- resources / concurrency

def test_cgroup_v2_limits(tmp_path):
    (tmp_path / "memory.max").write_text("4294967296\n")
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    assert batch_scheduler._cgroup_memory_limit_bytes(str(tmp_path)) == 4 * 1024**3
    assert batch_scheduler._cgroup_cpu_limit(str(tmp_path)) == 2


def test_cgroup_v2_unlimited_is_none(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert batch_scheduler._cgroup_memory_limit_bytes(str(tmp_path)) is None
    assert batch_scheduler._cgroup_cpu_limit(str(tmp_path)) is None


def test_cgroup_v1_limits(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("2147483648\n")
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("150000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert batch_scheduler._cgroup_memory_limit_bytes(str(tmp_path)) == 2 * 1024**3
    assert batch_scheduler._cgroup_cpu_limit(str(tmp_path)) == 2  # ceil(1.5)


def test_cgroup_v1_unlimited_sentinel_is_none(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert batch_scheduler._cgroup_memory_limit_bytes(str(tmp_path)) is None


def test_detect_host_resources_applies_a_container_memory_limit(tmp_path):
    (tmp_path / "memory.max").write_text(str(512 * 1024 * 1024))
    res = batch_scheduler.detect_host_resources(str(tmp_path))
    assert res.total_mem_mb <= 512


def test_paddle_is_one_at_a_time_on_the_small_ec2_host(monkeypatch):
    """4.8GB per Paddle worker does not fit twice in 7.8GB -- RAM, not CPU,
    is the binding limit, so file-level concurrency must stay at 1."""
    _settings(monkeypatch)
    d = batch_scheduler.resolve_concurrency("paddleocr", EC2_SMALL)
    assert d.limit == 1
    assert d.limiting_factor in ("ram", "cpu")


def test_paddle_gets_two_on_a_bigger_host(monkeypatch):
    _settings(monkeypatch)
    d = batch_scheduler.resolve_concurrency("paddleocr", DEV_BIG)
    assert d.limit == 2 and d.limiting_factor == "ram"


def test_configured_cap_only_lowers_never_raises(monkeypatch):
    _settings(monkeypatch, ocr_max_concurrency=1)
    assert batch_scheduler.resolve_concurrency("paddleocr", DEV_BIG).limit == 1
    _settings(monkeypatch, ocr_max_concurrency=50)
    assert batch_scheduler.resolve_concurrency("paddleocr", EC2_SMALL).limit == 1  # asking for 50 does not OOM the box


def test_tesseract_is_cpu_bound(monkeypatch):
    _settings(monkeypatch)
    assert batch_scheduler.resolve_concurrency("tesseract", EC2_SMALL).limit == 1  # 2 vCPU - 1 reserved
    assert batch_scheduler.resolve_concurrency("tesseract", DEV_BIG).limit == 13   # RAM allows 14, cores allow 13


def test_remote_engine_uses_the_api_bound_not_the_cpu_count(monkeypatch):
    _settings(monkeypatch)
    d = batch_scheduler.resolve_concurrency("nvidia", EC2_SMALL)
    assert d.limit == 2 and d.limiting_factor == "api"


def test_ollama_and_unknown_engines_run_one_at_a_time(monkeypatch):
    _settings(monkeypatch)
    assert batch_scheduler.resolve_concurrency("ollama", DEV_BIG).limit == 1
    assert batch_scheduler.resolve_concurrency("something-new", DEV_BIG).limit == 1


def test_unknown_total_memory_still_yields_a_usable_limit(monkeypatch):
    _settings(monkeypatch)
    d = batch_scheduler.resolve_concurrency("paddleocr", HostResources(cpu_count=4, total_mem_mb=None))
    assert d.limit >= 1 and "ram" not in d.bounds


def test_pool_cap_is_none_for_a_single_document(monkeypatch):
    _settings(monkeypatch)
    assert batch_scheduler.pool_workers_per_job("paddleocr", 1, DEV_BIG) is None


def test_pool_cap_splits_the_worker_budget_across_concurrent_documents(monkeypatch):
    _settings(monkeypatch)
    # dev host budget: min(10 requested, 13 cpu, 2 by RAM) = 2 workers total -> 1 each for 2 documents
    assert batch_scheduler.pool_workers_per_job("paddleocr", 2, DEV_BIG) == 1
    _settings(monkeypatch, ocr_worker_est_memory_mb=1000)
    assert batch_scheduler.pool_workers_per_job("paddleocr", 2, DEV_BIG) == 5  # 10 workers / 2 documents


# --------------------------------------------------------------------- claim

@pytest.fixture()
def batch_with_items(test_db_session):
    """A started batch whose documents are real rows (FKs are enforced)."""
    from app.models.document import Document

    db = test_db_session
    batch = OCRBatch(ocr_provider="tesseract", dpi=150, preprocess_profile="FAST", started_at=datetime.now(timezone.utc))
    db.add(batch)
    db.flush()
    items = []
    for pos, score in enumerate([50.0, 2.0, 9.0]):
        doc = Document(
            original_filename=f"f{pos}.pdf", file_extension=".pdf", mime_type="application/pdf", file_size_bytes=10,
            page_count=int(score), status=DocumentStatus.QUEUED, ocr_provider=OCRProviderEnum.TESSERACT, dpi=150,
            preprocess_profile=PreprocessProfileEnum.FAST, storage_original_path="x",
        )
        db.add(doc)
        db.flush()
        item = OCRBatchItem(
            batch_id=batch.id, document_id=doc.id, position=pos, state=BatchItemState.QUEUED.value, priority_score=score
        )
        db.add(item)
        items.append(item)
    db.commit()
    return batch, items


def _redis_with_locks(*doc_ids: str) -> MagicMock:
    client = MagicMock()
    client.scan_iter.return_value = [f"pipeline:lock:document:{d}".encode() for d in doc_ids]
    return client


def test_claim_takes_the_smallest_file_and_marks_it_running(test_db_session, batch_with_items, monkeypatch):
    _settings(monkeypatch)
    batch, items = batch_with_items
    outcome, claim = batch_scheduler.claim_next_item(
        test_db_session, batch.id, "task-1", "tesseract", redis_client=_redis_with_locks(), resources=DEV_BIG
    )
    assert outcome is ClaimOutcome.CLAIMED
    assert claim.item_id == items[1].id  # the 2-page file
    test_db_session.refresh(items[1])
    assert items[1].state == BatchItemState.RUNNING.value
    assert items[1].celery_task_id == "task-1" and items[1].attempts == 1


def test_claim_reports_no_capacity_when_the_host_is_full(test_db_session, batch_with_items, monkeypatch):
    _settings(monkeypatch)
    batch, items = batch_with_items
    outcome, claim = batch_scheduler.claim_next_item(
        test_db_session, batch.id, "t", "paddleocr", redis_client=_redis_with_locks("some-other-doc"), resources=EC2_SMALL
    )
    assert outcome is ClaimOutcome.NO_CAPACITY and claim is None
    assert all(test_db_session.get(OCRBatchItem, i.id).state == BatchItemState.QUEUED.value for i in items)  # nothing claimed


def test_claim_counts_a_just_claimed_file_that_has_no_lock_yet(test_db_session, batch_with_items, monkeypatch):
    """The window between claim and the pipeline taking its Redis lock must
    still count as 'in use', or two slots could both claim past the limit."""
    _settings(monkeypatch)
    batch, items = batch_with_items
    first = batch_scheduler.claim_next_item(test_db_session, batch.id, "t1", "paddleocr", redis_client=_redis_with_locks(), resources=EC2_SMALL)
    assert first[0] is ClaimOutcome.CLAIMED
    second = batch_scheduler.claim_next_item(test_db_session, batch.id, "t2", "paddleocr", redis_client=_redis_with_locks(), resources=EC2_SMALL)
    assert second[0] is ClaimOutcome.NO_CAPACITY


def test_claim_stops_counting_a_stale_claim_after_the_grace_window(test_db_session, batch_with_items, monkeypatch):
    _settings(monkeypatch)
    batch, items = batch_with_items
    batch_scheduler.claim_next_item(test_db_session, batch.id, "t1", "paddleocr", redis_client=_redis_with_locks(), resources=EC2_SMALL)
    later = datetime.now(timezone.utc) + timedelta(seconds=batch_scheduler.CLAIM_GRACE_SECONDS + 5)
    outcome, _ = batch_scheduler.claim_next_item(
        test_db_session, batch.id, "t2", "paddleocr", redis_client=_redis_with_locks(), resources=EC2_SMALL, now=later
    )
    assert outcome is ClaimOutcome.CLAIMED  # the first claim died without ever taking a lock -> its slot is free again


def test_claim_honours_a_single_file_upload_already_running(test_db_session, batch_with_items, monkeypatch):
    """A running single-file pipeline holds the same Redis lock a batch item
    does, so it counts against the limit too."""
    _settings(monkeypatch)
    batch, items = batch_with_items
    outcome, _ = batch_scheduler.claim_next_item(
        test_db_session, batch.id, "t", "paddleocr", redis_client=_redis_with_locks("a-single-upload"), resources=EC2_SMALL
    )
    assert outcome is ClaimOutcome.NO_CAPACITY


def test_claim_is_empty_when_nothing_is_queued(test_db_session, batch_with_items, monkeypatch):
    _settings(monkeypatch)
    batch, items = batch_with_items
    for i in items:
        i.state = BatchItemState.COMPLETED.value
    test_db_session.commit()
    outcome, claim = batch_scheduler.claim_next_item(test_db_session, batch.id, "t", "tesseract", redis_client=_redis_with_locks(), resources=DEV_BIG)
    assert outcome is ClaimOutcome.EMPTY and claim is None


def test_claim_on_an_unknown_batch_is_empty(test_db_session, monkeypatch):
    _settings(monkeypatch)
    outcome, _ = batch_scheduler.claim_next_item(test_db_session, "nope", "t", "tesseract", redis_client=_redis_with_locks(), resources=DEV_BIG)
    assert outcome is ClaimOutcome.EMPTY


def test_two_lane_claims_start_small_then_the_longest(test_db_session, batch_with_items, monkeypatch):
    _settings(monkeypatch)
    batch, items = batch_with_items  # scores 50, 2, 9
    r = _redis_with_locks()
    first = batch_scheduler.claim_next_item(test_db_session, batch.id, "t1", "tesseract", redis_client=r, resources=DEV_BIG)[1]
    second = batch_scheduler.claim_next_item(test_db_session, batch.id, "t2", "tesseract", redis_client=r, resources=DEV_BIG)[1]
    assert first.item_id == items[1].id   # 2 pages first
    assert second.item_id == items[0].id  # then the 50-page file starts early, ahead of the 9-page one
    assert second.pool_cap is not None and first.limit > 1
