"""
Scheduling for multiple-file (batch) OCR: how much work each file is, which
file runs next, and how many may run at once on THIS host.

Nothing here performs OCR -- a claimed file is handed to the existing
pipeline (`app.workers.pipeline_tasks`) unchanged. This module only decides
*when* and *with how much parallelism*.

Why concurrency is limited by resources, not by "more workers = faster"
(measured/confirmed in this codebase, see config.py and page_worker_pool.py):
  * PaddleOCR is CPU-bound native code and each OCR worker holds its own
    non-shareable model (~4.5GB with en+hi loaded). RAM, not CPU, is the
    ceiling: two concurrent Paddle documents on an 8GB host OOM it.
  * Tesseract is CPU-bound and light on RAM, so cores are the ceiling.
  * NVIDIA VLM is a remote HTTP call -- the only engine here where extra
    concurrency is nearly free locally, but bounded to avoid request
    flooding (MAX_AI_CONCURRENCY).
  * Ollama is a local model server that serialises/loads models itself, so
    one document at a time is the only safe default.
Threads/asyncio inside one process are deliberately NOT used to run
documents concurrently: every document forks a billiard page-worker pool,
and forking after a process has used threads is a confirmed, reproduced
hang in this codebase (see celery_app.py `worker_max_tasks_per_child`).
Document-level parallelism therefore stays where it already lives -- one
Celery prefork process per document.
"""
from __future__ import annotations

import enum
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.locks import DOCUMENT_LOCK_PREFIX
from app.core.logging import get_logger
from app.models.ocr_batch import OCRBatch, OCRBatchItem, BatchItemState

logger = get_logger(__name__)

# Between `claim` (item marked RUNNING) and the pipeline acquiring its own
# Redis document lock there is a short window (task start-up) in which the
# item is running but has no lock yet -- count it as active for this long so
# a second claimer cannot slip through and exceed the limit.
CLAIM_GRACE_SECONDS = 120

# Rough size of one page of a scanned PDF, used ONLY when a page count is not
# available (it normally is: PyMuPDF reads it from the PDF's page tree on
# upload validation without rendering anything).
_FALLBACK_BYTES_PER_PAGE = 150 * 1024


# ----------------------------------------------------------------------------
# Workload estimate + ordering policy
# ----------------------------------------------------------------------------

def estimate_workload(page_count: Optional[int], size_bytes: int) -> float:
    """Lightweight relative cost of one file -- lower runs earlier.

    Page count dominates (OCR time is per-page); file size is only a
    sub-page tiebreak (bytes -> GB, so even a 500MB file adds <1.0, never
    outweighing a whole page) and the fallback when the page count is
    unknown. No rendering is ever done to compute this."""
    if page_count and page_count > 0:
        pages = float(page_count)
    else:
        pages = max(1.0, size_bytes / _FALLBACK_BYTES_PER_PAGE)
    return pages + size_bytes / float(1024 ** 3)


def pick_next(queued: list[OCRBatchItem], running: list[OCRBatchItem], concurrency: int) -> Optional[OCRBatchItem]:
    """Which queued file a free slot should take.

    Policy ("small-first with one long lane"):
      * concurrency == 1: strictly smallest-first. This minimises the mean
        completion time; the makespan is identical for any order on one lane.
      * concurrency >= 2: smallest-first, EXCEPT that once something is
        already running and no "long" file (score >= the batch median) is in
        flight, the next slot takes the LARGEST queued file. Short files then
        keep finishing quickly in the remaining lane(s) while the long file
        starts early instead of becoming a long tail at the end -- the
        classic longest-first rule that minimises total batch time, without
        giving up early results for small files.
    Ties break on upload order so scheduling is deterministic."""
    if not queued:
        return None
    ordered = sorted(queued, key=lambda i: (i.priority_score, i.position))
    if concurrency >= 2 and len(ordered) > 1 and running:
        scores = [i.priority_score for i in ordered + running]
        median = statistics.median(scores)
        if not any(r.priority_score >= median for r in running):
            return ordered[-1]
    return ordered[0]


# ----------------------------------------------------------------------------
# Host resources (cgroup-aware) + engine-aware concurrency
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HostResources:
    cpu_count: int
    total_mem_mb: Optional[int]  # None if it could not be determined


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _cgroup_memory_limit_bytes(root: str) -> Optional[int]:
    v2 = _read_text(f"{root}/memory.max")
    if v2 is not None:
        return None if v2 == "max" else int(v2)
    v1 = _read_text(f"{root}/memory/memory.limit_in_bytes")
    if v1 is not None:
        n = int(v1)
        return n if n < (1 << 60) else None  # v1 "unlimited" is ~2^63
    return None


def _cgroup_cpu_limit(root: str) -> Optional[int]:
    v2 = _read_text(f"{root}/cpu.max")
    if v2 is not None:
        quota, _, period = v2.partition(" ")
        if quota == "max" or not period:
            return None
        return max(1, math.ceil(int(quota) / int(period)))
    quota = _read_text(f"{root}/cpu/cpu.cfs_quota_us")
    period = _read_text(f"{root}/cpu/cpu.cfs_period_us")
    if quota is not None and period is not None and int(quota) > 0 and int(period) > 0:
        return max(1, math.ceil(int(quota) / int(period)))
    return None


def detect_host_resources(cgroup_root: str = "/sys/fs/cgroup") -> HostResources:
    """CPU count and total RAM the OCR workers can actually use: the host
    figure, lowered to the container's cgroup limit when one is set (a
    `mem_limit`/`cpus` in docker-compose would otherwise be invisible to
    `os.cpu_count()`/psutil, which report the whole host)."""
    cpu = os.cpu_count() or 1
    try:
        limit = _cgroup_cpu_limit(cgroup_root)
        if limit:
            cpu = min(cpu, limit)
    except (ValueError, OSError):
        pass

    total_mb: Optional[int] = None
    try:
        import psutil

        total_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:  # noqa: BLE001 - degrade to "unknown", never fail admission over telemetry
        total_mb = None
    try:
        mem_limit = _cgroup_memory_limit_bytes(cgroup_root)
        if mem_limit:
            limit_mb = int(mem_limit / (1024 * 1024))
            total_mb = limit_mb if total_mb is None else min(total_mb, limit_mb)
    except (ValueError, OSError):
        pass
    return HostResources(cpu_count=cpu, total_mem_mb=total_mb)


@dataclass(frozen=True)
class ConcurrencyDecision:
    provider: str
    limit: int
    limiting_factor: str
    bounds: dict[str, int] = field(default_factory=dict)
    cpu_count: int = 0
    total_mem_mb: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "limit": self.limit,
            "limiting_factor": self.limiting_factor,
            "bounds": dict(self.bounds),
            "cpu_count": self.cpu_count,
            "total_mem_mb": self.total_mem_mb,
        }


def _ram_fit(total_mem_mb: Optional[int], per_job_mb: int, reserved_mb: int) -> Optional[int]:
    if total_mem_mb is None:
        return None
    return max(1, int((total_mem_mb - reserved_mb) // max(1, per_job_mb)))


def resolve_concurrency(provider_name: str, resources: Optional[HostResources] = None) -> ConcurrencyDecision:
    """How many documents may be inside the OCR pipeline at once for this
    engine on this host: the tightest of the engine's CPU/RAM/API bounds,
    further capped (never raised) by OCR_MAX_CONCURRENCY. Always >= 1 so a
    batch can always make forward progress."""
    settings = get_settings()
    res = resources or detect_host_resources()
    cpu_budget = max(1, res.cpu_count - settings.ocr_reserved_cpu)
    reserved = settings.ocr_reserved_memory_mb

    bounds: dict[str, int] = {}
    if provider_name == "paddleocr":
        # One pool worker minimum per document, each holding its own model.
        ram = _ram_fit(res.total_mem_mb, settings.ocr_worker_est_memory_mb, reserved)
        bounds["cpu"] = cpu_budget
        if ram is not None:
            bounds["ram"] = ram
    elif provider_name == "tesseract":
        ram = _ram_fit(res.total_mem_mb, settings.batch_light_engine_est_memory_mb, reserved)
        bounds["cpu"] = cpu_budget
        if ram is not None:
            bounds["ram"] = ram
    elif provider_name == "nvidia":
        # Remote inference: local CPU is barely used while waiting, so the
        # bound is the API, not the host.
        bounds["api"] = max(1, settings.max_ai_concurrency)
        ram = _ram_fit(res.total_mem_mb, settings.batch_light_engine_est_memory_mb, reserved)
        if ram is not None:
            bounds["ram"] = ram
    elif provider_name == "ollama":
        # Local model server: it loads/serialises models itself (and may
        # share CPU/GPU with everything else) -- one document at a time.
        bounds["engine"] = 1
    else:  # unknown engine: be conservative
        bounds["engine"] = 1

    if settings.ocr_max_concurrency > 0:
        bounds["configured_cap"] = settings.ocr_max_concurrency

    limit = max(1, min(bounds.values()))
    limiting = min(bounds, key=lambda k: bounds[k])
    return ConcurrencyDecision(
        provider=provider_name,
        limit=limit,
        limiting_factor=limiting,
        bounds=bounds,
        cpu_count=res.cpu_count,
        total_mem_mb=res.total_mem_mb,
    )


def pool_workers_per_job(provider_name: str, concurrency: int, resources: Optional[HostResources] = None) -> Optional[int]:
    """Cap on the page-worker pool size for EACH of `concurrency` documents
    running at once, so N documents together stay within the host's total
    OCR-worker budget instead of each one claiming the whole budget (which
    is exactly what would happen if every document sized its pool
    independently at start-up while the others' models were still loading).

    Returns None (no cap -- the existing per-document sizing applies
    unchanged) when only one document runs at a time."""
    if concurrency <= 1:
        return None
    settings = get_settings()
    res = resources or detect_host_resources()
    per_worker_mb = settings.ocr_worker_est_memory_mb if provider_name == "paddleocr" else settings.batch_light_engine_est_memory_mb
    requested = settings.ocr_workers if settings.ocr_workers > 0 else settings.ocr_max_workers
    total = min(requested, max(1, res.cpu_count - settings.ocr_reserved_cpu))
    ram = _ram_fit(res.total_mem_mb, per_worker_mb, settings.ocr_reserved_memory_mb)
    if ram is not None:
        total = min(total, ram)
    return max(1, total // concurrency)


# ----------------------------------------------------------------------------
# Admission + atomic claim
# ----------------------------------------------------------------------------

def live_pipeline_document_ids(redis_client=None) -> set[str]:
    """Documents that currently have a live pipeline lock. Every running
    pipeline -- single-file or batch -- holds one (see
    `pipeline_tasks._acquire_document_lock`, renewed after each page), so
    this counts *all* real work on the host, letting a batch item respect a
    single-file upload that is already running, not only other batch items."""
    import redis as redis_lib

    owns_client = redis_client is None
    client = redis_client or redis_lib.Redis.from_url(get_settings().redis_url, socket_connect_timeout=5, socket_timeout=5)
    try:
        prefix = DOCUMENT_LOCK_PREFIX
        return {
            (k.decode() if isinstance(k, bytes) else k)[len(prefix):]
            for k in client.scan_iter(match=prefix + "*", count=100)
        }
    except redis_lib.exceptions.RedisError as exc:
        logger.warning("batch_live_lock_scan_failed", error=str(exc))
        return set()
    finally:
        if owns_client:
            client.close()


def _serialize_claims(db: Session) -> None:
    """Makes the capacity-check-then-claim critical section atomic across
    Celery processes. Postgres advisory lock scoped to the transaction (it
    releases on the commit that persists the claim); a no-op elsewhere
    (SQLite in unit tests is single-connection)."""
    if db.get_bind().dialect.name == "postgresql":
        db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": 0x0C7A5C4ED})


@dataclass(frozen=True)
class Claim:
    item_id: str
    batch_id: str
    document_id: str
    job_id: str
    provider: str
    limit: int
    pool_cap: Optional[int]


class ClaimOutcome(str, enum.Enum):
    CLAIMED = "claimed"
    NO_CAPACITY = "no_capacity"  # something is queued, but the host is full -- retry later
    EMPTY = "empty"  # nothing left to claim in this batch


def claim_next_item(
    db: Session,
    batch_id: str,
    task_id: Optional[str],
    provider_name: str,
    *,
    redis_client=None,
    resources: Optional[HostResources] = None,
    now: Optional[datetime] = None,
) -> tuple[ClaimOutcome, Optional[Claim]]:
    """Atomically picks the next file for this batch and marks it RUNNING,
    unless the host is already at its concurrency limit."""
    _serialize_claims(db)
    now = now or datetime.now(timezone.utc)

    batch = db.get(OCRBatch, batch_id)
    if batch is None:
        db.rollback()
        return ClaimOutcome.EMPTY, None

    items = list(db.scalars(select(OCRBatchItem).where(OCRBatchItem.batch_id == batch_id)).all())
    queued = [i for i in items if i.state == BatchItemState.QUEUED.value]
    if not queued:
        db.rollback()
        return ClaimOutcome.EMPTY, None

    decision = resolve_concurrency(provider_name, resources)

    # Everything currently doing OCR on this host: live pipeline locks, plus
    # anything claimed a moment ago that hasn't taken its lock yet.
    live_docs = live_pipeline_document_ids(redis_client)
    grace_cutoff = now - timedelta(seconds=CLAIM_GRACE_SECONDS)
    recently_claimed = db.scalars(
        select(OCRBatchItem.document_id).where(
            OCRBatchItem.state == BatchItemState.RUNNING.value, OCRBatchItem.claimed_at >= grace_cutoff
        )
    ).all()
    in_use = len(live_docs | set(recently_claimed))
    if in_use >= decision.limit:
        db.rollback()
        return ClaimOutcome.NO_CAPACITY, None

    running = [i for i in items if i.state == BatchItemState.RUNNING.value]
    chosen = pick_next(queued, running, decision.limit)
    if chosen is None:
        db.rollback()
        return ClaimOutcome.EMPTY, None

    chosen.state = BatchItemState.RUNNING.value
    chosen.claimed_at = now
    chosen.celery_task_id = task_id
    chosen.attempts = (chosen.attempts or 0) + 1
    chosen.error_message = None
    db.commit()

    claim = Claim(
        item_id=chosen.id,
        batch_id=batch_id,
        document_id=chosen.document_id,
        job_id=chosen.processing_job_id or "",
        provider=provider_name,
        limit=decision.limit,
        pool_cap=pool_workers_per_job(provider_name, decision.limit, resources),
    )
    return ClaimOutcome.CLAIMED, claim
