"""
Celery application factory.

Workers register tasks in `app.workers.*`. The pipeline is a chain of small
tasks (one per stage) so failures are isolated and retryable, and so progress
can be reported stage-by-stage via Redis pub/sub -> WebSocket.
"""
from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "document_clean_reconstruct",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.pipeline_tasks",
        "app.workers.export_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=settings.celery_worker_concurrency,
    # Two queues, not one: `pipeline.process_document` and
    # `export.regenerate_document_export` have very different resource
    # profiles (the former is CPU/RAM-heavy and can run for hours; the
    # latter is a lightweight, usually-seconds-long DB+file read/write) and
    # are now consumed by separate worker services (see docker-compose.yml
    # `worker` vs `export-worker`) so a slow OCR job never blocks a
    # regenerate/export request behind it in the queue (a real regression
    # risk introduced by deliberately dropping CELERY_WORKER_CONCURRENCY to
    # 1 on the pipeline worker -- see that setting's comment in .env). A
    # worker service consuming only its own queue (`-Q pipeline` /
    # `-Q export`) still reads this same routing table, so this is the
    # only place the split needs to be declared.
    task_routes={
        "pipeline.process_document": {"queue": "pipeline"},
        "export.regenerate_document_export": {"queue": "export"},
    },
    task_default_queue="pipeline",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    # A single document job processes every page sequentially (render+OCR+
    # layout+table+reconstruct) inside one task — for a 400+ page scan that
    # can legitimately run for hours, so this must scale with document size,
    # not sit at a fixed 30/40-minute limit that would kill any large job
    # partway through regardless of how well it's otherwise performing.
    task_soft_time_limit=60 * 60 * 6,
    task_time_limit=60 * 60 * 7,
    broker_connection_retry_on_startup=True,
    # Recycle each prefork worker process after exactly one document job.
    # This was a real, confirmed production bug: a Celery prefork worker
    # process that had already run one process_document task (which forks
    # `app.workers.page_worker_pool` child processes via billiard) went on
    # to handle a *second* document in the same OS process, and that
    # second job's pool-worker fork hung indefinitely with zero pages ever
    # starting -- classic "fork() is unsafe after your process has used
    # threads" corruption. PaddleOCR/PaddlePaddle's native runtime spawns
    # its own background threads (visible as its "bvar is busy at
    # sampling" logging); once a process has touched that, forking a new
    # child from it again is not reliably safe. Restarting the worker
    # process between document jobs costs a few seconds of Python/Celery
    # startup -- negligible next to a multi-minute-to-hour OCR job -- and
    # completely eliminates this class of cross-job contamination.
    worker_max_tasks_per_child=1,
)
