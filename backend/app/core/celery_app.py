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
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=settings.celery_worker_concurrency,
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
)
