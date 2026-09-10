"""
Processing-progress pub/sub (spec section 20): Celery workers publish stage/
percent updates to a per-document Redis channel; the FastAPI WebSocket
endpoint (`/ws/documents/{id}`) subscribes and relays them to connected
clients in real time. `GET /api/documents/{id}/progress` is the polling
fallback, reading the latest snapshot straight from the ProcessingJob row.
"""
from __future__ import annotations

import json

import redis as redis_sync
import redis.asyncio as redis_async

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.document import ProgressEvent

_CHANNEL_PREFIX = "doc_progress:"

logger = get_logger(__name__)


def channel_name(document_id: str) -> str:
    return f"{_CHANNEL_PREFIX}{document_id}"


def publish_progress_sync(event: ProgressEvent) -> None:
    """Called from Celery worker tasks (sync context).

    Best-effort only: `update_job_progress` (the sole caller) has already
    committed the authoritative ProcessingJob/Document row to Postgres
    before calling this -- that row is what `GET .../progress` and a
    reconnecting WebSocket client read, so a lost publish here costs a
    live client one push update, nothing more (they catch up on the next
    stage change, or via polling). Given that, a transient Redis hiccup
    (briefly flaky Docker-internal DNS under host load has been observed
    in production) must never be allowed to propagate: this call sits
    directly in the main per-page pipeline loop, so an unhandled
    exception here previously killed the *entire* multi-hour document
    job over what is, by design, a disposable notification.
    """
    settings = get_settings()
    try:
        client = redis_sync.Redis.from_url(settings.redis_url, socket_connect_timeout=5, socket_timeout=5)
        try:
            client.publish(channel_name(event.document_id), event.model_dump_json())
        finally:
            client.close()
    except redis_sync.exceptions.RedisError as exc:
        logger.warning(
            "progress_publish_failed",
            document_id=event.document_id,
            stage=event.stage,
            error=str(exc),
        )


async def subscribe(document_id: str):
    """Async generator of ProgressEvent for the WebSocket handler."""
    settings = get_settings()
    client = redis_async.Redis.from_url(settings.redis_url)
    pubsub = client.pubsub()
    await pubsub.subscribe(channel_name(document_id))
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            try:
                yield ProgressEvent(**json.loads(data))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    finally:
        await pubsub.unsubscribe(channel_name(document_id))
        await client.close()
