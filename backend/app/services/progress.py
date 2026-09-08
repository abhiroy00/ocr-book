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
from app.schemas.document import ProgressEvent

_CHANNEL_PREFIX = "doc_progress:"


def channel_name(document_id: str) -> str:
    return f"{_CHANNEL_PREFIX}{document_id}"


def publish_progress_sync(event: ProgressEvent) -> None:
    """Called from Celery worker tasks (sync context)."""
    settings = get_settings()
    client = redis_sync.Redis.from_url(settings.redis_url)
    try:
        client.publish(channel_name(event.document_id), event.model_dump_json())
    finally:
        client.close()


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
