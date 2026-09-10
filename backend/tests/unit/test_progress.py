from unittest.mock import MagicMock, patch

import redis as redis_sync

from app.schemas.document import ProgressEvent
from app.services.progress import publish_progress_sync


def _make_event() -> ProgressEvent:
    return ProgressEvent(
        document_id="11111111-1111-1111-1111-111111111111",
        job_id="22222222-2222-2222-2222-222222222222",
        stage="ocr",
        percent=50,
        page=3,
        message="Running OCR",
        status="OCR_PROCESSING",
    )


def test_publish_progress_sync_happy_path_publishes_once():
    fake_client = MagicMock()
    with patch("app.services.progress.redis_sync.Redis.from_url", return_value=fake_client):
        publish_progress_sync(_make_event())
    fake_client.publish.assert_called_once()
    fake_client.close.assert_called_once()


def test_publish_progress_sync_swallows_redis_connection_error():
    """Regression test for a real production incident: a transient Redis
    DNS blip during `client.publish()` previously propagated all the way
    out of the Celery task, killing an entire in-progress multi-hour
    document processing run over what is, by design, a best-effort
    real-time notification (the authoritative ProcessingJob/Document row
    is already committed to Postgres by the caller before this runs)."""
    fake_client = MagicMock()
    fake_client.publish.side_effect = redis_sync.exceptions.ConnectionError(
        "Error -3 connecting to redis:6379. Temporary failure in name resolution."
    )
    with patch("app.services.progress.redis_sync.Redis.from_url", return_value=fake_client):
        publish_progress_sync(_make_event())  # must not raise
    fake_client.close.assert_called_once()


def test_publish_progress_sync_swallows_redis_timeout_error():
    fake_client = MagicMock()
    fake_client.publish.side_effect = redis_sync.exceptions.TimeoutError("timed out")
    with patch("app.services.progress.redis_sync.Redis.from_url", return_value=fake_client):
        publish_progress_sync(_make_event())  # must not raise
