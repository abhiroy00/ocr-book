"""
Unit tests for the cooperative cancellation flag used by the `/cancel`
endpoint and checked between pages inside the pipeline task (see
`app.workers.pipeline_tasks._is_cancel_requested` /
`_clear_cancel_flag`). Mocks `redis.Redis.from_url` the same way the
existing `detect_and_fail_stale_job` tests do, rather than requiring a
live Redis -- these two functions are pure Redis GET/DELETE wrappers, so
there is nothing pipeline- or Celery-specific left to exercise here.
"""
from unittest.mock import MagicMock, patch

from app.core.locks import DOCUMENT_CANCEL_PREFIX
from app.workers.pipeline_tasks import _clear_cancel_flag, _is_cancel_requested


def test_is_cancel_requested_true_when_flag_set():
    fake_client = MagicMock()
    fake_client.exists.return_value = 1
    with patch("redis.Redis.from_url", return_value=fake_client):
        assert _is_cancel_requested("doc-1") is True
    fake_client.exists.assert_called_once_with(DOCUMENT_CANCEL_PREFIX + "doc-1")


def test_is_cancel_requested_false_when_flag_absent():
    fake_client = MagicMock()
    fake_client.exists.return_value = 0
    with patch("redis.Redis.from_url", return_value=fake_client):
        assert _is_cancel_requested("doc-1") is False


def test_is_cancel_requested_fails_safe_on_redis_error():
    """A Redis hiccup while checking must never be mistaken for a cancel
    request -- that would silently kill unrelated jobs."""
    fake_client = MagicMock()
    fake_client.exists.side_effect = ConnectionError("boom")
    with patch("redis.Redis.from_url", return_value=fake_client):
        assert _is_cancel_requested("doc-1") is False


def test_clear_cancel_flag_deletes_the_right_key():
    fake_client = MagicMock()
    with patch("redis.Redis.from_url", return_value=fake_client):
        _clear_cancel_flag("doc-1")
    fake_client.delete.assert_called_once_with(DOCUMENT_CANCEL_PREFIX + "doc-1")


def test_clear_cancel_flag_does_not_raise_on_redis_error():
    fake_client = MagicMock()
    fake_client.delete.side_effect = ConnectionError("boom")
    with patch("redis.Redis.from_url", return_value=fake_client):
        _clear_cancel_flag("doc-1")  # must not raise
