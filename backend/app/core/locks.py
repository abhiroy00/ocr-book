"""
Shared constants for the per-document Redis processing lock.

Split out from `app.workers.pipeline_tasks` (which owns acquire/renew/
release) so `app.services.document_service` can check whether a lock is
currently held without importing the worker module and creating a
circular import (pipeline_tasks already imports document_service).
"""
from __future__ import annotations

DOCUMENT_LOCK_PREFIX = "pipeline:lock:document:"

# Set by the `/cancel` API endpoint, checked cooperatively by the pipeline
# task between pages (see `pipeline_tasks._is_cancel_requested`). A plain
# Redis key rather than a DB column: cancellation is a transient signal
# the running task consumes and clears, not state anything else needs to
# query or migrate for.
DOCUMENT_CANCEL_PREFIX = "pipeline:cancel:document:"

# Lock VALUES are `"{node_id}|{random}"` when OCR_NODE_ID is set (else just
# `"{random}"`), so a scheduler running on one OCR host can count only the
# documents running on ITS OWN host against ITS OWN RAM/CPU limit -- see
# `batch_scheduler.live_pipeline_document_ids`. Without this, a second
# worker machine compared the GLOBAL running count against its local limit
# and never admitted any work while the first machine was busy.
NODE_SEPARATOR = "|"


def lock_value_for_node(node_id: str, token: str) -> str:
    return f"{node_id}{NODE_SEPARATOR}{token}" if node_id else token


def lock_value_belongs_to_node(value: str | bytes | None, node_id: str) -> bool:
    if not node_id:
        return True  # node scoping disabled: every lock counts (single-host behaviour)
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode()
    return value.startswith(node_id + NODE_SEPARATOR)
