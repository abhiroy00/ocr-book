"""WebSocket progress endpoint (spec section 20): `/ws/documents/{id}`."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.services.progress import subscribe

router = APIRouter(tags=["websocket"])
logger = get_logger(__name__)


@router.websocket("/ws/documents/{document_id}")
async def document_progress_ws(websocket: WebSocket, document_id: str) -> None:
    await websocket.accept()
    try:
        async for event in subscribe(document_id):
            await websocket.send_json(event.model_dump(mode="json"))
            if event.status.value in ("COMPLETED", "FAILED"):
                break
    except WebSocketDisconnect:
        logger.info("ws_client_disconnected", document_id=document_id)
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed
