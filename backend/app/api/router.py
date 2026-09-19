from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import accession, batches, documents, files, ws

api_router = APIRouter()
api_router.include_router(documents.router, prefix="/api")
api_router.include_router(batches.router, prefix="/api")
api_router.include_router(files.router, prefix="/api")
api_router.include_router(accession.router, prefix="/api")

ws_router = APIRouter()
ws_router.include_router(ws.router)
