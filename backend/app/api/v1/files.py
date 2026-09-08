"""
Local-storage file serving for the frontend (before/after preview images,
etc.) when STORAGE_BACKEND=local. In production with S3, `StorageBackend.
url_for()` returns a presigned URL directly and this route isn't used.
"""
from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.api.deps import get_storage_backend
from app.services.storage import StorageBackend

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{path:path}")
def get_file(path: str, storage: StorageBackend = Depends(get_storage_backend)):
    if not storage.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    data = storage.read(path)
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return Response(content=data, media_type=media_type)
