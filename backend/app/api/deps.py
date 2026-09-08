"""
Shared FastAPI dependencies. `get_current_user` is an auth-ready stub
(spec section 32): the API is structured so real authentication can be
dropped in here without touching route handlers.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Header

from app.db.session import get_db  # noqa: F401  (re-exported for router imports)
from app.services.storage import StorageBackend, get_storage


def get_storage_backend() -> StorageBackend:
    return get_storage()


def get_current_user(x_user_email: Optional[str] = Header(default=None)) -> Optional[str]:
    """Placeholder identity extraction. Wire real auth (JWT/session) here;
    routes already accept this dependency so no call sites change later."""
    return x_user_email
