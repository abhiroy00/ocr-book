import io
import sys
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.base import Base  # noqa: E402
from app.models import *  # noqa: F401,F403,E402  register all models


@pytest.fixture()
def test_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def tmp_storage(tmp_path):
    from app.services.storage import LocalStorageBackend

    return LocalStorageBackend(str(tmp_path / "storage"))


@pytest.fixture()
def sample_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (200, 300), color="white").save(buf, format="PNG")
    return buf.getvalue()
