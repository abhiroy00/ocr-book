"""Replica-safe database migration entrypoint.

The `backend` service is scaled behind the nginx load balancer
(`BACKEND_REPLICAS`, see docker-compose.yml / docker-compose.prod.yml), so
every replica runs this on startup. Running `alembic upgrade head` from N
containers at once is not safe on its own: each replica reads the current
revision from `alembic_version`, all of them see the same "not yet at head",
and all of them race to apply the same migration ("relation already exists",
or worse, partially-applied schema). A Postgres session-level advisory lock
serializes the critical section across every replica and host:

  replica A acquires the lock -> applies migrations -> releases
  replica B blocks on the lock -> acquires -> alembic sees head already
             reached -> no-op

The lock is tied to the connection, so it is released automatically if a
replica dies mid-migration (no stuck lock to clear by hand). This keeps the
existing dev workflow working (`docker compose up` re-runs migrations on
every start, picking up newly added revisions) while making it safe under
horizontal scaling -- an alternative one-shot `migrate` service would only
run once and would silently skip new migrations on later `up`s.
"""
from __future__ import annotations

import sys
from pathlib import Path

# This file is invoked as `python scripts/migrate_with_lock.py`, which puts
# `scripts/` -- not the backend root -- on sys.path, so `import app` would
# fail ("No module named 'app'"). Put the backend root (the parent of this
# script's directory) on sys.path explicitly, the same directory alembic's
# own `prepend_sys_path = .` resolves to when run from the working dir.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from app.core.config import get_settings  # noqa: E402

# Arbitrary but fixed application-wide key for the migration lock. Must not
# collide with any other advisory lock in this application.
_MIGRATION_LOCK_KEY = 0x0C7B00  # 816896


def main() -> int:
    settings = get_settings()
    config = Config(str(_BACKEND_ROOT / "alembic.ini"))
    # alembic/env.py reads the URL from Settings itself, but set it here too
    # so `alembic.ini`'s empty `sqlalchemy.url` can never win if that changes.
    config.set_main_option("sqlalchemy.url", settings.database_url)

    engine = create_engine(settings.database_url, future=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_LOCK_KEY})
        try:
            command.upgrade(config, "head")
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _MIGRATION_LOCK_KEY})
    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
