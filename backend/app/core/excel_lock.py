"""
File-based lock protecting the persisted Master Accession Register
workbook (`app.services.accession_register_service`) from concurrent
writes when multiple Celery workers process documents in parallel on the
same EC2 host.

`app.core.locks` already exists but only holds Redis-lock *constants* for
the per-document pipeline lock (`app.workers.pipeline_tasks`) -- there is
nothing there to reuse for a filesystem workbook, and coupling this to
Redis would be the wrong tool anyway: the thing being protected is a
single on-disk file, so an OS-level file lock is the direct fit.

Backed by `filelock.FileLock`, which uses the OS's own advisory locking
(`fcntl.flock` on Linux/EC2, `msvcrt` on Windows) rather than a hand-rolled
PID file. That is what actually gives "no orphan lock recovery needed":
the OS releases the underlying file descriptor the instant the holding
process dies (crash, SIGKILL, OOM-kill), so the very next `acquire()` from
another worker succeeds immediately -- there is no stale-PID file to
detect or clean up.
"""
from __future__ import annotations

from filelock import FileLock, Timeout

__all__ = ["FileLock", "Timeout", "lock_path_for", "new_file_lock"]


def lock_path_for(workbook_path: str) -> str:
    """`<workbook_path>.lock`, e.g.
    `storage/master_register/master_accession_register.xlsx.lock`."""
    return f"{workbook_path}.lock"


def new_file_lock(workbook_path: str, timeout: float) -> FileLock:
    """One lock instance per acquire/release cycle (filelock's own
    recommended usage -- a `FileLock` object tracks its own acquire depth,
    so reusing one instance across unrelated critical sections would let a
    later section silently ride on an earlier, already-released lock's
    reentrancy counter)."""
    return FileLock(lock_path_for(workbook_path), timeout=timeout)
