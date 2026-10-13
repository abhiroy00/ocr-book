"""
Tests for the controlled parallel page-processing pool
(`app.workers.page_worker_pool`). Uses the real `tesseract` provider (no
network, no large model download) rather than paddleocr, to keep these
tests fast and CI-friendly; a separate manual/benchmark run against
paddleocr on the real document is what the perf report is based on.
"""
import fitz

from app.vision.preprocessor import PreprocessProfile
from app.workers.page_worker_pool import PageWorkerPool


def _make_pdf_bytes(n_pages: int) -> bytes:
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1} sample text {1000 + i}", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def test_pool_processes_all_pages_with_correct_content_and_no_cross_contamination():
    """Regression guard for the core correctness requirement: page N's
    result must contain page N's own content, never another worker's --
    each worker renders/OCRs its own page independently."""
    pdf_bytes = _make_pdf_bytes(4)
    pool = PageWorkerPool("tesseract", "test-doc-a", pdf_bytes, True, 150, PreprocessProfile.FAST, num_workers=2)
    try:
        pool.submit([1, 2, 3, 4])
        results = {r.page_number: r for r in pool.results(4)}
    finally:
        pool.shutdown()

    assert set(results.keys()) == {1, 2, 3, 4}
    for n, r in results.items():
        assert r.error is None
        text = " ".join(w.text for w in r.words)
        assert str(n) in text or str(999 + n) in text  # "Page N" / "100{n-1}" both landed on the right page


def test_pool_never_spawns_more_workers_than_requested():
    pdf_bytes = _make_pdf_bytes(3)
    pool = PageWorkerPool("tesseract", "test-doc-b", pdf_bytes, True, 150, PreprocessProfile.FAST, num_workers=2)
    try:
        assert len(pool._procs) == 2  # exactly the requested count -- never one-per-page
        pool.submit([1, 2, 3])
        list(pool.results(3))
    finally:
        pool.shutdown()


def test_pool_reuses_the_same_worker_across_multiple_pages():
    """The whole point of a persistent pool (spec section 6): a worker
    that already loaded its OCR engine must go on to process more than
    one page, not be torn down after its first."""
    pdf_bytes = _make_pdf_bytes(6)
    pool = PageWorkerPool("tesseract", "test-doc-c", pdf_bytes, True, 150, PreprocessProfile.FAST, num_workers=2)
    try:
        pool.submit([1, 2, 3, 4, 5, 6])
        results = list(pool.results(6))
    finally:
        pool.shutdown()

    worker_names = {r.worker_name for r in results}
    assert len(worker_names) <= 2  # only the 2 pool workers ever did this work
    # with 6 pages and 2 workers, at least one worker must have handled >1 page
    from collections import Counter

    counts = Counter(r.worker_name for r in results)
    assert max(counts.values()) >= 2


def test_pool_shutdown_terminates_all_worker_processes():
    pdf_bytes = _make_pdf_bytes(2)
    pool = PageWorkerPool("tesseract", "test-doc-d", pdf_bytes, True, 150, PreprocessProfile.FAST, num_workers=2)
    pool.submit([1, 2])
    list(pool.results(2))
    pool.shutdown()
    assert pool.alive_count() == 0


def test_pool_recycles_a_worker_after_its_page_quota_and_keeps_processing():
    """Regression test for a real incident (2026-09-14): a single
    long-lived worker's memory grew unbounded over a large document (8GB
    after 68 pages vs. ~4.5GB after 10 in a short benchmark), degrading
    per-page time roughly 10x alongside it. A worker recycled after
    `max_pages_per_worker` pages must be transparently replaced -- every
    submitted page still gets a correct result, with no duplicates and no
    pages lost across the respawn."""
    pdf_bytes = _make_pdf_bytes(5)
    pool = PageWorkerPool(
        "tesseract", "test-doc-e", pdf_bytes, True, 150, PreprocessProfile.FAST, num_workers=1, max_pages_per_worker=2,
    )
    original_pid = pool._procs[0].pid
    try:
        pool.submit([1, 2, 3, 4, 5])
        results = list(pool.results(5))
    finally:
        pool.shutdown()

    assert sorted(r.page_number for r in results) == [1, 2, 3, 4, 5]
    assert all(r.error is None for r in results)
    # 1 worker, quota of 2 pages, 5 pages submitted -> this can only have
    # succeeded via at least one recycle (a real, different OS process at
    # the same pw-0 slot); the worker_name itself is stable across a
    # recycle (see `_spawn_worker`), so the PID is the signal that
    # actually distinguishes "the same process kept running" from
    # "an old process exited and a new one took its place".
    assert pool._procs[0].pid != original_pid


# ---------------------------------------------------------------------------
# Self-healing (2026-10-08: a 402-page book stalled after an OOM killed its
# page workers -- the pool neither replaced them nor re-queued their pages,
# and a hung worker blocked another document for 40 minutes).
# ---------------------------------------------------------------------------

import os
import time

import pytest

import app.services.pdf_ingest as pdf_ingest


def _sabotage_page(monkeypatch, tmp_path, page: int, action: str, times: int = 1):
    """Make `render_single_page` misbehave for `page` the first `times`
    times it is attempted (counted across processes via a marker file --
    pool workers are forked, so the patch reaches them)."""
    real = pdf_ingest.render_single_page
    counter = tmp_path / f"attempts_{page}"

    def patched(file_bytes, is_pdf, page_number, dpi):
        if page_number == page:
            n = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(n + 1))
            if n < times:
                if action == "die":
                    os._exit(137)  # like an OOM kill: no cleanup, no result
                if action == "hang":
                    time.sleep(600)
        return real(file_bytes, is_pdf, page_number, dpi)

    monkeypatch.setattr(pdf_ingest, "render_single_page", patched)
    return counter


def test_pool_replaces_a_worker_killed_mid_page_and_retries_that_page(monkeypatch, tmp_path):
    counter = _sabotage_page(monkeypatch, tmp_path, page=2, action="die")
    pool = PageWorkerPool("tesseract", "test-doc-f", _make_pdf_bytes(4), True, 150, PreprocessProfile.FAST, num_workers=2)
    try:
        pool.submit([1, 2, 3, 4])
        results = {r.page_number: r for r in pool.results(4)}
    finally:
        pool.shutdown()

    assert set(results) == {1, 2, 3, 4}
    assert all(r.error is None for r in results.values())  # page 2 succeeded on its retry
    assert counter.read_text() == "2"


def test_pool_kills_and_replaces_a_hung_worker(monkeypatch, tmp_path):
    _sabotage_page(monkeypatch, tmp_path, page=1, action="hang")
    pool = PageWorkerPool(
        "tesseract", "test-doc-g", _make_pdf_bytes(2), True, 150, PreprocessProfile.FAST, num_workers=1, page_timeout_seconds=3,
    )
    started = time.monotonic()
    try:
        pool.submit([1, 2])
        results = {r.page_number: r for r in pool.results(2)}
    finally:
        pool.shutdown()

    assert set(results) == {1, 2} and all(r.error is None for r in results.values())
    assert time.monotonic() - started < 120  # not the 600s the hang would have taken


def test_pool_gives_up_on_a_page_that_keeps_killing_workers(monkeypatch, tmp_path):
    _sabotage_page(monkeypatch, tmp_path, page=2, action="die", times=99)
    pool = PageWorkerPool("tesseract", "test-doc-h", _make_pdf_bytes(3), True, 150, PreprocessProfile.FAST, num_workers=1)
    try:
        pool.submit([1, 2, 3])
        results = {r.page_number: r for r in pool.results(3)}
    finally:
        pool.shutdown()

    assert set(results) == {1, 2, 3}
    assert results[2].error and not results[2].processed_path  # handed to the pipeline's sequential sweep
    assert results[1].error is None and results[3].error is None  # the other pages were not lost


def test_pool_recycles_a_worker_that_crosses_the_memory_ceiling():
    pool = PageWorkerPool(
        "tesseract", "test-doc-i", _make_pdf_bytes(3), True, 150, PreprocessProfile.FAST, num_workers=1, max_rss_mb=1,
    )
    first_pid = pool._procs[0].pid
    try:
        pool.submit([1, 2, 3])
        results = list(pool.results(3))
    finally:
        pool.shutdown()

    assert sorted(r.page_number for r in results) == [1, 2, 3] and all(r.error is None for r in results)
    assert pool._procs[0].pid != first_pid
    assert sum(r.worker_exiting for r in results) >= 2  # every page crossed a 1MB ceiling


def test_pool_calls_heartbeat_while_waiting(monkeypatch, tmp_path):
    _sabotage_page(monkeypatch, tmp_path, page=1, action="hang")
    beats = []
    pool = PageWorkerPool(
        "tesseract", "test-doc-j", _make_pdf_bytes(1), True, 150, PreprocessProfile.FAST, num_workers=1, page_timeout_seconds=12,
    )
    try:
        pool.submit([1])
        list(pool.results(1, heartbeat=lambda: beats.append(time.monotonic())))
    finally:
        pool.shutdown()

    assert len(beats) >= 2  # kept the document lock alive during the slow page
