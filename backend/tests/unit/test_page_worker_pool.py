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
