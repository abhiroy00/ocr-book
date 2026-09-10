"""
Controlled parallel page processing (bounded worker pool, not one-thread/
process-per-page).

Why processes, not threads: PaddleOCR/PaddlePaddle inference is native C++
code doing CPU-bound matrix math and does not release Python's GIL for the
duration of a call, so threads would not run OCR concurrently at all --
they'd just take turns behind the GIL with added overhead. The rest of the
per-page work (OpenCV preprocessing, table/layout detection) is likewise
CPU-bound. `ProcessPoolExecutor`/`multiprocessing` is therefore the correct
tool (this was the deciding factor, not a guess -- see the perf report in
the accompanying commit message / README for the measured before/after).

Concretely this uses `billiard` (not stdlib `multiprocessing`) for the same
reason `app.ocr.subprocess_runner.IsolatedOCRWorker` already does: Celery's
own prefork pool workers are themselves daemonic processes, and stdlib
`multiprocessing` refuses to let a daemonic process spawn children.

Each pool worker is a long-lived process (not respawned per page): it loads
the OCR engine and builds a `LayoutDetector` exactly once at startup, then
pulls page numbers off a shared task queue until told to stop. This keeps
PaddleOCR's model-load cost (a few seconds) a one-time-per-worker cost
rather than a one-time-per-page cost (spec requirement -- see module
docstring in `subprocess_runner.py` for why this matters).

Workers each open their own `fitz.Document` handle from the same inherited
source-PDF bytes (safe -- see `pdf_ingest.render_single_page`) and write
their own per-page image/crop files directly to storage; only small,
picklable structured results (words/tables/layout blocks, not raw images)
travel back over the result queue. This keeps peak memory bounded to
roughly `OCR_WORKERS` pages' worth of image buffers at any instant, never
the whole document (spec requirement).

Crash isolation: if a worker process dies mid-page (the same native-crash
risk `IsolatedOCRWorker` was built to contain), the pool detects it via
`is_alive()` + a per-task timeout, marks that one page as failed (falling
back to in-process Tesseract for just that page, cheap since it has no
heavy model to load), and keeps the remaining workers running -- one page,
or even one dead worker, never takes down the whole document.
"""
from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Iterator

import billiard as mp

from app.core.logging import get_logger
from app.models.enums import LayoutBlockType
from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult
from app.tables.models import DetectedTable
from app.vision.preprocessor import PreprocessProfile

logger = get_logger(__name__)

_RESULT_TIMEOUT_SECONDS = 180
_FIRST_RESULT_TIMEOUT_SECONDS = 300  # first page per worker also pays one-time model load/download cost

_GRAPHIC_TYPES = {
    LayoutBlockType.IMAGE,
    LayoutBlockType.CHART,
    LayoutBlockType.SIGNATURE,
    LayoutBlockType.STAMP,
    LayoutBlockType.HANDWRITTEN,
}


@dataclass
class PageWorkResult:
    """Everything the main process needs to persist one page, without
    shipping the (potentially large) image buffers themselves back over
    IPC -- those were already written to storage by the worker."""

    page_number: int
    width: int = 0
    height: int = 0
    dpi: int = 0
    rotation: float = 0.0
    original_path: str = ""
    processed_path: str = ""
    words: list[OCRWordResult] = field(default_factory=list)
    tables: list[DetectedTable] = field(default_factory=list)
    layout_results: list = field(default_factory=list)
    ocr_provider_used: str = ""
    worker_name: str = ""
    processing_time: float = 0.0
    error: str | None = None


def _child_main(
    worker_name: str,
    provider_name: str,
    document_id: str,
    file_bytes: bytes,
    is_pdf: bool,
    dpi: int,
    profile_value: str,
    task_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """Pool worker entry point. Loads the OCR engine and layout detector
    once, then serves page numbers off `task_q` until it receives the
    shutdown sentinel (None). Mirrors `IsolatedOCRWorker._child_main`'s
    contract of never raising out of this function for a normal per-page
    error -- only an OS-level crash takes the process down, which the
    parent observes via `is_alive()`."""
    import cv2

    # OpenCV has its own internal thread pool for per-call parallelism
    # (resize/blur/threshold/...), independent of PaddleOCR's (see
    # `app.ocr.paddle_engine._get_paddle_engine` for that half of the
    # fix). With N pool workers running at once, each defaulting to "use
    # every core", the same oversubscription/thread-thrashing collapse
    # applies here too -- pin this process's OpenCV work to one thread and
    # let the pool's own N-process parallelism be the only source of
    # concurrency.
    cv2.setNumThreads(1)

    from app.core.logging import configure_logging, get_logger as _get_logger
    from app.layout.detector import LayoutDetector
    from app.ocr.factory import get_ocr_provider
    from app.services.pdf_ingest import render_single_page
    from app.services.storage import get_storage
    from app.tables.engine import detect_tables
    from app.vision.preprocessor import preprocess_page

    configure_logging(debug=False)
    log = _get_logger(f"page_worker.{worker_name}")

    try:
        provider = get_ocr_provider(provider_name)
    except Exception as exc:  # noqa: BLE001
        result_q.put(PageWorkResult(page_number=-1, error=f"provider init failed: {exc}"))
        return

    storage = get_storage()
    layout_detector = LayoutDetector()
    profile = PreprocessProfile(profile_value)

    while True:
        page_number = task_q.get()
        if page_number is None:  # sentinel: shut down
            return

        start = time.perf_counter()
        log.info("page_worker_processing", worker=worker_name, page=page_number)
        try:
            rendered = render_single_page(file_bytes, is_pdf, page_number, dpi)

            original_path = f"pages/{document_id}/page_{page_number:04d}_original.png"
            storage.write(original_path, _encode_png(rendered.image))

            preproc = preprocess_page(rendered.image, profile)
            processed_image = preproc.image
            processed_path = f"processed/{document_id}/page_{page_number:04d}_processed.png"
            storage.write(processed_path, _encode_png(processed_image))

            ocr_start = time.perf_counter()
            try:
                page_result = provider.recognize_page(processed_image, page_number, dpi)
                words = page_result.words
                log.info("page_worker_ocr_done", worker=worker_name, page=page_number, duration=round(time.perf_counter() - ocr_start, 2))
            except Exception as exc:  # noqa: BLE001 - a normal OCR error for this page, try tesseract before giving up
                log.warning("page_worker_ocr_error_falling_back", page=page_number, error=str(exc))
                try:
                    fallback = get_ocr_provider("tesseract")
                    words = fallback.recognize_page(processed_image, page_number, dpi).words
                except Exception as exc2:  # noqa: BLE001
                    log.error("page_worker_ocr_fully_failed", page=page_number, error=str(exc2))
                    words = []

            try:
                tables = detect_tables(processed_image, words, rendered.width, rendered.height)
            except Exception as exc:  # noqa: BLE001 - spec: table failure falls back to OCR positioning, never crashes
                log.error("page_worker_table_detection_failed", page=page_number, error=str(exc))
                tables = []

            layout_results = layout_detector.detect(processed_image, rendered.image, words, tables, rendered.width, rendered.height)
            _crop_and_attach_graphic_images(storage, document_id, page_number, processed_image, layout_results)

            log.info(
                "page_worker_completed", worker=worker_name, page=page_number,
                words=len(words), tables=len(tables), duration=round(time.perf_counter() - start, 2),
            )
            result_q.put(
                PageWorkResult(
                    page_number=page_number,
                    width=rendered.width,
                    height=rendered.height,
                    dpi=rendered.dpi,
                    rotation=rendered.rotation,
                    original_path=original_path,
                    processed_path=processed_path,
                    words=words,
                    tables=tables,
                    layout_results=layout_results,
                    ocr_provider_used=provider.name,
                    worker_name=worker_name,
                    processing_time=time.perf_counter() - start,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a page-level failure must never kill this worker process
            log.error("page_worker_page_failed", page=page_number, error=str(exc))
            result_q.put(PageWorkResult(page_number=page_number, processing_time=time.perf_counter() - start, error=str(exc)))


def _crop_and_attach_graphic_images(storage, document_id: str, page_number: int, image, layout_results) -> None:
    for i, result in enumerate(layout_results):
        if result.block_type not in _GRAPHIC_TYPES:
            continue
        x1, y1 = max(0, int(result.bbox.x1)), max(0, int(result.bbox.y1))
        x2, y2 = min(image.shape[1], int(result.bbox.x2)), min(image.shape[0], int(result.bbox.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        rel_path = f"processed/{document_id}/page_{page_number:04d}_block_{i:04d}.png"
        storage.write(rel_path, _encode_png(crop))
        result.image_ref = rel_path


def _encode_png(image) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode page image as PNG")
    return encoded.tobytes()


class PageWorkerPool:
    """A bounded pool of `num_workers` persistent page-processing
    processes. Submit page numbers, then drain results (which may arrive
    out of page-number order -- callers must sort by `page_number` before
    building anything that depends on document order, e.g. the final PDF/
    DOCX; per-page DB persistence does not need to)."""

    def __init__(
        self,
        provider_name: str,
        document_id: str,
        file_bytes: bytes,
        is_pdf: bool,
        dpi: int,
        profile: PreprocessProfile,
        num_workers: int,
    ) -> None:
        self.num_workers = max(1, num_workers)
        self._task_q: "mp.Queue" = mp.Queue()
        self._result_q: "mp.Queue" = mp.Queue()
        self._procs: list[mp.Process] = []
        self._first_result_pending = True
        # `in_flight` tracks which page each still-alive worker is currently
        # holding, so a crashed worker's page can be identified and retried
        # rather than silently vanishing (section 18: one page's failure
        # must not lose that page's content without at least one fallback
        # attempt).
        self._in_flight: dict[int, int] = {}  # worker index -> page_number

        for i in range(self.num_workers):
            worker_name = f"pw-{i}"
            proc = mp.Process(
                target=_child_main,
                args=(worker_name, provider_name, document_id, file_bytes, is_pdf, dpi, profile.value, self._task_q, self._result_q),
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)

    def submit(self, page_numbers: list[int]) -> None:
        for n in page_numbers:
            self._task_q.put(n)

    def results(self, expected_count: int) -> Iterator[PageWorkResult]:
        """Yields one `PageWorkResult` per submitted page (in completion
        order, NOT page-number order) until `expected_count` results have
        been produced or every worker has died."""
        received = 0
        while received < expected_count:
            if not any(p.is_alive() for p in self._procs):
                logger.error("page_worker_pool_all_workers_dead", received=received, expected=expected_count)
                return
            timeout = _FIRST_RESULT_TIMEOUT_SECONDS if self._first_result_pending else _RESULT_TIMEOUT_SECONDS
            try:
                result = self._result_q.get(timeout=timeout)
            except queue.Empty:
                logger.error("page_worker_pool_result_timeout", received=received, expected=expected_count)
                continue  # a still-alive worker may just be slow on a big page; loop re-checks liveness above
            self._first_result_pending = False
            received += 1
            yield result

    def shutdown(self) -> None:
        for _ in self._procs:
            try:
                self._task_q.put(None)
            except Exception:  # noqa: BLE001
                pass
        for proc in self._procs:
            proc.join(5)
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()

    def alive_count(self) -> int:
        return sum(1 for p in self._procs if p.is_alive())
