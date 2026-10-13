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
travel back over each worker's pipe. This keeps peak memory bounded to
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

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator

import billiard as mp
import billiard.connection as mp_connection

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import LayoutBlockType
from app.schemas.geometry import BBox
from app.schemas.ocr import OCRWordResult
from app.tables.models import DetectedTable
from app.vision.preprocessor import PreprocessProfile

logger = get_logger(__name__)


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
    # Set on the LAST result a worker sends before it exits on purpose
    # (page quota or memory ceiling reached) -- the parent replaces it.
    worker_exiting: bool = False


def _make_oom_preferred_victim() -> None:
    """Ask the Linux OOM killer to pick THIS page worker first when memory
    runs out. Without it the kernel picked whichever process was biggest
    or unluckiest -- including the Celery process that owns the whole
    document job -- and the document stalled with no way to recover
    (2026-10-08: a 402-page book ended up "Processing stalled" after an
    OOM). A killed page worker is recoverable: the pool re-queues its page
    and starts a fresh worker. Raising one's own score needs no privilege;
    silently skipped where /proc is unavailable (Windows/macOS dev)."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - telemetry only; never fail a page over it
        return 0.0


def _child_main(
    worker_name: str,
    provider_name: str,
    document_id: str,
    file_bytes: bytes,
    is_pdf: bool,
    dpi: int,
    profile_value: str,
    conn,
    max_pages: int,
    max_rss_mb: int = 0,
) -> None:
    """Pool worker entry point. Loads the OCR engine and layout detector
    once, then serves page numbers the parent sends over `conn` until it receives the
    shutdown sentinel (None) OR has processed `max_pages` pages, whichever
    comes first -- at which point it exits cleanly and `PageWorkerPool`
    replaces it with a fresh process at the same slot (see `results()`).

    This recycling is not optional cleanup: confirmed by direct
    measurement on a real 446-page document (2026-09-14) that a single
    long-lived worker's memory climbed to 8GB after only 68 pages (vs.
    ~4.5GB after 10 in a short benchmark) -- unbounded growth over a long
    run, not a one-time model-load cost -- and per-page time degraded
    roughly 10x alongside it (171s/page vs. a clean ~15-17s/page
    baseline), consistent with memory-pressure thrashing. Recycling
    periodically resets that growth back to baseline instead of letting it
    compound for the rest of a large document.

    Mirrors `IsolatedOCRWorker._child_main`'s contract of never raising
    out of this function for a normal per-page error -- only an OS-level
    crash takes the process down, which the parent observes via
    `is_alive()`."""
    import os

    _make_oom_preferred_victim()

    # Must happen before numpy/cv2/paddle are imported -- OpenMP/MKL/
    # OpenBLAS read these once at library-load time, not per-call, so
    # setting them any later has no effect. Defense-in-depth alongside
    # `cpu_threads=1` (paddle_engine.py) and `cv2.setNumThreads(1)` below:
    # those cover the two libraries this codebase calls directly, but
    # numpy/scipy/scikit-image underneath can link against the same
    # OpenMP/MKL/OpenBLAS runtimes and default to "use every core" on
    # their own. With N pool workers each spawning their own full-core
    # thread pool, the result is the exact oversubscription collapse
    # already confirmed once in this codebase (2 workers x 10 threads ->
    # per-page time 4s -> 1150s, see paddle_engine.py) -- pinning every
    # thread-pool env var to 1 here makes this process-level pool
    # (N == OCR_WORKERS) the sole source of concurrency, regardless of
    # what any dependency's own default would otherwise pick.
    for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_env_var, "1")
    if provider_name == "paddleocr":
        # PaddleOCR's own MKLDNN/MKL thread pool is the one place this worker
        # deliberately uses several threads (OCR_PADDLE_THREADS): with only
        # one worker fitting in RAM on this host, a 1-thread cap left 13 of
        # 14 cores idle. Everything else above stays at 1.
        _paddle_threads = str(max(1, get_settings().ocr_paddle_threads))
        os.environ["OMP_NUM_THREADS"] = _paddle_threads
        os.environ["MKL_NUM_THREADS"] = _paddle_threads

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
        conn.send(PageWorkResult(page_number=-1, worker_name=worker_name, error=f"provider init failed: {exc}"))
        return

    storage = get_storage()
    layout_detector = LayoutDetector()
    profile = PreprocessProfile(profile_value)

    pages_done = 0
    while True:
        try:
            page_number = conn.recv()
        except EOFError:
            return  # parent went away
        if page_number is None:  # sentinel: shut down
            return

        start = time.perf_counter()
        log.info("page_worker_processing", worker=worker_name, page=page_number)
        result: PageWorkResult
        try:
            rendered = render_single_page(file_bytes, is_pdf, page_number, dpi)

            original_path = f"pages/{document_id}/page_{page_number:04d}_original.png"
            storage.write(original_path, _encode_png(rendered.image))

            preproc = preprocess_page(rendered.image, profile, dpi=rendered.dpi)
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

            # Both args must be `processed_image`, not `rendered.image` (the
            # pre-deskew/pre-crop raw scan): OCR ran on `processed_image`,
            # so `words` bboxes are already in its coordinate frame.
            # Passing the raw scan here put graphic-region bboxes in a
            # DIFFERENT (un-rotated) frame, so cropping them out of
            # `processed_image` afterward grabbed a skewed/misaligned
            # rectangle whenever a page actually needed deskewing --
            # confirmed against real output: a map rendered rotated within
            # its own crop, with black wedge corners where the rotation
            # didn't fill the rectangle.
            layout_results = layout_detector.detect(processed_image, processed_image, words, tables, rendered.width, rendered.height)
            # Graphics are cropped from the raw full-colour scan only when
            # preprocessing left its geometry untouched (no rotate/deskew/
            # border crop), so the processed-frame bboxes still line up.
            _crop_and_attach_graphic_images(
                storage, document_id, page_number, rendered.image, processed_image, layout_results,
                geometry_unchanged=_geometry_unchanged(preproc, rendered.image, processed_image),
            )

            log.info(
                "page_worker_completed", worker=worker_name, page=page_number,
                words=len(words), tables=len(tables), duration=round(time.perf_counter() - start, 2),
                rss_mb=round(_rss_mb()),
            )
            result = PageWorkResult(
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
        except Exception as exc:  # noqa: BLE001 - a page-level failure must never kill this worker process
            log.error("page_worker_page_failed", page=page_number, error=str(exc))
            result = PageWorkResult(
                page_number=page_number, worker_name=worker_name, processing_time=time.perf_counter() - start, error=str(exc)
            )

        pages_done += 1
        # Recycle on page quota OR memory ceiling, whichever comes first.
        # The quota alone was not enough: measured 2026-10-08, one PaddleOCR
        # worker reached 8.3GB after only 14 pages of a real scanned book --
        # far below the 25-page quota -- and with two documents running the
        # host OOMed. The RSS ceiling bounds each worker's memory directly,
        # however fast a particular book makes it grow.
        rss = _rss_mb()
        reason = None
        if pages_done >= max_pages:
            reason = "page_quota"
        elif max_rss_mb and rss >= max_rss_mb:
            reason = "memory_ceiling"
        result.worker_exiting = reason is not None
        conn.send(result)
        if reason:
            log.info("page_worker_recycling", worker=worker_name, pages_done=pages_done, reason=reason, rss_mb=round(rss))
            return  # clean exit -- PageWorkerPool.results() spawns a replacement at this slot


def _geometry_unchanged(preproc, original_image, processed_image) -> bool:
    return (
        preproc.rotation_applied_deg == 0.0
        and preproc.skew_angle_deg == 0.0
        and original_image.shape[:2] == processed_image.shape[:2]
    )


def _crop_and_attach_graphic_images(
    storage, document_id: str, page_number: int, original_image, processed_image, layout_results, geometry_unchanged: bool = False
) -> None:
    # `layout_results` bboxes are in `processed_image`'s frame; the raw scan
    # is only a valid crop source when that frame matches it exactly.
    source_image = original_image if geometry_unchanged else processed_image
    for i, result in enumerate(layout_results):
        if result.block_type not in _GRAPHIC_TYPES:
            continue
        x1, y1 = max(0, int(result.bbox.x1)), max(0, int(result.bbox.y1))
        x2, y2 = min(source_image.shape[1], int(result.bbox.x2)), min(source_image.shape[0], int(result.bbox.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = source_image[y1:y2, x1:x2]
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
    DOCX; per-page DB persistence does not need to).

    Each worker has its OWN duplex pipe, and the parent hands out one page
    at a time to an idle worker. This replaced a shared task queue + shared
    result queue after a real 402-page book stalled (2026-10-08): workers
    killed by the OOM killer took their page with them (nothing knew which
    page they held), and a process killed while holding a shared queue's
    internal lock can block every other worker forever. With per-worker
    pipes the parent always knows exactly which page each worker holds, and
    no lock is shared between processes.

    Self-healing:
      * a worker that DIES mid-page (OOM kill, native crash) is replaced and
        the page it held is retried -- up to `_MAX_PAGE_ATTEMPTS` per page;
      * a worker that HANGS on one page past `page_timeout_seconds` is
        killed and handled the same way (one previously stalled a document
        for 40 minutes while still counting as "alive");
      * a worker that exits on purpose (page quota / memory ceiling) is
        replaced without losing anything.
    Only a page that keeps killing workers comes back as an error result,
    which the pipeline's sequential sweep then retries one last time."""

    _MAX_PAGE_ATTEMPTS = 2
    _POLL_SECONDS = 5

    def __init__(
        self,
        provider_name: str,
        document_id: str,
        file_bytes: bytes,
        is_pdf: bool,
        dpi: int,
        profile: PreprocessProfile,
        num_workers: int,
        max_pages_per_worker: int | None = None,
        max_rss_mb: int | None = None,
        page_timeout_seconds: int | None = None,
    ) -> None:
        self.num_workers = max(1, num_workers)
        # Confirmed by direct measurement (2026-09-14) that a single
        # worker's memory grows over a long document; recycling each worker
        # after this many pages (or once it crosses the memory ceiling)
        # resets that growth periodically instead of letting a large
        # document compound it for hours.
        settings = get_settings()
        self.max_pages_per_worker = max_pages_per_worker or settings.ocr_worker_max_pages
        self.max_rss_mb = settings.ocr_worker_max_rss_mb if max_rss_mb is None else max_rss_mb
        self.page_timeout_seconds = page_timeout_seconds or settings.ocr_page_timeout_seconds
        self._provider_name = provider_name
        self._document_id = document_id
        self._file_bytes = file_bytes
        self._is_pdf = is_pdf
        self._dpi = dpi
        self._profile = profile
        self._procs: list = [None] * self.num_workers
        self._conns: list = [None] * self.num_workers
        self._names: list[str] = [""] * self.num_workers  # "" = slot abandoned
        self._generation = 0
        # Unplanned replacements (crash/hang) allowed for the whole pool, so
        # a page that reliably kills every worker cannot respawn forever.
        self._crash_respawns_left = max(3, 2 * self.num_workers)
        self._pending: deque[int] = deque()
        self._in_flight: dict[int, tuple[int, float]] = {}  # slot -> (page_number, dispatched monotonic)
        self._attempts: dict[int, int] = {}

        for i in range(self.num_workers):
            self._spawn_worker(i)

    def _spawn_worker(self, index: int) -> "mp.Process":
        self._generation += 1
        worker_name = f"pw-{index}.{self._generation}"
        parent_conn, child_conn = mp.Pipe(duplex=True)
        proc = mp.Process(
            target=_child_main,
            args=(
                worker_name, self._provider_name, self._document_id, self._file_bytes, self._is_pdf,
                self._dpi, self._profile.value, child_conn, self.max_pages_per_worker, self.max_rss_mb,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # the parent keeps only its own end, so EOF is seen when the child dies
        self._procs[index] = proc
        self._conns[index] = parent_conn
        self._names[index] = worker_name
        return proc

    def submit(self, page_numbers: list[int]) -> None:
        for n in page_numbers:
            self._attempts.setdefault(n, 0)
            self._pending.append(n)
        self._dispatch()

    def _dispatch(self) -> None:
        for slot in range(self.num_workers):
            if not self._pending:
                return
            if not self._names[slot] or slot in self._in_flight or not self._procs[slot].is_alive():
                continue
            page_number = self._pending.popleft()
            try:
                self._conns[slot].send(page_number)
                self._in_flight[slot] = (page_number, time.monotonic())
            except (OSError, EOFError, BrokenPipeError):
                self._pending.appendleft(page_number)  # worker just died; the health check handles it

    def _retire(self, index: int, kill: bool) -> None:
        proc = self._procs[index]
        if kill and proc.is_alive():
            proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        try:
            self._conns[index].close()
        except OSError:
            pass

    def _handle_lost_worker(self, index: int, why: str) -> PageWorkResult | None:
        """A worker died or hung. Retry (or give up on) the page it held,
        and replace the worker while the crash budget lasts. Returns an
        error result when that page has now used up its attempts."""
        self._retire(index, kill=True)
        lost = self._in_flight.pop(index, None)
        given_up: PageWorkResult | None = None
        if lost is not None:
            page_number = lost[0]
            self._attempts[page_number] = self._attempts.get(page_number, 0) + 1
            if self._attempts[page_number] < self._MAX_PAGE_ATTEMPTS:
                logger.warning("page_worker_lost_page_requeued", worker=self._names[index], page=page_number, reason=why)
                self._pending.appendleft(page_number)
            else:
                logger.error("page_worker_page_gave_up", worker=self._names[index], page=page_number, reason=why)
                given_up = PageWorkResult(
                    page_number=page_number, error=f"page worker {why} on this page {self._attempts[page_number]} times"
                )
        if self._crash_respawns_left > 0:
            self._crash_respawns_left -= 1
            logger.warning("page_worker_replaced", worker=self._names[index], reason=why, respawns_left=self._crash_respawns_left)
            self._spawn_worker(index)
        else:
            logger.error("page_worker_not_replaced_budget_exhausted", worker=self._names[index], reason=why)
            self._names[index] = ""
        return given_up

    def results(self, expected_count: int, heartbeat: Callable[[], None] | None = None) -> Iterator[PageWorkResult]:
        """Yields exactly one `PageWorkResult` per submitted page (in
        completion order, NOT page-number order) until `expected_count`
        distinct pages are accounted for, or no worker can be kept alive.

        `heartbeat` is called at least every `_POLL_SECONDS` while waiting,
        so the caller can keep its document lock alive even while a single
        page is very slow -- renewing only after each finished page let the
        lock expire under memory pressure, and the job was then wrongly
        reported as "Processing stalled" while it was still running."""
        done: set[int] = set()
        last_beat = 0.0
        while len(done) < expected_count:
            now = time.monotonic()
            if heartbeat is not None and now - last_beat >= self._POLL_SECONDS:
                last_beat = now
                try:
                    heartbeat()
                except Exception as exc:  # noqa: BLE001 - a heartbeat problem must not stop page processing
                    logger.warning("page_worker_pool_heartbeat_failed", error=str(exc))

            self._dispatch()
            live = {self._conns[i]: i for i, name in enumerate(self._names) if name}
            if not live:
                logger.error("page_worker_pool_all_workers_dead", received=len(done), expected=expected_count)
                return

            for conn in mp_connection.wait(list(live), timeout=self._POLL_SECONDS):
                slot = live[conn]
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    continue  # the worker died; handled by the health check below
                if not isinstance(msg, PageWorkResult):
                    continue
                self._in_flight.pop(slot, None)
                if msg.page_number < 0:
                    # Worker could not even start (e.g. OCR engine init failed)
                    # and has exited; the health check below refills the slot.
                    logger.error("page_worker_init_failed", worker=msg.worker_name, error=msg.error)
                    continue
                if msg.page_number not in done:
                    done.add(msg.page_number)
                    yield msg
                if msg.worker_exiting and len(done) < expected_count:
                    self._retire(slot, kill=False)
                    logger.info("page_worker_pool_respawning", worker=msg.worker_name)
                    self._spawn_worker(slot)

            for slot, proc in enumerate(self._procs):
                if not self._names[slot]:
                    continue
                started = self._in_flight.get(slot)
                if not proc.is_alive():
                    given_up = self._handle_lost_worker(slot, f"died (exit code {proc.exitcode})")
                elif started is not None and time.monotonic() - started[1] > self.page_timeout_seconds:
                    given_up = self._handle_lost_worker(slot, f"hung for over {self.page_timeout_seconds}s")
                else:
                    continue
                if given_up is not None and given_up.page_number not in done:
                    done.add(given_up.page_number)
                    yield given_up

    def shutdown(self) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
            except Exception:  # noqa: BLE001 - worker already gone
                pass
        for proc in self._procs:
            proc.join(5)
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def alive_count(self) -> int:
        return sum(1 for p in self._procs if p.is_alive())
