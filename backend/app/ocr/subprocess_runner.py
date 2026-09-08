"""
Runs OCR inference in an isolated, long-lived subprocess.

Why this exists: a native inference backend (PaddlePaddle's C++ engine in
particular) can terminate the *entire process* with a raw OS signal — e.g.
SIGILL when the prebuilt wheel was compiled for CPU instructions (AVX2/
AVX512) the host doesn't actually expose, which is a real, observed failure
mode under some Docker/virtualization setups. That kind of crash happens
below the Python interpreter and is invisible to `try/except`, so an
in-process `except Exception` fallback (spec section 29) cannot catch it —
it would take the whole Celery worker process down mid-document.

Running inference in a separate child process turns that hard crash into
something the parent CAN observe and act on: the child simply exits
non-zero (or dies), the parent notices, discards that page's result, marks
the engine dead for the rest of this job, and falls back to Tesseract
in-process for the remaining pages. The subprocess is kept alive across
pages (rather than respawned per page) so PaddleOCR's model-loading cost is
paid once per document, not once per page.
"""
from __future__ import annotations

import pickle
import queue
import traceback
from dataclasses import dataclass

import billiard as mp  # Celery's prefork pool workers are themselves daemonic
# processes, and stdlib multiprocessing refuses to let a daemonic process
# spawn children ("daemonic processes are not allowed to have children").
# billiard is Celery's own fork of multiprocessing that lifts that
# restriction — it is already a transitive dependency of `celery` and is
# the standard way to run subprocesses from inside a Celery task.
import cv2
import numpy as np

from app.core.logging import get_logger
from app.schemas.ocr import OCRPageResult

logger = get_logger(__name__)

_TASK_TIMEOUT_SECONDS = 180
_STARTUP_TIMEOUT_SECONDS = 300  # first task also pays PaddleOCR's one-time model load/download cost


@dataclass
class _Task:
    image_png: bytes
    page_number: int
    dpi: int


def _child_main(provider_name: str, task_q: "mp.Queue", result_q: "mp.Queue") -> None:
    """Entry point for the isolated subprocess. Never raises out of this
    function on a normal OCR error — it reports {"error": ...} back and
    keeps serving the next task; only a signal-level crash (SIGILL/SIGSEGV)
    takes this process down, which the parent observes via `is_alive()`."""
    from app.ocr.factory import get_ocr_provider

    try:
        provider = get_ocr_provider(provider_name)
    except Exception as exc:  # noqa: BLE001
        result_q.put({"error": f"provider init failed: {exc}"})
        return

    while True:
        task = task_q.get()
        if task is None:  # sentinel: shut down
            return
        try:
            image = cv2.imdecode(np.frombuffer(task.image_png, dtype=np.uint8), cv2.IMREAD_COLOR)
            page_result = provider.recognize_page(image, task.page_number, task.dpi)
            result_q.put({"ok": pickle.dumps(page_result)})
        except Exception as exc:  # noqa: BLE001 - a normal (non-crash) OCR error for this page
            result_q.put({"error": f"{exc}\n{traceback.format_exc()}"})


class IsolatedOCRWorker:
    """One subprocess per processing job. Call `run_page()` per page; on a
    hard crash it returns None and marks itself dead — check `.alive`
    afterwards and stop calling it once False."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.alive = True
        self._task_q = mp.Queue()
        self._result_q = mp.Queue()
        self._proc = mp.Process(target=_child_main, args=(provider_name, self._task_q, self._result_q), daemon=True)
        self._proc.start()
        self._first_task = True

    def run_page(self, image: np.ndarray, page_number: int, dpi: int) -> OCRPageResult | None:
        if not self.alive:
            return None

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return None

        timeout = _STARTUP_TIMEOUT_SECONDS if self._first_task else _TASK_TIMEOUT_SECONDS
        self._first_task = False

        self._task_q.put(_Task(image_png=encoded.tobytes(), page_number=page_number, dpi=dpi))
        try:
            outcome = self._result_q.get(timeout=timeout)
        except queue.Empty:
            logger.error("ocr_subprocess_timeout", provider=self.provider_name, page=page_number)
            self._kill()
            return None

        if "error" in outcome:
            logger.warning("ocr_subprocess_page_error", provider=self.provider_name, page=page_number, error=outcome["error"])
            return None  # a normal per-page OCR error, not a crash — worker stays alive for the next page

        if not self._proc.is_alive() and self._result_q.empty():
            # Belt-and-braces: the child could have crashed right after
            # emitting a result for a prior task, e.g. corrupting state that
            # only surfaces on the NEXT call. Nothing to do here — normal
            # path already handled above; kept for clarity.
            pass

        return pickle.loads(outcome["ok"])

    def shutdown(self) -> None:
        if self._proc.is_alive():
            try:
                self._task_q.put(None)
                self._proc.join(5)
            except Exception:  # noqa: BLE001
                pass
        if self._proc.is_alive():
            self._proc.terminate()
        self.alive = False

    def _kill(self) -> None:
        logger.error("ocr_subprocess_crashed", provider=self.provider_name)
        try:
            self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        self.alive = False

    def poll_crashed(self) -> bool:
        """Call after run_page() returns None to distinguish "this page
        failed but the engine is still up" from "the engine died"."""
        if self.alive and not self._proc.is_alive():
            self._kill()
        return not self.alive
