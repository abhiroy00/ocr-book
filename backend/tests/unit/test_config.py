from unittest.mock import MagicMock, patch

from app.core.config import Settings

# `resolved_ocr_workers` is bounded by CPU AND by actually-available RAM
# (see the property's docstring for why: a real incident where 4 OCR
# workers pushed a 7.5GB Docker Desktop VM to 6.4GB and crashed the
# engine). Most of these tests mock RAM to be effectively unlimited so
# they isolate the CPU-budget behavior being tested; the RAM-specific
# tests mock it scarce instead to isolate that path.
_AMPLE_RAM_MB = 10 ** 6  # ~1TB "available" -- never the limiting factor


def _mock_ram(available_mb: float):
    mem = MagicMock()
    mem.available = available_mb * 1024 * 1024
    return patch("psutil.virtual_memory", return_value=mem)


def test_resolved_ocr_workers_honors_explicit_value_when_resources_allow():
    s = Settings(ocr_workers=3)
    with patch("os.cpu_count", return_value=8), _mock_ram(_AMPLE_RAM_MB):
        assert s.resolved_ocr_workers == 3


def test_resolved_ocr_workers_auto_picks_max_workers_ceiling_when_zero():
    s = Settings(ocr_workers=0, ocr_max_workers=10, ocr_reserved_cpu=2)
    with patch("os.cpu_count", return_value=14), _mock_ram(_AMPLE_RAM_MB):
        assert s.resolved_ocr_workers == 10  # cpu_budget=12, requested=10 -> min is 10


def test_resolved_ocr_workers_never_goes_below_configured_minimum():
    s = Settings(ocr_workers=0, ocr_min_workers=1)
    with patch("os.cpu_count", return_value=1), _mock_ram(_AMPLE_RAM_MB):
        assert s.resolved_ocr_workers == 1


def test_resolved_ocr_workers_cpu_budget_reserves_configured_cores():
    """A bounded pool is the whole point -- OCR must never claim every
    core, starving Postgres/Redis/the web server."""
    s = Settings(ocr_workers=0, ocr_max_workers=64, ocr_reserved_cpu=4)
    with patch("os.cpu_count", return_value=64), _mock_ram(_AMPLE_RAM_MB):
        assert s.resolved_ocr_workers == 60  # 64 - 4 reserved


def test_resolved_ocr_workers_is_limited_by_available_ram_not_just_cpu():
    """Regression test for the real incident this exists to prevent: 4 OCR
    workers (each loading its own PaddleOCR model -- real, non-shareable
    memory) pushed a Docker Desktop VM to its ceiling and crashed the
    engine, even though CPU cores were still idle. RAM must be able to
    override an otherwise-CPU-sized request."""
    s = Settings(ocr_workers=0, ocr_max_workers=10, ocr_reserved_cpu=2, ocr_reserved_memory_mb=1000, ocr_worker_est_memory_mb=4800)
    # Plenty of CPU (14 cores -> budget 12) but only ~10.6GB RAM available,
    # matching the real measurement this constant is based on.
    with patch("os.cpu_count", return_value=14), _mock_ram(10600):
        # usable = 10600 - 1000 = 9600; 9600 // 4800 = 2
        assert s.resolved_ocr_workers == 2


def test_resolved_ocr_workers_falls_back_to_minimum_under_severe_memory_pressure():
    s = Settings(ocr_workers=0, ocr_max_workers=10, ocr_min_workers=1, ocr_reserved_memory_mb=1000, ocr_worker_est_memory_mb=4800)
    with patch("os.cpu_count", return_value=14), _mock_ram(1200):  # barely more than the reserve
        assert s.resolved_ocr_workers == 1  # floored at ocr_min_workers, never 0


def test_resolved_ocr_workers_degrades_to_cpu_only_heuristic_if_psutil_unavailable():
    """A psutil failure must never crash config loading or block OCR
    entirely -- degrade to the old CPU-only heuristic instead."""
    s = Settings(ocr_workers=0)
    with patch("os.cpu_count", return_value=8), patch("psutil.virtual_memory", side_effect=RuntimeError("boom")):
        assert s.resolved_ocr_workers == 4  # min(6, 8 // 2)


def test_resolved_ocr_workers_auto_fallback_false_bypasses_ram_check():
    """The explicit escape hatch for an operator who has already sized
    their own host -- must honor OCR_WORKERS literally even under memory
    pressure that would otherwise clamp it."""
    s = Settings(ocr_workers=8, ocr_auto_fallback=False)
    with patch("os.cpu_count", return_value=14), _mock_ram(500):  # would otherwise force this down to 1
        assert s.resolved_ocr_workers == 8


def test_ocr_worker_sizing_debug_reports_ram_as_the_limiting_factor():
    s = Settings(ocr_workers=0, ocr_max_workers=10, ocr_reserved_cpu=2, ocr_reserved_memory_mb=1000, ocr_worker_est_memory_mb=4800)
    with patch("os.cpu_count", return_value=14), _mock_ram(10600):
        debug = s.ocr_worker_sizing_debug
    assert debug["chosen"] == 2
    assert debug["limiting_factor"] == "ram"
    assert debug["cpu_budget"] == 12
