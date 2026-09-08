import io

import pytest
from PIL import Image

from app.utils.file_safety import UploadValidationError, safe_join, sanitize_filename, validate_upload


def test_sanitize_filename_strips_path_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\evil.pdf") == "evil.pdf"


def test_sanitize_filename_replaces_bad_chars():
    result = sanitize_filename("my report (final)!.pdf")
    assert "(" not in result and ")" not in result and "!" not in result


def test_safe_join_blocks_escape(tmp_path):
    root = str(tmp_path)
    with pytest.raises(UploadValidationError):
        safe_join(root, "../outside.txt")


def test_safe_join_allows_normal_path(tmp_path):
    root = str(tmp_path)
    result = safe_join(root, "sub", "file.txt")
    assert result.startswith(str(tmp_path))


def test_validate_upload_rejects_unsupported_extension():
    with pytest.raises(UploadValidationError):
        validate_upload("virus.exe", b"MZ\x90\x00")


def test_validate_upload_rejects_empty_file():
    with pytest.raises(UploadValidationError):
        validate_upload("empty.pdf", b"")


def test_validate_upload_rejects_corrupted_pdf():
    with pytest.raises(UploadValidationError):
        validate_upload("bad.pdf", b"not a real pdf")


def test_validate_upload_accepts_valid_png():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="white").save(buf, format="PNG")
    result = validate_upload("scan.png", buf.getvalue())
    assert result.extension == ".png"
    assert result.page_count == 1


def test_validate_upload_rejects_corrupted_image():
    with pytest.raises(UploadValidationError):
        validate_upload("scan.png", b"not a real png")
