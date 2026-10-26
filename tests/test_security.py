"""Unit tests for forensic.security module.

Validates defensive security controls:
- Whitelist extension filtering (rejects .exe, .sh, .py, .bat, etc.)
- File size caps
- Header magic byte detection and consistency enforcement
- Path traversal mitigation (null bytes, directory breakouts)
- Filename sanitization
- Secure temporary file creation
"""

import os
import io
import pytest
from forensic.security import (
    sanitize_filename,
    sanitize_filepath,
    detect_file_type_from_header,
    validate_file_upload,
    create_secure_temp_file,
    SecurityValidationError,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)


def test_sanitize_filename_normal():
    assert sanitize_filename("my_image.png") == "my_image.png"
    assert sanitize_filename("Face-Sample_01.JPEG") == "Face-Sample_01.JPEG"


def test_sanitize_filename_traversal():
    # POSIX and Windows traversal patterns
    assert sanitize_filename("../../../etc/passwd.jpg") == "passwd.jpg"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe.png") == "cmd.exe.png"
    assert sanitize_filename("./local_file.mp4") == "local_file.mp4"


def test_sanitize_filename_null_bytes():
    # Embedded null bytes must be stripped
    clean = sanitize_filename("suspicious\x00_test.png")
    assert "\x00" not in clean
    assert clean == "suspicious_test.png"


def test_sanitize_filename_invalid_inputs():
    with pytest.raises(SecurityValidationError):
        sanitize_filename("")

    with pytest.raises(SecurityValidationError):
        sanitize_filename("...")

    with pytest.raises(SecurityValidationError):
        sanitize_filename("///")


def test_sanitize_filepath_valid(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    target = sub / "file.txt"
    target.write_text("ok")

    safe_path = sanitize_filepath("sub/file.txt", str(tmp_path))
    assert os.path.exists(safe_path)
    assert os.path.commonpath([str(tmp_path), safe_path]) == str(tmp_path)


def test_sanitize_filepath_traversal_blocked(tmp_path):
    with pytest.raises(SecurityValidationError):
        sanitize_filepath("../../outside.txt", str(tmp_path))


def test_detect_magic_bytes():
    # JPEG
    jpeg_hdr = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    assert detect_file_type_from_header(jpeg_hdr) == "jpeg"

    # PNG
    png_hdr = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert detect_file_type_from_header(png_hdr) == "png"

    # WebP (RIFF....WEBP)
    webp_hdr = b"RIFF\x20\x00\x00\x00WEBPVP8 "
    assert detect_file_type_from_header(webp_hdr) == "webp"

    # MP4 (....ftypisom)
    mp4_hdr = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00"
    assert detect_file_type_from_header(mp4_hdr) == "mp4"


def test_validate_file_upload_success():
    # Real valid PNG header + dummy bytes
    valid_png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    res = validate_file_upload("sample_photo.png", valid_png_bytes)
    assert res["sanitized_name"] == "sample_photo.png"
    assert res["extension"] == ".png"
    assert res["media_type"] == "image"
    assert res["size_bytes"] == 108
    assert res["detected_format"] == "png"


def test_validate_file_upload_disallowed_extension():
    data = b"print('malicious')"
    with pytest.raises(SecurityValidationError, match="extension '.py' is not permitted"):
        validate_file_upload("exploit.py", data)

    with pytest.raises(SecurityValidationError, match="extension '.exe' is not permitted"):
        validate_file_upload("trojan.exe", data)


def test_validate_file_upload_size_limit():
    data = b"\x00" * 1024  # 1 KB
    # Set artificial 500-byte cap
    with pytest.raises(SecurityValidationError, match="exceeds maximum allowed limit"):
        validate_file_upload("test.png", data, max_size_bytes=500)


def test_validate_file_upload_empty_file():
    with pytest.raises(SecurityValidationError, match="empty"):
        validate_file_upload("empty.jpg", b"")


def test_validate_file_upload_type_mismatch():
    # MP4 header but declared as PNG
    mp4_hdr = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00" + b"\x00" * 50
    with pytest.raises(SecurityValidationError, match="appears to be video"):
        validate_file_upload("disguised.png", mp4_hdr, enforce_magic_bytes=True)


def test_create_secure_temp_file():
    fd, path = create_secure_temp_file(suffix=".png")
    try:
        assert os.path.exists(path)
        assert path.endswith(".png")
    finally:
        os.close(fd)
        if os.path.exists(path):
            os.remove(path)
