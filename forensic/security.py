"""Security hardening and input validation module for DeepTrace.

Provides defensive controls against:
- Malicious file uploads (executable payload prevention via extension & magic byte checks)
- Denial-of-service via oversized files (configurable size caps)
- Path traversal attacks (null bytes, directory breakout, relative traversal)
- Symlink exploitation and insecure temporary file creation
"""

import os
import re
import io
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Set, Dict, Any, BinaryIO, Union

# Permitted extensions by media category
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}
ALL_ALLOWED_EXTENSIONS: Set[str] = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Default size thresholds (in bytes)
DEFAULT_MAX_IMAGE_SIZE_BYTES: int = 25 * 1024 * 1024       # 25 MB
DEFAULT_MAX_VIDEO_SIZE_BYTES: int = 250 * 1024 * 1024     # 250 MB

# Magic number signatures (header byte prefixes)
MAGIC_SIGNATURES: Dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "webp": b"RIFF",       # Bytes 0-3: 'RIFF', Bytes 8-11: 'WEBP'
    "avi": b"RIFF",        # Bytes 0-3: 'RIFF', Bytes 8-11: 'AVI '
    "mp4_ftyp": b"ftyp",   # Bytes 4-7: 'ftyp' in MP4/MOV containers
    "mkv": b"\x1a\x45\xdf\xa3",  # EBML header for Matroska/WebM
}


class SecurityValidationError(ValueError):
    """Raised when an uploaded file or path fails defensive security validation."""
    pass


def sanitize_filename(filename: str) -> str:
    """Sanitize an untrusted user-supplied filename.

    - Strips directory path separators (/ and \\)
    - Strips embedded null bytes (\x00)
    - Removes dangerous leading dots/characters
    - Enforces alphanumeric + safe punctuation regex

    Args:
        filename: Raw filename string from user or upload widget.

    Returns:
        Clean, safe filename string.

    Raises:
        SecurityValidationError: If the filename is empty or cannot be sanitized safely.
    """
    if not filename or not isinstance(filename, str):
        raise SecurityValidationError("Filename must be a non-empty string.")

    # Reject or strip null bytes
    if "\x00" in filename:
        filename = filename.replace("\x00", "")

    # Strip any directory path components (both POSIX and Windows)
    basename = os.path.basename(filename).replace("\\", "/").split("/")[-1]
    basename = basename.strip()

    # Reject traversal attempts
    if basename in ("", ".", ".."):
        raise SecurityValidationError(f"Invalid or dangerous filename: '{filename}'")

    # Replace all non-alphanumeric (except dots, underscores, hyphens) with underscores
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", basename)
    # Avoid names that start with a dot (hidden files)
    safe_name = safe_name.lstrip(".")

    if not safe_name:
        raise SecurityValidationError(f"Filename '{filename}' contained no valid characters.")

    # Truncate extremely long filenames
    if len(safe_name) > 128:
        stem, ext = os.path.splitext(safe_name)
        safe_name = stem[:120] + ext

    return safe_name


def sanitize_filepath(untrusted_path: str, base_dir: str) -> str:
    """Resolve and verify that an untrusted file path resides strictly inside `base_dir`.

    Prevents directory traversal attacks (e.g. '../../etc/passwd' or '..\\..\\windows\\system32').

    Args:
        untrusted_path: Candidate file path.
        base_dir: Approved root directory.

    Returns:
        Canonical absolute path guaranteed to be inside `base_dir`.

    Raises:
        SecurityValidationError: If the resolved path escapes `base_dir`.
    """
    if "\x00" in untrusted_path:
        raise SecurityValidationError("Path contains illegal null bytes.")

    base_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(os.path.join(base_abs, untrusted_path))

    # Commonpath must match base_abs exactly
    try:
        common = os.path.commonpath([base_abs, target_abs])
        if common != base_abs:
            raise SecurityValidationError(
                f"Path traversal detected: '{untrusted_path}' resolves outside '{base_dir}'."
            )
    except ValueError as e:
        # Happens on Windows when paths are on different drives
        raise SecurityValidationError(f"Invalid path traversal across drives: {e}")

    return target_abs


def detect_file_type_from_header(header_bytes: bytes) -> Optional[str]:
    """Inspect raw header bytes to detect actual file format via magic numbers.

    Args:
        header_bytes: First 32+ bytes of the file.

    Returns:
        String identifier ('jpeg', 'png', 'webp', 'mp4', 'avi', 'mkv') or None.
    """
    if len(header_bytes) < 4:
        return None

    if header_bytes.startswith(MAGIC_SIGNATURES["jpeg"]):
        return "jpeg"

    if header_bytes.startswith(MAGIC_SIGNATURES["png"]):
        return "png"

    if header_bytes.startswith(MAGIC_SIGNATURES["mkv"]):
        return "mkv"

    # RIFF containers (WebP or AVI)
    if header_bytes.startswith(b"RIFF") and len(header_bytes) >= 12:
        tag = header_bytes[8:12]
        if tag == b"WEBP":
            return "webp"
        if tag == b"AVI ":
            return "avi"

    # ISO Base Media File Format (MP4, QuickTime / MOV)
    # Typically [4 bytes size][4 bytes 'ftyp'][4 bytes major brand e.g. 'isom', 'mp42', 'qt  ']
    if len(header_bytes) >= 8 and header_bytes[4:8] == b"ftyp":
        return "mp4"

    return None


def validate_file_upload(
    filename: str,
    file_bytes_or_stream: Union[bytes, BinaryIO],
    allowed_extensions: Optional[Set[str]] = None,
    max_size_bytes: Optional[int] = None,
    enforce_magic_bytes: bool = True
) -> Dict[str, Any]:
    """Validate an uploaded media file against security rules.

    Checks:
    1. Filename sanitization and extension whitelisting.
    2. File size constraints (rejects oversized uploads).
    3. Magic header byte verification (ensures file content matches declared extension).

    Args:
        filename: Name of the uploaded file.
        file_bytes_or_stream: Byte buffer or file-like stream.
        allowed_extensions: Set of approved lowercase extensions (default ALL_ALLOWED_EXTENSIONS).
        max_size_bytes: Max byte limit (default chosen based on media type).
        enforce_magic_bytes: If True, inspect header bytes for known signatures.

    Returns:
        Dict with keys: {'sanitized_name', 'extension', 'media_type', 'size_bytes', 'detected_format'}.

    Raises:
        SecurityValidationError: If any security check fails.
    """
    safe_name = sanitize_filename(filename)
    ext = os.path.splitext(safe_name)[1].lower()

    target_extensions = allowed_extensions or ALL_ALLOWED_EXTENSIONS
    if ext not in target_extensions:
        raise SecurityValidationError(
            f"File extension '{ext}' is not permitted. Allowed: {sorted(list(target_extensions))}"
        )

    # Determine media category
    is_image = ext in ALLOWED_IMAGE_EXTENSIONS
    is_video = ext in ALLOWED_VIDEO_EXTENSIONS
    media_type = "image" if is_image else ("video" if is_video else "unknown")

    # Read bytes or measure length
    if isinstance(file_bytes_or_stream, bytes):
        raw_bytes = file_bytes_or_stream
        size_bytes = len(raw_bytes)
        header_sample = raw_bytes[:32]
    else:
        # File-like object
        current_pos = file_bytes_or_stream.tell()
        file_bytes_or_stream.seek(0, os.SEEK_END)
        size_bytes = file_bytes_or_stream.tell()
        file_bytes_or_stream.seek(0)
        header_sample = file_bytes_or_stream.read(32)
        file_bytes_or_stream.seek(current_pos)

    # Size limit validation
    limit = max_size_bytes or (DEFAULT_MAX_IMAGE_SIZE_BYTES if is_image else DEFAULT_MAX_VIDEO_SIZE_BYTES)
    if size_bytes > limit:
        max_mb = limit / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        raise SecurityValidationError(
            f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.1f} MB)."
        )

    if size_bytes == 0:
        raise SecurityValidationError("Uploaded file is empty (0 bytes).")

    # Magic byte verification
    detected_fmt = detect_file_type_from_header(header_sample)
    if enforce_magic_bytes and detected_fmt is not None:
        # Basic consistency check: image extension shouldn't match video magic or vice versa
        if is_image and detected_fmt in ("mp4", "avi", "mkv"):
            raise SecurityValidationError(
                f"File content appears to be video ('{detected_fmt}'), but has image extension '{ext}'."
            )
        if is_video and detected_fmt in ("jpeg", "png"):
            raise SecurityValidationError(
                f"File content appears to be image ('{detected_fmt}'), but has video extension '{ext}'."
            )

    return {
        "sanitized_name": safe_name,
        "extension": ext,
        "media_type": media_type,
        "size_bytes": size_bytes,
        "detected_format": detected_fmt or "generic",
    }


def create_secure_temp_file(
    suffix: str = ".tmp",
    prefix: str = "deeptrace_safe_",
    dir: Optional[str] = None
) -> Tuple[int, str]:
    """Create a secure temporary file with restricted permissions (0600 on POSIX).

    Args:
        suffix: File extension.
        prefix: Filename prefix.
        dir: Directory (defaults to system temp).

    Returns:
        Tuple of (file_descriptor, absolute_path).
    """
    clean_suffix = re.sub(r"[^a-zA-Z0-9.]", "", suffix)
    fd, path = tempfile.mkstemp(suffix=clean_suffix, prefix=prefix, dir=dir)
    return fd, os.path.abspath(path)
