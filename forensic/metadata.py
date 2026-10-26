"""Digital media forensic metadata inspection and cryptographic hashing module."""

import os
import hashlib
from typing import Dict, Any, Optional
import cv2
from PIL import Image, ExifTags


def compute_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 cryptographic hash of a file.
    
    Args:
        file_path: Path to the target file.
        chunk_size: Buffer read size in bytes.
        
    Returns:
        Hexadecimal string of SHA-256 hash.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def inspect_image_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata, dimensions, and EXIF tags from an image file.
    
    Args:
        file_path: Path to the image file.
        
    Returns:
        Structured dictionary of image properties and EXIF data.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image not found: {file_path}")

    metadata: Dict[str, Any] = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size_bytes": os.path.getsize(file_path),
        "file_type": "image",
        "sha256": compute_sha256(file_path),
        "is_valid": False,
        "warnings": [],
        "exif": {}
    }

    try:
        with Image.open(file_path) as img:
            metadata["format"] = img.format
            metadata["mode"] = img.mode
            metadata["width"], metadata["height"] = img.size
            metadata["is_valid"] = True

            # Extract EXIF if available
            raw_exif = img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Convert bytes to string for JSON serialization
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="ignore")
                        except Exception:
                            value = str(value)
                    metadata["exif"][str(tag_name)] = str(value)
            else:
                metadata["warnings"].append("No EXIF metadata found (common in web-transferred or scrubbed media).")

    except Exception as e:
        metadata["warnings"].append(f"Failed to read image structure: {str(e)}")
        metadata["is_valid"] = False

    return metadata


def inspect_video_metadata(file_path: str) -> Dict[str, Any]:
    """Inspect video properties, container metadata, duration, and codecs using OpenCV.
    
    Args:
        file_path: Path to the video file.
        
    Returns:
        Structured dictionary with video duration, resolution, fps, frame count, codec.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Video not found: {file_path}")

    metadata: Dict[str, Any] = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size_bytes": os.path.getsize(file_path),
        "file_type": "video",
        "sha256": compute_sha256(file_path),
        "is_valid": False,
        "warnings": []
    }

    cap = cv2.VideoCapture(file_path)
    try:
        if not cap.isOpened():
            metadata["warnings"].append("Could not open video stream; file may be corrupted or codec unsupported.")
            return metadata

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        duration_sec = (frame_count / fps) if fps > 0 else 0.0

        metadata.update({
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_seconds": round(duration_sec, 2),
            "fourcc": fourcc.strip(),
            "is_valid": frame_count > 0 and width > 0 and height > 0
        })

        if frame_count <= 0:
            metadata["warnings"].append("Zero frames reported in video container.")
        if duration_sec < 0.5:
            metadata["warnings"].append("Video is extremely short (< 0.5s); temporal evidence will be constrained.")

    except Exception as e:
        metadata["warnings"].append(f"Error parsing video properties: {str(e)}")
        metadata["is_valid"] = False
    finally:
        cap.release()

    return metadata


def inspect_media_file(file_path: str) -> Dict[str, Any]:
    """Unified entry point to inspect either image or video media.
    
    Args:
        file_path: Path to the target media file.
        
    Returns:
        Structured forensic metadata record.
    """
    ext = os.path.splitext(file_path)[1].lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    if ext in video_exts:
        return inspect_video_metadata(file_path)
    elif ext in image_exts:
        return inspect_image_metadata(file_path)
    else:
        # Fallback inspection
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "file_type": "unknown",
            "sha256": compute_sha256(file_path) if os.path.exists(file_path) else "",
            "is_valid": False,
            "warnings": [f"Unsupported media extension '{ext}'."]
        }
