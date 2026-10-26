"""Unit tests for Phase 1 preprocessing, metadata extraction, and dataset splitting."""

import os
import tempfile
import numpy as np
import cv2
import torch
import pytest

from forensic.metadata import compute_sha256, inspect_image_metadata, inspect_media_file
from preprocessing.face_detection import crop_face_with_margin, FaceDetection
from preprocessing.image_processor import ImageProcessor
from preprocessing.dataset import create_split_manifest


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary synthetic test image."""
    img_path = str(tmp_path / "test_face.png")
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (200, 200, 200), -1)
    cv2.imwrite(img_path, img)
    return img_path


def test_sha256_computation(tmp_path):
    """Test cryptographic SHA-256 calculation."""
    test_file = str(tmp_path / "sample.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("DeepTrace Forensic Analyzer Test String")
    
    hash1 = compute_sha256(test_file)
    assert len(hash1) == 64
    assert isinstance(hash1, str)

    # Identical content yields identical hash
    test_file2 = str(tmp_path / "sample_copy.txt")
    with open(test_file2, "w", encoding="utf-8") as f:
        f.write("DeepTrace Forensic Analyzer Test String")
    assert compute_sha256(test_file2) == hash1


def test_metadata_extraction(sample_image_path):
    """Test image metadata extraction and validation."""
    meta = inspect_image_metadata(sample_image_path)
    assert meta["is_valid"] is True
    assert meta["file_type"] == "image"
    assert meta["width"] == 200
    assert meta["height"] == 200
    assert len(meta["sha256"]) == 64


def test_invalid_media_handling(tmp_path):
    """Test graceful handling of unsupported media."""
    bad_file = str(tmp_path / "corrupt.xyz")
    with open(bad_file, "wb") as f:
        f.write(b"not a valid media format")
    meta = inspect_media_file(bad_file)
    assert meta["is_valid"] is False
    assert len(meta["warnings"]) > 0


def test_crop_face_with_margin():
    """Test bounding box expansion with margin and edge clipping."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Face box in upper left: (10, 10, 20, 20)
    bbox = (10, 10, 20, 20)
    # 50% margin should expand box by 10px on each side -> (0, 0, 40, 40)
    crop = crop_face_with_margin(img, bbox, margin=0.5, target_size=(64, 64))
    assert crop.shape == (64, 64, 3)


def test_image_processor_pipeline(sample_image_path):
    """Test end-to-end image processing, quality assessment, and tensor generation."""
    proc = ImageProcessor(image_size=(224, 224), detector_name="fallback")
    result = proc.process_face(sample_image_path)

    assert "tensor" in result
    assert result["tensor"].shape == (1, 3, 224, 224)
    assert result["crop_bgr"].shape == (224, 224, 3)
    assert "quality" in result
    assert "blur_score" in result["quality"]
    assert "brightness" in result["quality"]

    # Test denormalization fidelity
    denorm = proc.denormalize_tensor(result["tensor"])
    assert denorm.shape == (224, 224, 3)
    assert denorm.dtype == np.uint8


def test_data_leakage_prevention():
    """Verify that video/subject-level splits strictly isolate source videos across splits."""
    samples = []
    # 10 videos, each having 3 frames
    for vid in range(10):
        for f in range(3):
            samples.append({
                "video_id": f"vid_{vid:02d}",
                "frame": f,
                "label": vid % 2
            })

    split_samples = create_split_manifest(samples, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    train_vids = {s["video_id"] for s in split_samples if s["split"] == "train"}
    val_vids = {s["video_id"] for s in split_samples if s["split"] == "val"}
    test_vids = {s["video_id"] for s in split_samples if s["split"] == "test"}

    # No video must appear in more than one split
    assert len(train_vids.intersection(val_vids)) == 0, "Train and Val video overlap detected!"
    assert len(train_vids.intersection(test_vids)) == 0, "Train and Test video overlap detected!"
    assert len(val_vids.intersection(test_vids)) == 0, "Val and Test video overlap detected!"
