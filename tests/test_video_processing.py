"""Unit tests for video decoding, frame sampling, and metadata inspection."""

import os
import tempfile
import numpy as np
import cv2
import pytest

from preprocessing.video_processor import VideoProcessor
from preprocessing.image_processor import ImageProcessor
from forensic.metadata import inspect_video_metadata


@pytest.fixture
def synthetic_video_path(tmp_path):
    """Generate a small 2-second synthetic MP4 video (30 fps, 60 frames total)."""
    vid_path = str(tmp_path / "test_synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(vid_path, fourcc, 30.0, (160, 120))

    for i in range(60):
        frame = np.full((120, 160, 3), (i * 4) % 255, dtype=np.uint8)
        # Add a circular marker
        cv2.circle(frame, (80, 60), 25, (200, 200, 200), -1)
        out.write(frame)
    out.release()
    return vid_path


def test_video_metadata_inspection(synthetic_video_path):
    """Test video metadata extraction (fps, frame count, duration, dimensions)."""
    meta = inspect_video_metadata(synthetic_video_path)
    assert meta["is_valid"] is True
    assert meta["width"] == 160
    assert meta["height"] == 120
    assert meta["frame_count"] == 60
    assert meta["fps"] == 30.0
    assert meta["duration_seconds"] == 2.0
    assert len(meta["sha256"]) == 64


def test_video_frame_sampling(synthetic_video_path):
    """Verify that VideoProcessor correctly samples frames at target fps."""
    img_proc = ImageProcessor(image_size=(224, 224), detector_name="fallback")
    video_proc = VideoProcessor(image_processor=img_proc, sampling_fps=1.0, max_frames=10)

    sampled = list(video_proc.stream_sampled_frames(synthetic_video_path))
    # 2-second video sampled at 1.0 fps should produce ~2 frames (frame 0 and frame 30)
    assert len(sampled) == 2
    assert sampled[0]["frame_index"] == 0
    assert sampled[0]["timestamp"] == 0.0
    assert sampled[1]["frame_index"] == 30
    assert sampled[1]["timestamp"] == 1.0


def test_video_processing_pipeline(synthetic_video_path, tmp_path):
    """Test full video analysis pipeline including face crops and metadata summary."""
    img_proc = ImageProcessor(image_size=(224, 224), detector_name="fallback")
    video_proc = VideoProcessor(image_processor=img_proc, sampling_fps=1.0)

    out_dir = str(tmp_path / "crops")
    res = video_proc.process_video(synthetic_video_path, output_dir=out_dir)

    assert res["num_sampled_frames"] == 2
    assert len(res["frames"]) == 2
    assert res["video_metadata"]["duration_seconds"] == 2.0
    for frame in res["frames"]:
        assert "timestamp" in frame
        assert "crop_bgr" in frame
        assert frame["crop_bgr"].shape == (224, 224, 3)
        assert os.path.exists(frame["saved_crop_path"])


def test_corrupted_video_handling(tmp_path):
    """Verify graceful handling when given non-video or corrupted data."""
    bad_vid = str(tmp_path / "fake.mp4")
    with open(bad_vid, "wb") as f:
        f.write(b"not a real video stream")

    video_proc = VideoProcessor()
    val = video_proc.validate_video_file(bad_vid)
    assert val["is_valid"] is False
    assert len(val["errors"]) > 0
