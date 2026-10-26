"""Unit and integration tests for video score aggregation, temporal consistency, and timeline plots."""

import os
import tempfile
import numpy as np
import cv2
import torch
import pytest

from forensic.aggregation import (
    compute_iou,
    aggregate_scores,
    analyze_temporal_consistency,
    VideoForensicAnalyzer
)
from models.detector import build_detector
from models.checkpoint import save_checkpoint
from models.predict import DeepfakeDetector
from preprocessing.image_processor import ImageProcessor
from preprocessing.video_processor import VideoProcessor
from evaluation.plots import plot_video_score_timeline


def test_iou_calculation():
    """Verify Intersection over Union for identical, partial, and disjoint boxes."""
    box_a = (10, 10, 50, 50)
    # Identical
    assert compute_iou(box_a, box_a) == 1.0

    # Disjoint
    box_b = (100, 100, 50, 50)
    assert compute_iou(box_a, box_b) == 0.0

    # Half overlap: box_c starts at x=35 (overlap w=25, h=50 -> inter=1250, union=3750 -> 0.333)
    box_c = (35, 10, 50, 50)
    iou = compute_iou(box_a, box_c)
    assert 0.30 <= iou <= 0.36


def test_aggregation_strategies():
    """Test mean, median, top_k, and trimmed_mean behavior on uniform and localized scores."""
    # Localized manipulation scenario: 8 clean frames (0.10) and 2 fake frames (0.90)
    scores = [0.10] * 8 + [0.90, 0.90]

    # Mean drowns out the localized manipulation
    mean_score = aggregate_scores(scores, method="mean")
    assert round(mean_score, 2) == 0.26

    # Median completely ignores the localized manipulation
    median_score = aggregate_scores(scores, method="median")
    assert round(median_score, 2) == 0.10

    # Top-K (top 20% = top 2 frames) correctly alerts on localized deepfakes!
    top_k_score = aggregate_scores(scores, method="top_k", top_k_ratio=0.20)
    assert round(top_k_score, 2) == 0.90

    # Trimmed mean
    trimmed = aggregate_scores(scores, method="trimmed_mean")
    assert isinstance(trimmed, float)

    # Empty edge case
    assert aggregate_scores([], method="mean") == 0.0


def test_temporal_consistency_analysis():
    """Verify detection of temporal manipulation spikes and volatility."""
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # Manipulation burst between 2.0s and 3.0s
    scores = [0.15, 0.18, 0.85, 0.92, 0.20, 0.12]

    stats = analyze_temporal_consistency(scores, timestamps, suspicious_threshold=0.65)

    assert stats["mean"] > 0
    assert stats["variance"] > 0
    assert stats["max_consecutive_delta"] >= 0.65
    assert len(stats["suspicious_intervals"]) == 1

    interval = stats["suspicious_intervals"][0]
    assert interval["start_seconds"] == 2.0
    assert interval["end_seconds"] == 3.0
    assert interval["peak_manipulation_score"] == 0.92
    assert stats["volatility_warning"] is True


@pytest.fixture
def test_video_and_detector(tmp_path):
    """Create a synthetic test video and a minimal detector checkpoint for integration testing."""
    # 1. Synthetic video
    vid_path = str(tmp_path / "forensic_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(vid_path, fourcc, 10.0, (160, 120))
    for i in range(30):  # 3 seconds at 10 fps
        frame = np.full((120, 160, 3), 60, dtype=np.uint8)
        # Face ellipse
        cv2.ellipse(frame, (80, 60), (30, 40), 0, 0, 360, (180, 200, 220), -1)
        out.write(frame)
    out.release()

    # 2. Checkpoint
    chk_path = str(tmp_path / "model.pt")
    model = build_detector("efficientnet_b0", num_classes=2, pretrained=False)
    save_checkpoint(
        model=model,
        optimizer=None,
        epoch=1,
        metrics={},
        config={"model": {"labels": ["Real", "Manipulated"]}},
        checkpoint_path=chk_path
    )

    image_proc = ImageProcessor(image_size=(224, 224), detector_name="fallback")
    detector = DeepfakeDetector(checkpoint_path=chk_path, device="cpu", image_processor=image_proc)
    video_proc = VideoProcessor(image_processor=image_proc, sampling_fps=1.0)

    return vid_path, detector, video_proc


def test_video_forensic_analyzer_integration(test_video_and_detector, tmp_path):
    """Test full end-to-end video forensic analysis and timeline visualization."""
    vid_path, detector, video_proc = test_video_and_detector

    analyzer = VideoForensicAnalyzer(
        detector=detector,
        video_processor=video_proc,
        default_aggregation="top_k",
        top_k_ratio=0.33,
        fake_threshold=0.65
    )

    result = analyzer.analyze_video(vid_path)

    assert "video_level_score" in result
    assert 0.0 <= result["video_level_score"] <= 1.0
    assert result["predicted_class"] in ["Real", "Manipulated"]
    assert result["num_analyzed_frames"] == 3
    assert len(result["per_frame_scores"]) == 3
    assert len(result["timestamps"]) == 3
    assert "temporal_consistency" in result
    assert result["processing_time_ms"] > 0.0

    # Test timeline plot generation
    timeline_path = str(tmp_path / "timeline.png")
    plot_path = plot_video_score_timeline(
        timestamps=result["timestamps"],
        scores=[r["manipulation_score"] for r in result["per_frame_scores"]],
        video_level_score=result["video_level_score"],
        threshold=0.65,
        suspicious_intervals=result["temporal_consistency"]["suspicious_intervals"],
        output_path=timeline_path
    )
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 1000
