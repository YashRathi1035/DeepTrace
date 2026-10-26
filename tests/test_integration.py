"""End-to-end integration test: full forensic pipeline from raw input to PDF report.

Covers the complete DeepTrace inference chain:
    Raw Image/Video → Face Preprocessing → Neural Inference → Temperature Scaling
    → Grad-CAM Explanation → Video Aggregation → Evidence Record → PDF Export

This test uses only in-memory synthetic fixtures (no real media files required)
so it runs without any dataset download.
"""

import os
import json
import tempfile
import numpy as np
import cv2
import pytest
import torch

from preprocessing.image_processor import ImageProcessor
from preprocessing.video_processor import VideoProcessor
from models.detector import build_detector
from models.checkpoint import save_checkpoint, load_checkpoint
from models.predict import DeepfakeDetector
from forensic.calibration import TemperatureScaler, ReliabilityDecisionPolicy
from forensic.aggregation import VideoForensicAnalyzer, aggregate_scores, analyze_temporal_consistency
from forensic.metadata import compute_sha256, inspect_media_file
from forensic.quality import evaluate_face_quality
from forensic.report import build_evidence_record, export_evidence_json, generate_forensic_pdf
from explainability.gradcam import generate_explanation


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_face_bgr():
    """Create a synthetic 224×224 face-like BGR image (gradient + skin tones)."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Simulate skin-tone face region
    cv2.ellipse(img, (112, 112), (80, 100), 0, 0, 360, (140, 170, 200), -1)
    # Simulate eyes
    cv2.circle(img, (80, 90), 12, (30, 30, 40), -1)
    cv2.circle(img, (144, 90), 12, (30, 30, 40), -1)
    # Simulate nose
    cv2.ellipse(img, (112, 130), (10, 14), 0, 0, 360, (100, 130, 155), -1)
    return img


@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory):
    """Create a synthetic 2-second MP4 video with moving synthetic face frames."""
    tmp_dir = tmp_path_factory.mktemp("integration_video")
    vid_path = str(tmp_dir / "test_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_path, fourcc, 10.0, (224, 224))

    for i in range(20):  # 20 frames @ 10 FPS = 2 seconds
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        offset = int(i * 2)
        cv2.ellipse(frame, (112 + offset % 10, 112), (80, 100), 0, 0, 360, (140, 170, 200), -1)
        cv2.circle(frame, (80, 90), 12, (30, 30, 40), -1)
        cv2.circle(frame, (144, 90), 12, (30, 30, 40), -1)
        writer.write(frame)
    writer.release()

    return vid_path


@pytest.fixture(scope="module")
def trained_detector(tmp_path_factory):
    """Build, save, and reload a minimal EfficientNet-B0 detector for integration testing."""
    tmp_dir = tmp_path_factory.mktemp("integration_model")
    chk_path = str(tmp_dir / "test_checkpoint.pt")

    model = build_detector(architecture="efficientnet_b0", num_classes=2, pretrained=False)

    # Minimal config and metrics matching save_checkpoint's actual signature
    save_checkpoint(
        model=model,
        optimizer=None,
        epoch=1,
        metrics={"val_loss": 0.5, "val_acc": 0.75},
        config={
            "model": {
                "architecture": "efficientnet_b0",
                "labels": ["Real", "Manipulated"],
                "dropout": 0.2,
            }
        },
        checkpoint_path=chk_path,
    )

    detector = DeepfakeDetector(checkpoint_path=chk_path, device="cpu")
    return detector


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Image preprocessing → face crop → quality check
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_preprocessing(synthetic_face_bgr, tmp_path):
    """Verify face preprocessing pipeline produces a valid normalized tensor and quality metrics."""
    processor = ImageProcessor()

    # Save synthetic face to temp PNG
    img_path = str(tmp_path / "synthetic_face.png")
    cv2.imwrite(img_path, synthetic_face_bgr)

    result = processor.process_face(img_path)

    assert "tensor" in result, "process_face must return a tensor"
    assert "quality" in result, "process_face must return quality metrics"

    tensor = result["tensor"]
    assert tensor.shape == (1, 3, 224, 224), f"Expected (1,3,224,224), got {tensor.shape}"
    assert tensor.dtype == torch.float32

    q = result["quality"]
    # ImageProcessor.assess_quality returns blur_score / brightness / quality_pass (not quality_score)
    assert "blur_score" in q, f"Expected 'blur_score' in quality dict, got keys: {list(q.keys())}"
    assert "brightness" in q
    assert "quality_pass" in q

    # SHA-256 verification
    sha = compute_sha256(img_path)
    assert len(sha) == 64, "SHA-256 must be 64 hex characters"

    # Metadata inspection
    meta = inspect_media_file(img_path)
    assert meta["file_type"] == "image"
    assert meta["width"] == 224
    assert meta["height"] == 224


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Neural inference → raw probabilities
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_neural_inference(trained_detector, synthetic_face_bgr, tmp_path):
    """Verify end-to-end neural inference produces valid structured predictions."""
    img_path = str(tmp_path / "face_for_inference.png")
    cv2.imwrite(img_path, synthetic_face_bgr)

    pred = trained_detector.predict_image(img_path)

    assert "manipulation_score" in pred
    assert "confidence" in pred
    assert "predicted_class" in pred
    assert "raw_scores" in pred
    assert "inference_time_ms" in pred

    m_score = pred["manipulation_score"]
    assert 0.0 <= m_score <= 1.0, f"manipulation_score {m_score} out of [0,1]"
    assert pred["predicted_class"] in ["Real", "Manipulated"]
    assert pred["inference_time_ms"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Temperature Scaling calibration → calibrated probabilities
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_temperature_scaling(trained_detector, synthetic_face_bgr, tmp_path):
    """Verify temperature scaler produces calibrated probabilities in valid range."""
    img_path = str(tmp_path / "face_for_calibration.png")
    cv2.imwrite(img_path, synthetic_face_bgr)

    pred = trained_detector.predict_image(img_path)
    logits_arr = np.array([pred["raw_scores"]], dtype=np.float32)

    scaler = TemperatureScaler(initial_temperature=1.5)
    cal_probs = scaler.calibrate_logits(logits_arr)

    assert cal_probs.shape == (1, 2), f"Expected shape (1,2), got {cal_probs.shape}"
    assert abs(cal_probs[0].sum() - 1.0) < 1e-5, "Calibrated probabilities must sum to 1.0"
    assert all(0.0 <= p <= 1.0 for p in cal_probs[0]), "All probabilities must be in [0, 1]"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Decision policy → three-tier verdict
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_decision_policy():
    """Verify decision policy correctly assigns the three evidential tiers."""
    policy = ReliabilityDecisionPolicy(fake_threshold=0.65, real_threshold=0.35)

    # Manipulated
    res = policy.decide(manipulation_score=0.82, quality_score=0.7)
    assert res["decision"] == "Potentially Manipulated"
    assert res["is_decisive"] is True

    # Authentic
    res = policy.decide(manipulation_score=0.12, quality_score=0.8)
    assert res["decision"] == "Likely Authentic"
    assert res["is_decisive"] is True

    # Inconclusive (ambiguous score)
    res = policy.decide(manipulation_score=0.50, quality_score=0.9)
    assert res["decision"] == "Inconclusive"
    assert res["is_decisive"] is False

    # Inconclusive (low quality)
    res = policy.decide(manipulation_score=0.80, quality_score=0.1)
    assert res["decision"] == "Inconclusive"
    assert res["is_decisive"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Grad-CAM visual attribution
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_gradcam(trained_detector, synthetic_face_bgr, tmp_path):
    """Verify Grad-CAM generates valid heatmap and overlay arrays."""
    img_path = str(tmp_path / "face_for_gradcam.png")
    cv2.imwrite(img_path, synthetic_face_bgr)

    proc = trained_detector.image_processor.process_face(img_path)
    tensor = proc["tensor"]
    crop_bgr = proc["crop_bgr"]

    cam_res = generate_explanation(
        model=trained_detector.model,
        input_tensor=tensor,
        original_bgr=crop_bgr,
        target_class=1
    )

    assert "heatmap" in cam_res
    assert "overlay_rgb" in cam_res
    assert "overlay_bgr" in cam_res

    heatmap = cam_res["heatmap"]
    assert heatmap.ndim == 2, "Heatmap must be 2D"
    assert heatmap.dtype == np.float32
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0, "Heatmap values must be in [0, 1]"

    overlay = cam_res["overlay_rgb"]
    assert overlay.shape == crop_bgr.shape, "Overlay must match input crop shape"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Video analysis → temporal evidence
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_video_analysis(trained_detector, synthetic_video_path):
    """Verify end-to-end video forensic analysis produces structured temporal evidence."""
    analyzer = VideoForensicAnalyzer(
        detector=trained_detector,
        default_aggregation="top_k",
        top_k_ratio=0.30,
        fake_threshold=0.65,
    )

    result = analyzer.analyze_video(synthetic_video_path, aggregation_method="mean")

    # Structure validation
    required_keys = [
        "video_level_score", "predicted_class", "per_frame_scores",
        "timestamps", "num_analyzed_frames", "temporal_consistency",
        "selected_aggregation_method", "processing_time_ms",
    ]
    for key in required_keys:
        assert key in result, f"Missing key in video result: {key}"

    assert 0.0 <= result["video_level_score"] <= 1.0
    assert result["predicted_class"] in ["Real", "Manipulated", "Inconclusive"]
    assert result["selected_aggregation_method"] == "mean"
    assert result["processing_time_ms"] > 0

    tc = result["temporal_consistency"]
    assert "mean" in tc
    assert "variance" in tc
    assert "suspicious_intervals" in tc
    assert isinstance(tc["suspicious_intervals"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Score aggregation policies
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_aggregation_policies():
    """Verify all four aggregation strategies produce valid scores."""
    scores = [0.1, 0.2, 0.9, 0.85, 0.3, 0.4, 0.1]

    for method in ["mean", "median", "top_k", "trimmed_mean"]:
        score = aggregate_scores(scores, method=method)
        assert 0.0 <= score <= 1.0, f"{method} produced out-of-range score: {score}"

    # Top-K should be >= mean (it takes the worst frames)
    top_k = aggregate_scores(scores, method="top_k", top_k_ratio=0.30)
    mean_s = aggregate_scores(scores, method="mean")
    assert top_k >= mean_s, "Top-K score should be >= mean for skewed distributions"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Evidence record compilation → JSON export
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_evidence_record(tmp_path, synthetic_face_bgr):
    """Verify evidence record contains all required forensic fields and exports to valid JSON."""
    img_path = str(tmp_path / "evidence_target.png")
    cv2.imwrite(img_path, synthetic_face_bgr)

    analysis_result = {
        "predicted_class": "Potentially Manipulated",
        "manipulation_score": 0.7823,
        "confidence": 0.7823,
        "raw_scores": [-0.9, 1.8],
        "inference_time_ms": 18.4,
        "quality_warnings": ["Low sharpness detected."],
        "is_calibrated": True,
    }

    record = build_evidence_record(
        media_path=img_path,
        analysis_result=analysis_result,
        model_name="efficientnet_b0",
        checkpoint_version="1.0",
    )

    # Required top-level keys
    required_keys = [
        "record_id", "timestamp_utc", "tool_name", "tool_version",
        "model_metadata", "file_observations", "model_findings",
        "explainability", "quality_warnings", "forensic_limitations",
    ]
    for key in required_keys:
        assert key in record, f"Missing key in evidence record: {key}"

    assert record["model_findings"]["manipulation_score"] == 0.7823
    assert "disclaimer" in record["forensic_limitations"]
    assert record["file_observations"]["sha256"] != ""

    # JSON export
    json_path = str(tmp_path / "evidence_output.json")
    saved = export_evidence_json(record, json_path)
    assert os.path.exists(saved)

    with open(saved, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["record_id"] == record["record_id"]
    assert loaded["model_findings"]["manipulation_score"] == 0.7823


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: PDF report generation with embedded images
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_pdf_report(tmp_path, synthetic_face_bgr):
    """Verify PDF report is generated with correct magic bytes and non-trivial size."""
    img_path = str(tmp_path / "pdf_target.png")
    cam_path = str(tmp_path / "pdf_cam.png")
    cv2.imwrite(img_path, synthetic_face_bgr)
    cv2.imwrite(cam_path, synthetic_face_bgr)

    analysis_result = {
        "predicted_class": "Likely Authentic",
        "manipulation_score": 0.21,
        "confidence": 0.79,
        "raw_scores": [1.5, -0.3],
        "inference_time_ms": 12.1,
        "quality_warnings": [],
        "is_calibrated": True,
    }

    record = build_evidence_record(
        media_path=img_path,
        analysis_result=analysis_result,
        model_name="efficientnet_b0",
        heatmap_paths=[cam_path],
    )

    pdf_path = str(tmp_path / "Forensic_Report.pdf")
    out_path = generate_forensic_pdf(
        evidence_record=record,
        output_pdf_path=pdf_path,
        visual_paths=[img_path, cam_path],
    )

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 4000, "PDF should be multi-page (>4 KB)"

    with open(out_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-", f"Invalid PDF magic bytes: {header}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Full image forensic pipeline (E2E)
# ─────────────────────────────────────────────────────────────────────────────
def test_integration_full_image_pipeline(trained_detector, tmp_path, synthetic_face_bgr):
    """Execute the complete image forensic chain in sequence and validate all outputs."""
    # Step 1: Save input
    img_path = str(tmp_path / "e2e_face.png")
    cv2.imwrite(img_path, synthetic_face_bgr)

    # Step 2: Metadata audit
    meta = inspect_media_file(img_path)
    assert meta["sha256"] != ""
    assert meta["file_type"] == "image"

    # Step 3: Inference
    pred = trained_detector.predict_image(img_path)
    assert "manipulation_score" in pred

    # Step 4: Calibration
    scaler = TemperatureScaler(initial_temperature=1.2)
    logits_arr = np.array([pred["raw_scores"]], dtype=np.float32)
    cal_probs = scaler.calibrate_logits(logits_arr)
    cal_score = float(cal_probs[0][1])
    assert 0.0 <= cal_score <= 1.0

    # Step 5: Quality gate
    q_eval = evaluate_face_quality(pred["crop_bgr"])
    assert "quality_score" in q_eval

    # Step 6: Decision
    policy = ReliabilityDecisionPolicy(fake_threshold=0.65, real_threshold=0.35)
    decision = policy.decide(
        manipulation_score=cal_score,
        quality_score=q_eval["quality_score"],
        quality_warnings=q_eval["warnings"],
    )
    assert decision["decision"] in ["Likely Authentic", "Potentially Manipulated", "Inconclusive"]

    # Step 7: Grad-CAM
    proc = trained_detector.image_processor.process_face(img_path)
    cam_res = generate_explanation(
        model=trained_detector.model,
        input_tensor=proc["tensor"],
        original_bgr=pred["crop_bgr"],
        target_class=1,
    )
    assert "overlay_bgr" in cam_res

    # Step 8: Evidence record
    record = build_evidence_record(
        media_path=img_path,
        analysis_result={
            "predicted_class": decision["decision"],
            "manipulation_score": cal_score,
            "confidence": pred["confidence"],
            "raw_scores": pred["raw_scores"],
            "inference_time_ms": pred["inference_time_ms"],
            "quality_warnings": decision["warnings"],
            "is_calibrated": True,
        },
        model_name=trained_detector.architecture,
        heatmap_paths=[],
    )
    assert record["model_findings"]["manipulation_score"] == pytest.approx(cal_score, abs=1e-3)

    # Step 9: PDF export
    cam_save = str(tmp_path / "e2e_cam.png")
    cv2.imwrite(cam_save, cam_res["overlay_bgr"])

    pdf_path = str(tmp_path / "E2E_Report.pdf")
    out_pdf = generate_forensic_pdf(record, pdf_path, visual_paths=[img_path, cam_save])

    assert os.path.exists(out_pdf)
    with open(out_pdf, "rb") as f:
        assert f.read(5) == b"%PDF-"
