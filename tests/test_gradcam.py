"""Automated unit tests for Grad-CAM explainability, quality assessment, and evidence records."""

import os
import json
import tempfile
import numpy as np
import cv2
import torch
import pytest

from models.detector import build_detector
from explainability.gradcam import GradCAM, generate_explanation
from forensic.quality import evaluate_face_quality
from forensic.report import build_evidence_record, export_evidence_json


@pytest.fixture
def dummy_model():
    """Create a lightweight EfficientNet-B0 model for explainability testing."""
    model = build_detector("efficientnet_b0", num_classes=2, pretrained=False)
    model.eval()
    return model


def test_gradcam_generation(dummy_model):
    """Verify Grad-CAM saliency map generation, shape consistency, and value normalization."""
    grad_cam = GradCAM(model=dummy_model)
    x = torch.randn(1, 3, 224, 224)

    heatmap = grad_cam.generate_cam(x, target_class=1)
    grad_cam.remove_hooks()

    assert heatmap.shape == (224, 224)
    assert isinstance(heatmap, np.ndarray)
    assert 0.0 <= np.min(heatmap)
    assert np.max(heatmap) <= 1.0


def test_gradcam_overlay(dummy_model):
    """Verify heatmap blending and color mapping onto original image."""
    grad_cam = GradCAM(model=dummy_model)
    x = torch.randn(1, 3, 224, 224)
    heatmap = grad_cam.generate_cam(x, target_class=1)
    grad_cam.remove_hooks()

    dummy_bgr = np.full((224, 224, 3), 100, dtype=np.uint8)
    overlay, colored_cam = grad_cam.overlay_heatmap(dummy_bgr, heatmap, alpha=0.4)

    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8
    assert colored_cam.shape == (224, 224, 3)


def test_generate_explanation_artifact_saving(dummy_model, tmp_path):
    """Test unified generate_explanation wrapper and file saving."""
    x = torch.randn(1, 3, 224, 224)
    dummy_bgr = np.full((224, 224, 3), 120, dtype=np.uint8)

    out_dir = str(tmp_path / "explanations")
    res = generate_explanation(
        model=dummy_model,
        input_tensor=x,
        original_bgr=dummy_bgr,
        target_class=1,
        output_dir=out_dir,
        prefix="sample_frame"
    )

    assert res["is_success"] is True
    assert "saved_overlay_path" in res
    assert os.path.exists(res["saved_overlay_path"])
    assert os.path.exists(res["saved_heatmap_path"])


def test_face_quality_evaluation():
    """Verify quality scoring on sharp vs severely blurred crops."""
    # Sharp image with high variance
    sharp_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    q_sharp = evaluate_face_quality(sharp_crop)
    assert q_sharp["sharpness_score"] > 50.0

    # Severely blurred crop
    blurred_crop = cv2.GaussianBlur(sharp_crop, (25, 25), 10.0)
    q_blur = evaluate_face_quality(blurred_crop, min_blur_variance=40.0)
    assert any("Low sharpness" in w for w in q_blur["warnings"])
    assert q_blur["quality_pass"] is False


def test_evidence_record_compilation(tmp_path):
    """Verify evidence record structure, categorization, and JSON export."""
    # Create temporary image file
    test_img = str(tmp_path / "test_media.png")
    img = np.full((160, 160, 3), 80, dtype=np.uint8)
    cv2.imwrite(test_img, img)

    mock_analysis = {
        "predicted_class": "Real",
        "manipulation_score": 0.22,
        "confidence": 0.78,
        "raw_scores": [1.45, -1.22],
        "inference_time_ms": 12.5,
        "quality_warnings": ["Minor blur detected."]
    }

    record = build_evidence_record(
        media_path=test_img,
        analysis_result=mock_analysis,
        model_name="efficientnet_b0",
        heatmap_paths=["artifacts/reports/sample_cam.png"]
    )

    # Check the 3 required forensic categories
    assert "model_findings" in record
    assert "file_observations" in record
    assert "forensic_limitations" in record
    assert record["file_observations"]["sha256"] != ""
    assert record["model_findings"]["manipulation_score"] == 0.22

    # Test JSON export
    json_path = str(tmp_path / "evidence.json")
    saved_path = export_evidence_json(record, json_path)
    assert os.path.exists(saved_path)

    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["record_id"] == record["record_id"]
    assert "disclaimer" in loaded["forensic_limitations"]
