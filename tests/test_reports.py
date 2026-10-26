"""Automated unit tests for forensic evidence PDF and JSON report generation."""

import os
import json
import tempfile
import numpy as np
import cv2
import pytest

from forensic.report import (
    build_evidence_record,
    export_evidence_json,
    generate_forensic_pdf,
    STANDARD_DISCLAIMER
)


@pytest.fixture
def sample_evidence(tmp_path):
    """Create a sample image, visual overlay, and structured evidence record."""
    # Dummy media file
    media_path = str(tmp_path / "forensic_target.png")
    img = np.full((180, 180, 3), 100, dtype=np.uint8)
    cv2.imwrite(media_path, img)

    # Dummy visual overlay
    cam_path = str(tmp_path / "gradcam_overlay.png")
    cv2.imwrite(cam_path, img)

    mock_analysis = {
        "predicted_class": "Potentially Manipulated",
        "manipulation_score": 0.8842,
        "confidence": 0.8842,
        "raw_scores": [-1.8, 2.1],
        "inference_time_ms": 24.5,
        "quality_warnings": ["Minor sensor noise detected."]
    }

    record = build_evidence_record(
        media_path=media_path,
        analysis_result=mock_analysis,
        model_name="efficientnet_b0",
        heatmap_paths=[cam_path]
    )
    return record, cam_path


def test_json_evidence_export(sample_evidence, tmp_path):
    """Verify evidence JSON export, integrity, and disclaimer preservation."""
    record, _ = sample_evidence
    json_path = str(tmp_path / "evidence_output.json")

    saved_path = export_evidence_json(record, json_path)
    assert os.path.exists(saved_path)

    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["record_id"] == record["record_id"]
    assert data["model_findings"]["manipulation_score"] == 0.8842
    assert "disclaimer" in data["forensic_limitations"]


def test_pdf_report_generation(sample_evidence, tmp_path):
    """Verify ReportLab PDF generation, page composition, and image embedding."""
    record, cam_path = sample_evidence
    pdf_path = str(tmp_path / "Forensic_Report.pdf")

    out_path = generate_forensic_pdf(
        evidence_record=record,
        output_pdf_path=pdf_path,
        visual_paths=[cam_path]
    )

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 3000  # Multi-page/table PDF > 3KB

    # Validate PDF magic header
    with open(out_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"
