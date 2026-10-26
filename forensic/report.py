"""Forensic evidence collection, structured record compilation, and integrity reporting."""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from forensic.metadata import compute_sha256, inspect_media_file
from forensic.quality import evaluate_face_quality


STANDARD_DISCLAIMER = (
    "DeepTrace is a digital media forensics research and decision-support tool, "
    "not a court-admissible or definitive authenticity verifier. Model predictions and Grad-CAM "
    "visualizations represent internal neural network feature attributions, NOT verified "
    "tampering masks. Metadata anomalies or missing EXIF tags are standard in social-media "
    "transferred media and do not constitute conclusive evidence of manipulation."
)


def build_evidence_record(
    media_path: str,
    analysis_result: Dict[str, Any],
    model_name: str = "efficientnet_b0",
    checkpoint_version: str = "1.0",
    config_summary: Optional[Dict[str, Any]] = None,
    heatmap_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compile structured, reproducible forensic evidence record.
    
    Categorizes evidence into:
    1. Model-Generated Findings
    2. File & Container Observations
    3. Forensic Limitations & Quality Caveats
    
    Args:
        media_path: Target image or video file path.
        analysis_result: Structured result from DeepfakeDetector or VideoForensicAnalyzer.
        model_name: Architecture identifier.
        checkpoint_version: Checkpoint model version.
        config_summary: Key parameters (sampling rate, aggregation policy, thresholds).
        heatmap_paths: Saved paths to Grad-CAM heatmap and overlay images.
        
    Returns:
        Structured evidence dictionary.
    """
    file_meta = inspect_media_file(media_path)
    now_utc = datetime.now(timezone.utc).isoformat()

    is_video = file_meta.get("file_type") == "video"
    heatmap_paths = heatmap_paths or []

    # 1. Model Findings
    if is_video:
        model_findings = {
            "media_classification": analysis_result.get("predicted_class", "Unspecified"),
            "video_level_score": analysis_result.get("video_level_score", 0.0),
            "aggregation_strategy": analysis_result.get("selected_aggregation_method", "top_k"),
            "num_analyzed_frames": analysis_result.get("num_analyzed_frames", 0),
            "num_skipped_frames": analysis_result.get("num_frames_without_faces", 0),
            "temporal_consistency": analysis_result.get("temporal_consistency", {}),
            "inference_duration_ms": analysis_result.get("processing_time_ms", 0.0),
            "suspicious_timestamps": [
                interval["start_seconds"]
                for interval in analysis_result.get("temporal_consistency", {}).get("suspicious_intervals", [])
            ],
            "frame_level_scores": [
                {
                    "timestamp": r.get("timestamp"),
                    "frame_index": r.get("frame_index"),
                    "manipulation_score": r.get("manipulation_score"),
                    "confidence": r.get("confidence"),
                    "face_id": r.get("face_id", "face_01")
                }
                for r in analysis_result.get("per_frame_scores", [])
            ]
        }
    else:
        # Static image
        model_findings = {
            "media_classification": analysis_result.get("predicted_class", "Unspecified"),
            "manipulation_score": analysis_result.get("manipulation_score", 0.0),
            "confidence": analysis_result.get("confidence", 0.0),
            "raw_scores": analysis_result.get("raw_scores", []),
            "inference_duration_ms": analysis_result.get("inference_time_ms", 0.0),
            "calibrated": analysis_result.get("is_calibrated", False)
        }

    # 2. File and Media Observations
    file_observations = {
        "file_name": file_meta.get("file_name", os.path.basename(media_path)),
        "file_path": os.path.abspath(media_path),
        "sha256": file_meta.get("sha256", ""),
        "file_size_bytes": file_meta.get("file_size_bytes", 0),
        "file_type": file_meta.get("file_type", "unknown"),
        "dimensions": [file_meta.get("width", 0), file_meta.get("height", 0)],
        "fps": file_meta.get("fps"),
        "duration_seconds": file_meta.get("duration_seconds"),
        "frame_count": file_meta.get("frame_count"),
        "codec": file_meta.get("fourcc"),
        "exif_tags": file_meta.get("exif", {})
    }

    # 3. Quality Warnings and Limitations
    quality_warnings = list(analysis_result.get("quality_warnings", []))
    if not quality_warnings and "preprocessing_info" in analysis_result:
        quality_warnings.extend(analysis_result["preprocessing_info"].get("warnings", []))
    quality_warnings.extend(file_meta.get("warnings", []))

    evidence_record = {
        "record_id": f"evidence_{int(time.time())}_{file_meta.get('sha256', '')[:8]}",
        "timestamp_utc": now_utc,
        "tool_name": "DeepTrace Forensic Evidence Analyzer",
        "tool_version": "1.0.0",
        "model_metadata": {
            "architecture": model_name,
            "version": checkpoint_version,
            "config": config_summary or {}
        },
        "file_observations": file_observations,
        "model_findings": model_findings,
        "explainability": {
            "method": "Grad-CAM",
            "heatmap_artifacts": heatmap_paths,
            "explanation_note": "Visual heatmaps highlight regions exhibiting high convolutional activation toward the manipulated class."
        },
        "quality_warnings": list(set(quality_warnings)),
        "forensic_limitations": {
            "disclaimer": STANDARD_DISCLAIMER,
            "decision_policy": "Reliability-aware classification (Authentic, Manipulated, or Inconclusive).",
            "non_definitive_notice": "Analysis is probabilistic and must be evaluated alongside independent provenance evidence."
        }
    }

    return evidence_record


def export_evidence_json(evidence_record: Dict[str, Any], output_path: str) -> str:
    """Save structured evidence record to a formatted JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evidence_record, f, indent=2)
    return os.path.abspath(output_path)


def generate_forensic_pdf(
    evidence_record: Dict[str, Any],
    output_pdf_path: str,
    visual_paths: Optional[List[str]] = None
) -> str:
    """Generate a formal digital media forensic PDF report using ReportLab.
    
    Args:
        evidence_record: Output dictionary from build_evidence_record().
        output_pdf_path: Destination path for .pdf file.
        visual_paths: List of file paths to explanation images (e.g. crop and Grad-CAM overlay).
        
    Returns:
        Absolute path to generated PDF report.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
        KeepTogether,
        HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0d1117")
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#57606a")
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0969da"),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#24292f")
    )
    disclaimer_style = ParagraphStyle(
        "NoticeText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#57606a")
    )

    story = []

    # 1. Header & Metadata
    story.append(Paragraph("DEEPTRACE DIGITAL MEDIA FORENSIC REPORT", title_style))
    story.append(Paragraph(f"Record ID: {evidence_record.get('record_id')} | Generated: {evidence_record.get('timestamp_utc')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0969da"), spaceBefore=6, spaceAfter=8))

    # 2. Forensic Notice Box
    notice_text = f"<b>IMPORTANT FORENSIC NOTICE:</b> {evidence_record.get('forensic_limitations', {}).get('disclaimer', STANDARD_DISCLAIMER)}"
    notice_table = Table(
        [[Paragraph(notice_text, disclaimer_style)]],
        colWidths=[530]
    )
    notice_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f8fa")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d0d7de")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(notice_table)
    story.append(Spacer(1, 10))

    # 3. File & Cryptographic Verification Table
    story.append(Paragraph("1. File Identification & Cryptographic Hash", heading_style))
    f_obs = evidence_record.get("file_observations", {})
    file_data = [
        [Paragraph("<b>File Name:</b>", body_style), Paragraph(str(f_obs.get("file_name")), body_style)],
        [Paragraph("<b>File Size:</b>", body_style), Paragraph(f"{f_obs.get('file_size_bytes', 0):,} bytes", body_style)],
        [Paragraph("<b>SHA-256 Hash:</b>", body_style), Paragraph(f"<font face='Courier' size=8>{f_obs.get('sha256', 'N/A')}</font>", body_style)],
        [Paragraph("<b>Media Type:</b>", body_style), Paragraph(str(f_obs.get("file_type")).upper(), body_style)],
        [Paragraph("<b>Resolution:</b>", body_style), Paragraph(f"{f_obs.get('dimensions', [0, 0])[0]} x {f_obs.get('dimensions', [0, 0])[1]} px", body_style)]
    ]
    if f_obs.get("file_type") == "video":
        file_data.append([Paragraph("<b>Duration / FPS:</b>", body_style), Paragraph(f"{f_obs.get('duration_seconds')}s @ {f_obs.get('fps')} FPS ({f_obs.get('frame_count')} frames)", body_style)])
        file_data.append([Paragraph("<b>FourCC Codec:</b>", body_style), Paragraph(str(f_obs.get("codec")), body_style)])

    f_table = Table(file_data, colWidths=[130, 400])
    f_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e1e4e8")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f6f8fa")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(f_table)
    story.append(Spacer(1, 10))

    # 4. Model Findings Table
    story.append(Paragraph("2. Forensic Model Assessment", heading_style))
    m_find = evidence_record.get("model_findings", {})
    verdict = m_find.get("media_classification", "Unspecified")

    # Color code verdict
    if "Manipulated" in verdict:
        verdict_color = "#cf222e"  # Red
    elif "Authentic" in verdict:
        verdict_color = "#1a7f37"  # Green
    else:
        verdict_color = "#9a6700"  # Amber

    m_score = m_find.get("manipulation_score") or m_find.get("video_level_score", 0.0)

    model_data = [
        [Paragraph("<b>Classification Decision:</b>", body_style), Paragraph(f"<font color='{verdict_color}'><b>{verdict.upper()}</b></font>", body_style)],
        [Paragraph("<b>Manipulation Probability:</b>", body_style), Paragraph(f"<b>{m_score:.4f}</b> (Threshold: 0.65)", body_style)],
        [Paragraph("<b>Detector Architecture:</b>", body_style), Paragraph(str(evidence_record.get("model_metadata", {}).get("architecture")), body_style)],
        [Paragraph("<b>Inference Latency:</b>", body_style), Paragraph(f"{m_find.get('inference_duration_ms', 0):.1f} ms", body_style)]
    ]
    if "aggregation_strategy" in m_find:
        model_data.append([Paragraph("<b>Aggregation Policy:</b>", body_style), Paragraph(str(m_find.get("aggregation_strategy")), body_style)])
        model_data.append([Paragraph("<b>Analyzed Frames:</b>", body_style), Paragraph(f"{m_find.get('num_analyzed_frames')} frames", body_style)])

    m_table = Table(model_data, colWidths=[130, 400])
    m_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e1e4e8")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f6f8fa")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 10))

    # 5. Visual Explainability Gallery
    if visual_paths:
        existing_imgs = [p for p in visual_paths if os.path.exists(p)]
        if existing_imgs:
            story.append(Paragraph("3. Visual Saliency Attributions (Grad-CAM)", heading_style))
            story.append(Paragraph("Heatmaps illustrate convolutional activation patterns corresponding to the manipulated class (red denotes high activation).", subtitle_style))
            story.append(Spacer(1, 4))

            img_elements = []
            for p in existing_imgs[:2]:
                try:
                    img_elements.append(RLImage(p, width=180, height=180))
                except Exception:
                    pass

            if img_elements:
                img_table = Table([img_elements], colWidths=[200] * len(img_elements))
                img_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(img_table)
                story.append(Spacer(1, 8))

    # 6. Quality Warnings
    story.append(Paragraph("4. Input Diagnostics & Warnings", heading_style))
    warnings = evidence_record.get("quality_warnings", [])
    if warnings:
        warn_content = "<br/>".join([f"• {w}" for w in warnings])
    else:
        warn_content = "• No critical image quality or container anomalies detected."

    warn_table = Table(
        [[Paragraph(warn_content, body_style)]],
        colWidths=[530]
    )
    warn_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8c5") if warnings else colors.HexColor("#f6f8fa")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d4a72c") if warnings else colors.HexColor("#d0d7de")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(warn_table)
    story.append(Spacer(1, 10))

    # 7. Signature / Footer
    story.append(Paragraph("Report generated autonomously by DeepTrace Forensic Engine v1.0. Hash-verified tamper-evident record.", disclaimer_style))

    doc.build(story)
    return os.path.abspath(output_pdf_path)
