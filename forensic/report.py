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
