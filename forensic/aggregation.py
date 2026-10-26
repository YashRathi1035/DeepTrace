"""Video forensic aggregation, multi-face tracking, and temporal consistency analysis."""

import os
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from models.predict import DeepfakeDetector
from preprocessing.video_processor import VideoProcessor
from preprocessing.face_detection import FaceDetection


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def aggregate_scores(
    scores: List[float],
    method: str = "top_k",
    top_k_ratio: float = 0.20
) -> float:
    """Aggregate per-frame manipulation probabilities into a unified video-level score.
    
    Args:
        scores: List of float manipulation probabilities [0.0, 1.0].
        method: Aggregation strategy:
            - 'mean': Standard arithmetic average.
            - 'median': Median score (robust against occasional single-frame noise).
            - 'top_k': Average of the top K highest scores (critical for localized deepfakes
                       where only brief sequences of frames are manipulated).
            - 'trimmed_mean': Average after trimming 10% extreme outliers.
        top_k_ratio: Fraction of frames to consider for 'top_k' (e.g. 0.20 = top 20%).
        
    Returns:
        Aggregated video-level manipulation score [0.0, 1.0].
    """
    if not scores:
        return 0.0

    scores_arr = np.array(scores, dtype=float)

    if method == "mean":
        return float(np.mean(scores_arr))

    elif method == "median":
        return float(np.median(scores_arr))

    elif method == "top_k":
        # Sort descending
        sorted_scores = np.sort(scores_arr)[::-1]
        k = max(1, int(np.ceil(len(sorted_scores) * top_k_ratio)))
        top_scores = sorted_scores[:k]
        return float(np.mean(top_scores))

    elif method == "trimmed_mean":
        if len(scores_arr) < 5:
            return float(np.mean(scores_arr))
        # Trim lowest and highest 10%
        trim_n = max(1, int(len(scores_arr) * 0.10))
        trimmed = np.sort(scores_arr)[trim_n:-trim_n]
        return float(np.mean(trimmed))

    else:
        raise ValueError(f"Unknown aggregation method: {method}. Choose 'mean', 'median', 'top_k', or 'trimmed_mean'.")


def analyze_temporal_consistency(
    scores: List[float],
    timestamps: List[float],
    suspicious_threshold: float = 0.65
) -> Dict[str, Any]:
    """Analyze prediction dynamics across time to identify manipulation spikes and volatility.
    
    Note: Temporal score variation alone does NOT prove manipulation, as it may also
    arise from changes in camera perspective, motion blur, facial pose, or lighting.
    
    Args:
        scores: Chronological frame manipulation probabilities.
        timestamps: Corresponding frame timestamps in seconds.
        suspicious_threshold: Probability threshold above which a frame is considered suspicious.
        
    Returns:
        Structured dictionary of temporal consistency metrics and detected anomalous intervals.
    """
    if not scores:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "std_dev": 0.0,
            "max_consecutive_delta": 0.0,
            "suspicious_intervals": [],
            "volatility_warning": False,
            "notes": "No frames available for temporal analysis."
        }

    scores_arr = np.array(scores, dtype=float)
    mean_val = float(np.mean(scores_arr))
    var_val = float(np.var(scores_arr))
    std_val = float(np.std(scores_arr))

    # Calculate frame-to-frame delta
    if len(scores_arr) > 1:
        deltas = np.abs(np.diff(scores_arr))
        max_delta = float(np.max(deltas))
    else:
        max_delta = 0.0

    # Locate suspicious intervals
    suspicious_intervals: List[Dict[str, Any]] = []
    in_interval = False
    start_ts = 0.0
    peak_score = 0.0

    for i, (score, ts) in enumerate(zip(scores, timestamps)):
        if score >= suspicious_threshold:
            if not in_interval:
                in_interval = True
                start_ts = ts
                peak_score = score
            else:
                peak_score = max(peak_score, score)
        else:
            if in_interval:
                suspicious_intervals.append({
                    "start_seconds": round(start_ts, 2),
                    "end_seconds": round(timestamps[i - 1], 2),
                    "peak_manipulation_score": round(peak_score, 4)
                })
                in_interval = False

    # Close any open interval at video end
    if in_interval and timestamps:
        suspicious_intervals.append({
            "start_seconds": round(start_ts, 2),
            "end_seconds": round(timestamps[-1], 2),
            "peak_manipulation_score": round(peak_score, 4)
        })

    # Flag high temporal volatility
    volatility_warning = (std_val > 0.25) or (max_delta > 0.45)

    return {
        "mean": round(mean_val, 4),
        "variance": round(var_val, 4),
        "std_dev": round(std_val, 4),
        "max_consecutive_delta": round(max_delta, 4),
        "suspicious_intervals": suspicious_intervals,
        "volatility_warning": volatility_warning,
        "notes": (
            "High frame-to-frame score volatility detected. Note that variations can arise from "
            "temporal manipulation, facial rotation, lighting shifts, or detector uncertainty."
            if volatility_warning else "Score trajectory displays consistent temporal stability."
        )
    }


class VideoForensicAnalyzer:
    """Unified video forensic engine executing frame sampling, model inference, and temporal evidence aggregation."""

    def __init__(
        self,
        detector: DeepfakeDetector,
        video_processor: Optional[VideoProcessor] = None,
        default_aggregation: str = "top_k",
        top_k_ratio: float = 0.20,
        fake_threshold: float = 0.65
    ):
        self.detector = detector
        self.video_processor = video_processor or VideoProcessor(
            image_processor=detector.image_processor
        )
        self.default_aggregation = default_aggregation
        self.top_k_ratio = top_k_ratio
        self.fake_threshold = fake_threshold

    def analyze_video(
        self,
        video_path: str,
        aggregation_method: Optional[str] = None,
        save_crops_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform end-to-end video forensic analysis.
        
        Args:
            video_path: Path to target video file.
            aggregation_method: Override aggregation strategy ('top_k', 'mean', 'median').
            save_crops_dir: Optional directory to save extracted face crops.
            
        Returns:
            Structured dictionary containing video-level verdict, per-frame scores,
            temporal consistency stats, quality warnings, and execution metrics.
        """
        start_time = time.perf_counter()
        agg_method = aggregation_method or self.default_aggregation

        # 1. Video validation and frame extraction
        extracted = self.video_processor.process_video(video_path, output_dir=save_crops_dir)
        frames_meta = extracted["frames"]
        quality_warnings = list(extracted.get("warnings", []))

        if not frames_meta:
            total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "video_path": video_path,
                "video_level_score": 0.0,
                "predicted_class": "Inconclusive",
                "per_frame_scores": [],
                "timestamps": [],
                "num_analyzed_frames": 0,
                "num_frames_without_faces": extracted.get("num_sampled_frames", 0),
                "selected_aggregation_method": agg_method,
                "temporal_consistency": analyze_temporal_consistency([], []),
                "processing_time_ms": round(total_elapsed_ms, 2),
                "quality_warnings": ["No frames could be extracted from video."],
                "video_metadata": extracted.get("video_metadata", {})
            }

        # 2. Run detector on extracted frame crops
        frame_scores: List[float] = []
        timestamps: List[float] = []
        per_frame_records: List[Dict[str, Any]] = []
        frames_without_faces = 0

        # Simple IoU face track manager across frames
        last_face_bbox = None
        face_id = 1

        for f_info in frames_meta:
            ts = f_info["timestamp"]
            idx = f_info["frame_index"]
            has_face = f_info["has_face"]
            tensor = f_info["tensor"]

            if not has_face:
                frames_without_faces += 1

            # Predict
            pred = self.detector.predict_batch(tensor)[0]
            m_score = pred["manipulation_score"]

            frame_scores.append(m_score)
            timestamps.append(ts)

            # Track association
            current_bbox = f_info["original_bbox"]
            if last_face_bbox is not None:
                iou = compute_iou(current_bbox, last_face_bbox)
                if iou < 0.2:
                    face_id += 1  # Likely a different person or abrupt scene change
            last_face_bbox = current_bbox

            record = {
                "frame_index": idx,
                "timestamp": ts,
                "has_face": has_face,
                "face_id": f"face_{face_id:02d}",
                "predicted_class": pred["predicted_class"],
                "manipulation_score": m_score,
                "confidence": pred["confidence"],
                "raw_scores": pred["raw_scores"],
                "detection_confidence": f_info.get("detection_confidence", 0.0),
                "quality": f_info.get("quality", {}),
                "crop_bgr": f_info.get("crop_bgr"),
                "crop_rgb": f_info.get("crop_rgb"),
                "saved_crop_path": f_info.get("saved_crop_path")
            }
            per_frame_records.append(record)

        # 3. Video-level aggregation
        video_level_score = aggregate_scores(
            scores=frame_scores,
            method=agg_method,
            top_k_ratio=self.top_k_ratio
        )

        # Determine video prediction
        predicted_class = "Manipulated" if video_level_score >= self.fake_threshold else "Real"

        # 4. Temporal consistency evaluation
        temporal_stats = analyze_temporal_consistency(
            scores=frame_scores,
            timestamps=timestamps,
            suspicious_threshold=self.fake_threshold
        )

        if temporal_stats["volatility_warning"]:
            quality_warnings.append(temporal_stats["notes"])

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "video_path": video_path,
            "video_level_score": round(video_level_score, 4),
            "predicted_class": predicted_class,
            "per_frame_scores": per_frame_records,
            "timestamps": timestamps,
            "num_analyzed_frames": len(frame_scores),
            "num_frames_without_faces": frames_without_faces,
            "selected_aggregation_method": agg_method,
            "temporal_consistency": temporal_stats,
            "processing_time_ms": round(total_elapsed_ms, 2),
            "quality_warnings": quality_warnings,
            "video_metadata": extracted.get("video_metadata", {})
        }
