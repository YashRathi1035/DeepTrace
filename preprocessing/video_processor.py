"""Video processing pipeline: frame sampling, face extraction, and temporal stream handling."""

import os
from typing import Dict, Any, List, Optional, Generator, Tuple
import cv2
import numpy as np

from preprocessing.image_processor import ImageProcessor
from forensic.metadata import inspect_video_metadata


class VideoProcessor:
    """Manages video decoding, frame sampling at target rates, face extraction, and memory-safe streaming."""

    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        sampling_fps: float = 1.0,
        max_frames: int = 60,
        max_duration_seconds: float = 300.0,
        max_file_size_mb: float = 150.0
    ):
        self.image_processor = image_processor or ImageProcessor()
        self.sampling_fps = max(0.1, sampling_fps)
        self.max_frames = max_frames
        self.max_duration_seconds = max_duration_seconds
        self.max_file_size_mb = max_file_size_mb

    def validate_video_file(self, video_path: str) -> Dict[str, Any]:
        """Validate file size, readable container, and duration constraints before processing.
        
        Args:
            video_path: Absolute or relative path to video.
            
        Returns:
            Dictionary with 'is_valid' boolean, 'metadata', and any 'errors'.
        """
        if not os.path.isfile(video_path):
            return {
                "is_valid": False,
                "errors": [f"File does not exist: {video_path}"],
                "metadata": {}
            }

        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return {
                "is_valid": False,
                "errors": [f"File size ({size_mb:.1f} MB) exceeds maximum allowed ({self.max_file_size_mb} MB)."],
                "metadata": {}
            }

        meta = inspect_video_metadata(video_path)
        if not meta.get("is_valid", False):
            return {
                "is_valid": False,
                "errors": meta.get("warnings", ["Unable to parse video stream."]),
                "metadata": meta
            }

        duration = meta.get("duration_seconds", 0.0)
        if duration > self.max_duration_seconds:
            return {
                "is_valid": False,
                "errors": [f"Video duration ({duration:.1f}s) exceeds limit ({self.max_duration_seconds}s)."],
                "metadata": meta
            }

        return {
            "is_valid": True,
            "errors": [],
            "metadata": meta
        }

    def stream_sampled_frames(
        self,
        video_path: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator yielding sampled frames one-by-one to avoid high memory consumption.
        
        Yields:
            Dict containing:
            - 'frame_index': raw video frame index
            - 'timestamp': time in seconds from video start
            - 'frame_bgr': OpenCV BGR frame image
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for streaming: {video_path}")

        try:
            native_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if native_fps <= 0:
                native_fps = 25.0

            # Calculate frame step interval
            # E.g. native 30 fps, sampling_fps 1.0 -> step = 30 frames
            frame_step = max(1, int(round(native_fps / self.sampling_fps)))

            current_frame = 0
            sampled_count = 0

            while cap.isOpened() and sampled_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame % frame_step == 0:
                    timestamp = round(current_frame / native_fps, 2)
                    yield {
                        "frame_index": current_frame,
                        "timestamp": timestamp,
                        "frame_bgr": frame
                    }
                    sampled_count += 1

                current_frame += 1

        finally:
            cap.release()

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract and preprocess face crops from all sampled frames in a video.
        
        Args:
            video_path: Path to target video file.
            output_dir: Optional directory to save extracted face crops.
            
        Returns:
            Summary dictionary with:
            - 'frames': List of per-frame processed face dictionaries
            - 'num_sampled_frames': Total frames evaluated
            - 'num_usable_faces': Total frames where a face was detected
            - 'video_metadata': Basic stream metadata
            - 'warnings': Aggregate processing warnings
        """
        val = self.validate_video_file(video_path)
        if not val["is_valid"]:
            raise ValueError(f"Video validation failed: {'; '.join(val['errors'])}")

        frames_data: List[Dict[str, Any]] = []
        usable_face_count = 0
        total_sampled = 0
        warnings: List[str] = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for frame_item in self.stream_sampled_frames(video_path):
            total_sampled += 1
            idx = frame_item["frame_index"]
            ts = frame_item["timestamp"]
            frame_bgr = frame_item["frame_bgr"]

            try:
                face_result = self.image_processor.process_face(frame_bgr)
                has_detected_face = (face_result["num_faces_detected"] > 0)

                if has_detected_face:
                    usable_face_count += 1
                else:
                    face_result["quality"]["warnings"].append("No facial landmarks detected in frame.")

                record = {
                    "frame_index": idx,
                    "timestamp": ts,
                    "has_face": has_detected_face,
                    "detection_confidence": face_result["detection_confidence"],
                    "quality": face_result["quality"],
                    "crop_bgr": face_result["crop_bgr"],
                    "crop_rgb": face_result["crop_rgb"],
                    "tensor": face_result["tensor"],
                    "original_bbox": face_result["original_bbox"]
                }

                if output_dir:
                    save_path = os.path.join(output_dir, f"frame_{idx:05d}_ts_{ts:.1f}s.jpg")
                    cv2.imwrite(save_path, face_result["crop_bgr"])
                    record["saved_crop_path"] = save_path

                frames_data.append(record)

            except Exception as e:
                warnings.append(f"Error processing frame {idx} at timestamp {ts}s: {str(e)}")

        if usable_face_count == 0:
            warnings.append("No recognizable faces found in any sampled frame of the video.")
        elif usable_face_count < (total_sampled * 0.3):
            warnings.append(
                f"Low facial visibility: only {usable_face_count}/{total_sampled} frames contain usable faces."
            )

        return {
            "video_path": video_path,
            "num_sampled_frames": total_sampled,
            "num_usable_faces": usable_face_count,
            "frames": frames_data,
            "video_metadata": val["metadata"],
            "warnings": warnings
        }
