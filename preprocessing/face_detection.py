"""Modular face detection, landmark extraction, and face-crop preprocessing."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class FaceDetection:
    """Represents a detected face with bounding box, confidence, and optional landmarks."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in pixels
    confidence: float
    keypoints: Optional[List[Tuple[float, float]]] = None  # (x, y) normalized coordinates
    crop: Optional[np.ndarray] = None  # BGR face crop with context margin


class BaseFaceDetector(ABC):
    """Abstract base class for face detectors to ensure modularity and swappability."""

    @abstractmethod
    def detect_faces(self, image_bgr: np.ndarray, min_confidence: float = 0.5) -> List[FaceDetection]:
        """Detect faces in a BGR image.
        
        Args:
            image_bgr: Input image in OpenCV BGR format.
            min_confidence: Confidence threshold to retain detections.
            
        Returns:
            List of FaceDetection instances.
        """
        pass


class MediaPipeFaceDetector(BaseFaceDetector):
    """Face detector implementation using Google MediaPipe BlazeFace."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize MediaPipe FaceDetector task.
        
        Args:
            model_path: Path to blaze_face_short_range.tflite. Defaults to artifacts/weights/.
        """
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.vision = vision

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "artifacts", "weights", "blaze_face_short_range.tflite")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe face model not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.3)
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_faces(self, image_bgr: np.ndarray, min_confidence: float = 0.5) -> List[FaceDetection]:
        """Detect faces using MediaPipe BlazeFace."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Ensure memory alignment for LiteRT
        image_rgb = np.ascontiguousarray(image_rgb)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)

        results = self.detector.detect(mp_image)
        detections: List[FaceDetection] = []

        if not results or not results.detections:
            return detections

        for det in results.detections:
            score = float(det.categories[0].score) if det.categories else 0.0
            if score < min_confidence:
                continue

            bb = det.bounding_box
            x = max(0, int(bb.origin_x))
            y = max(0, int(bb.origin_y))
            box_w = min(int(bb.width), w - x)
            box_h = min(int(bb.height), h - y)

            if box_w <= 0 or box_h <= 0:
                continue

            keypoints: List[Tuple[float, float]] = []
            if det.keypoints:
                for kp in det.keypoints:
                    keypoints.append((float(kp.x), float(kp.y)))

            detections.append(FaceDetection(
                bbox=(x, y, box_w, box_h),
                confidence=round(score, 4),
                keypoints=keypoints
            ))

        return detections


class FallbackCenterFaceDetector(BaseFaceDetector):
    """Fallback detector that extracts a centered square region when no deep face detector is available."""

    def detect_faces(self, image_bgr: np.ndarray, min_confidence: float = 0.5) -> List[FaceDetection]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        return [FaceDetection(
            bbox=(x, y, size, size),
            confidence=0.50,
            keypoints=None
        )]


def crop_face_with_margin(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.20,
    target_size: Optional[Tuple[int, int]] = (224, 224)
) -> np.ndarray:
    """Crop face region from image expanding bbox by margin for forensic context.
    
    Forensic deepfake detection benefits from surrounding context (forehead, hair borders,
    jawline blending artifacts) rather than tightly cropped facial landmarks.
    
    Args:
        image_bgr: Source image in BGR format.
        bbox: (x, y, width, height) of detected face.
        margin: Fraction of expansion around bounding box (e.g. 0.20 = 20%).
        target_size: Optional (width, height) to resize crop using cubic interpolation.
        
    Returns:
        Cropped and optionally resized face image (BGR).
    """
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = bbox

    # Expand box
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        # Fallback to unexpanded if bounds were degenerate
        crop = image_bgr[max(0, y):min(img_h, y + h), max(0, x):min(img_w, x + w)]

    if target_size is not None and crop.size > 0:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)

    return crop


def get_face_detector(name: str = "mediapipe", model_path: Optional[str] = None) -> BaseFaceDetector:
    """Factory function to instantiate face detectors with graceful fallback.
    
    Args:
        name: Detector name ('mediapipe' or 'fallback').
        model_path: Optional path to detector model weights.
        
    Returns:
        Instance conforming to BaseFaceDetector.
    """
    if name == "mediapipe":
        try:
            return MediaPipeFaceDetector(model_path=model_path)
        except Exception:
            return FallbackCenterFaceDetector()
    return FallbackCenterFaceDetector()
