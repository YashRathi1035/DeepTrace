"""Image preprocessing pipeline, tensor normalization, quality checks, and face crop preparation."""

import os
from typing import Dict, Any, Tuple, Optional, List, Union
import cv2
import numpy as np
import torch
from torchvision import transforms

from preprocessing.face_detection import (
    BaseFaceDetector,
    FaceDetection,
    crop_face_with_margin,
    get_face_detector
)


class ImageProcessor:
    """Handles end-to-end image ingestion, face extraction, quality evaluation, and model tensor preparation."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        detector_name: str = "mediapipe",
        face_margin: float = 0.20,
        min_face_size: int = 40,
        detector_model_path: Optional[str] = None
    ):
        self.image_size = tuple(image_size)
        self.mean = list(mean)
        self.std = list(std)
        self.face_margin = face_margin
        self.min_face_size = min_face_size

        # PyTorch normalization transform
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),  # Scales [0, 255] to [0.0, 1.0] and permutes to (C, H, W)
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Detector
        self.detector: BaseFaceDetector = get_face_detector(detector_name, model_path=detector_model_path)

    @staticmethod
    def load_image(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Load image as BGR numpy array from filepath, raw bytes, or existing array."""
        if isinstance(image_input, np.ndarray):
            return image_input

        if isinstance(image_input, (bytes, bytearray)):
            np_arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from provided byte stream.")
            return img

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"OpenCV failed to decode image at: {image_input}")
            return img

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def assess_quality(crop_bgr: np.ndarray) -> Dict[str, Any]:
        """Assess face crop quality metrics: sharpness, brightness, and resolution.
        
        Args:
            crop_bgr: BGR face crop.
            
        Returns:
            Dictionary with blur score (Laplacian variance), brightness, warnings.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {
                "blur_score": 0.0,
                "brightness": 0.0,
                "quality_pass": False,
                "warnings": ["Empty or invalid crop."]
            }

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        h, w = crop_bgr.shape[:2]

        warnings = []
        if blur_score < 40.0:
            warnings.append(f"Low face sharpness (variance {blur_score:.1f} < 40); image appears blurred.")
        if brightness < 35.0:
            warnings.append(f"Under-exposed face crop (mean luminance {brightness:.1f} < 35).")
        elif brightness > 220.0:
            warnings.append(f"Over-exposed face crop (mean luminance {brightness:.1f} > 220).")
        if min(h, w) < 40:
            warnings.append(f"Low facial resolution ({w}x{h} px); artifacts may be distorted.")

        return {
            "blur_score": round(float(blur_score), 2),
            "brightness": round(brightness, 2),
            "dimensions": [w, h],
            "quality_pass": len(warnings) == 0,
            "warnings": warnings
        }

    def process_face(
        self,
        image_input: Union[str, bytes, np.ndarray],
        target_face_index: int = 0
    ) -> Dict[str, Any]:
        """Detect face, crop with margin, assess quality, and prepare model tensor.
        
        Args:
            image_input: Path, bytes, or BGR ndarray.
            target_face_index: Index of detected face to extract (default 0 = highest confidence).
            
        Returns:
            Structured dictionary containing:
            - 'tensor': PyTorch FloatTensor of shape (1, 3, H, W)
            - 'crop_bgr': BGR cropped image at target_size
            - 'crop_rgb': RGB cropped image at target_size
            - 'original_bbox': bounding box of face
            - 'detection_confidence': confidence of face detection
            - 'quality': quality metrics and warnings
            - 'num_faces_detected': total faces in frame
        """
        image_bgr = self.load_image(image_input)
        detections = self.detector.detect_faces(image_bgr, min_confidence=0.4)

        if not detections:
            # Fallback to whole frame centered crop
            h, w = image_bgr.shape[:2]
            size = min(h, w)
            x = (w - size) // 2
            y = (h - size) // 2
            best_det = FaceDetection(bbox=(x, y, size, size), confidence=0.0)
            warning = "No face detected by facial detector; used centered frame fallback."
        else:
            # Sort by confidence descending
            detections.sort(key=lambda d: d.confidence, reverse=True)
            best_det = detections[min(target_face_index, len(detections) - 1)]
            warning = None

        crop_bgr = crop_face_with_margin(
            image_bgr,
            best_det.bbox,
            margin=self.face_margin,
            target_size=self.image_size
        )

        quality = self.assess_quality(crop_bgr)
        if warning:
            quality["warnings"].append(warning)

        # Convert to RGB for model input
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tensor_transform(crop_rgb).unsqueeze(0)  # Shape (1, 3, H, W)

        return {
            "tensor": tensor,
            "crop_bgr": crop_bgr,
            "crop_rgb": crop_rgb,
            "original_bbox": best_det.bbox,
            "detection_confidence": best_det.confidence,
            "quality": quality,
            "num_faces_detected": len(detections)
        }

    def denormalize_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalize a normalized image tensor (C, H, W) back to RGB numpy array [0, 255].
        
        Args:
            tensor: PyTorch tensor with shape (3, H, W) or (1, 3, H, W).
            
        Returns:
            RGB numpy uint8 array of shape (H, W, 3).
        """
        t = tensor.detach().cpu().clone()
        if t.ndim == 4:
            t = t.squeeze(0)

        # Invert normalization: x * std + mean
        mean = torch.tensor(self.mean, device=t.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=t.device).view(3, 1, 1)
        t = t * std + mean
        t = torch.clamp(t, 0.0, 1.0)
        np_img = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return np_img
