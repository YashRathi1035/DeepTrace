"""Forensic input quality assessment, sharpness diagnostics, and integrity verification."""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


def evaluate_face_quality(
    face_crop_bgr: np.ndarray,
    min_resolution: int = 50,
    min_blur_variance: float = 35.0,
    min_brightness: float = 35.0,
    max_brightness: float = 225.0
) -> Dict[str, Any]:
    """Perform multi-factor forensic quality assessment on a facial crop.
    
    Poor image quality (excessive blur, severe under/over-exposure, low resolution)
    is the leading cause of false positives and detector uncertainty in media forensics.
    
    Args:
        face_crop_bgr: Cropped facial region (OpenCV BGR).
        min_resolution: Minimum acceptable pixel width or height.
        min_blur_variance: Minimum Laplacian variance below which image is flagged as blurred.
        min_brightness: Minimum mean grayscale luminance (under-exposure threshold).
        max_brightness: Maximum mean grayscale luminance (over-exposure threshold).
        
    Returns:
        Structured dictionary of quality scores, overall pass/fail boolean, and diagnostic warnings.
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return {
            "quality_score": 0.0,
            "quality_pass": False,
            "sharpness_score": 0.0,
            "brightness": 0.0,
            "resolution": [0, 0],
            "warnings": ["Empty or invalid face crop provided."]
        }

    h, w = face_crop_bgr.shape[:2]
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness via Laplacian variance
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_normalized = float(np.clip(blur_var / 300.0, 0.0, 1.0))

    # 2. Exposure & Illumination
    mean_lum = float(np.mean(gray))
    lum_score = 1.0 - abs(mean_lum - 128.0) / 128.0  # 1.0 is balanced mid-gray

    # 3. Resolution factor
    res_factor = float(np.clip(min(w, h) / 224.0, 0.0, 1.0))

    # Composite quality score [0.0, 1.0]
    composite_score = round(0.45 * sharpness_normalized + 0.30 * lum_score + 0.25 * res_factor, 4)

    warnings: List[str] = []
    if blur_var < min_blur_variance:
        warnings.append(f"Low sharpness (Laplacian variance {blur_var:.1f} < {min_blur_variance}); facial textures may be degraded.")
    if mean_lum < min_brightness:
        warnings.append(f"Under-exposed face crop (mean luminance {mean_lum:.1f} < {min_brightness}); shadow occlusion present.")
    elif mean_lum > max_brightness:
        warnings.append(f"Over-exposed face crop (mean luminance {mean_lum:.1f} > {max_brightness}); highlight blowout detected.")
    if min(w, h) < min_resolution:
        warnings.append(f"Low facial crop resolution ({w}x{h} px < {min_resolution} px); forensic artifacts may be indistinguishable.")

    quality_pass = len(warnings) == 0 and (composite_score >= 0.35)

    return {
        "quality_score": composite_score,
        "quality_pass": quality_pass,
        "sharpness_score": round(blur_var, 2),
        "brightness": round(mean_lum, 2),
        "resolution": [w, h],
        "warnings": warnings
    }
