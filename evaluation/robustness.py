"""Robustness evaluation: image perturbations (JPEG compression, blur, resizing, noise) and degradation curves."""

import os
import io
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.predict import DeepfakeDetector
from forensic.calibration import TemperatureScaler, compute_ece, ReliabilityDecisionPolicy
from evaluation.metrics import compute_metrics


# --- Controlled Image Perturbations ---

def apply_jpeg_compression(image_bgr: np.ndarray, quality: int = 50) -> np.ndarray:
    """Simulate lossy JPEG compression artifacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
    _, enc = cv2.imencode(".jpg", image_bgr, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec


def apply_gaussian_blur(image_bgr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Simulate optical defocus or motion blur."""
    # Ensure odd kernel size
    k_size = int(2 * np.ceil(2 * sigma) + 1)
    k_size = max(3, k_size | 1)
    return cv2.GaussianBlur(image_bgr, (k_size, k_size), sigma)


def apply_downsampling(image_bgr: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Simulate low-resolution capture followed by social media upscaling."""
    h, w = image_bgr.shape[:2]
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    down = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return up


def apply_gaussian_noise(image_bgr: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Simulate sensor ISO noise."""
    noise = np.random.normal(0, sigma, image_bgr.shape).astype(np.float32)
    noisy = np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


class RobustnessBenchmark:
    """Executes systematic forensic perturbation stress-tests across held-out media."""

    def __init__(
        self,
        detector: DeepfakeDetector,
        scaler: Optional[TemperatureScaler] = None,
        policy: Optional[ReliabilityDecisionPolicy] = None
    ):
        self.detector = detector
        self.scaler = scaler or TemperatureScaler(initial_temperature=1.0)
        self.policy = policy or ReliabilityDecisionPolicy()

    def run_perturbation_experiment(
        self,
        test_samples: List[Dict[str, Any]],
        perturbation_fn: Callable[[np.ndarray], np.ndarray],
        severity_label: str
    ) -> Dict[str, Any]:
        """Evaluate detector performance on test set under a specific perturbation condition.
        
        Args:
            test_samples: List of dicts with 'image_path' and 'label'.
            perturbation_fn: Transform function taking BGR array and returning degraded BGR.
            severity_label: Human-readable label (e.g. 'JPEG Q=50', 'Blur sigma=2').
            
        Returns:
            Dictionary with accuracy, ROC-AUC, ECE, coverage, and selective risk under degradation.
        """
        y_true = []
        raw_probs = []
        calibrated_probs = []

        for s in test_samples:
            img_path = s["image_path"]
            label = int(s["label"])
            if not os.path.exists(img_path):
                continue

            orig_bgr = cv2.imread(img_path)
            if orig_bgr is None:
                continue

            # Apply perturbation
            perturbed_bgr = perturbation_fn(orig_bgr)

            # Predict
            pred = self.detector.predict_frame(perturbed_bgr)
            raw_p = pred["manipulation_score"]
            logits = np.array([pred["raw_scores"]])
            cal_p = float(self.scaler.calibrate_logits(logits)[0][1])

            y_true.append(label)
            raw_probs.append(raw_p)
            calibrated_probs.append(cal_p)

        if not y_true:
            return {"severity": severity_label, "error": "No valid test samples processed."}

        # Compute metrics
        preds = [1 if p >= 0.5 else 0 for p in calibrated_probs]
        base_metrics = compute_metrics(y_true, preds, calibrated_probs)
        ece_res = compute_ece(y_true, calibrated_probs)
        sel_res = self.policy.evaluate_selective_risk(y_true, calibrated_probs, self.policy)

        return {
            "perturbation": severity_label,
            "accuracy": base_metrics["accuracy"],
            "roc_auc": base_metrics["roc_auc"],
            "brier_score": ece_res["brier_score"],
            "ece": ece_res["ece"],
            "coverage": sel_res["coverage"],
            "selective_risk": sel_res["selective_risk"],
            "abstention_rate": sel_res["abstention_rate"]
        }


def plot_robustness_curves(
    results: List[Dict[str, Any]],
    output_path: str
) -> str:
    """Plot performance, calibration, and abstention curves under increasing perturbation severity."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    labels = [r["perturbation"] for r in results]
    accs = [r["accuracy"] for r in results]
    aucs = [r["roc_auc"] for r in results]
    eces = [r["ece"] for r in results]
    abstentions = [r["abstention_rate"] for r in results]

    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Discrimination & Calibration plot
    ax1.plot(x, accs, "o-", label="Accuracy", color="#1f77b4", lw=2)
    ax1.plot(x, aucs, "s--", label="ROC-AUC", color="#2ca02c", lw=2)
    ax1.plot(x, eces, "^-.", label="ECE (Calibration Error)", color="#d62728", lw=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Score / Metric", fontsize=11)
    ax1.set_title("Discrimination & Calibration vs. Perturbation Severity", fontsize=12, fontweight="bold")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Abstention & Selective Risk plot
    ax2.plot(x, abstentions, "D-", label="Abstention Rate (Inconclusive)", color="#ff7f0e", lw=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("Fraction", fontsize=11)
    ax2.set_title("Uncertainty-Aware Abstention under Degradation", fontsize=12, fontweight="bold")
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return os.path.abspath(output_path)
