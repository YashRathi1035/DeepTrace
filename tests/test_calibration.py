"""Automated unit tests for confidence calibration, ECE, reliability abstention, and image perturbations."""

import os
import tempfile
import numpy as np
import torch
import pytest

from forensic.calibration import (
    TemperatureScaler,
    compute_ece,
    ReliabilityDecisionPolicy
)
from evaluation.robustness import (
    apply_jpeg_compression,
    apply_gaussian_blur,
    apply_downsampling,
    apply_gaussian_noise
)


def test_temperature_scaler_optimization(tmp_path):
    """Test Temperature Scaling parameter optimization and persistence."""
    scaler = TemperatureScaler(initial_temperature=1.0)

    # Simulated overconfident logits on validation set
    val_logits = np.array([
        [4.0, -4.0],
        [3.5, -3.5],
        [-3.8, 3.8],
        [-4.2, 4.2]
    ])
    val_labels = np.array([0, 0, 1, 1])

    optimal_t = scaler.fit(val_logits, val_labels)
    assert optimal_t > 0.0
    assert isinstance(optimal_t, float)

    # Test calibrate_logits
    test_logits = np.array([[2.0, -2.0]])
    cal_probs = scaler.calibrate_logits(test_logits)
    assert cal_probs.shape == (1, 2)
    assert np.isclose(np.sum(cal_probs), 1.0)

    # Test save and load
    save_file = str(tmp_path / "temp.json")
    scaler.save(save_file)
    assert os.path.exists(save_file)

    loaded_scaler = TemperatureScaler.load(save_file)
    assert np.isclose(float(loaded_scaler.temperature.item()), optimal_t, atol=1e-3)


def test_ece_computation():
    """Verify Expected Calibration Error calculation on well-calibrated and uncalibrated predictions."""
    # 1. Perfectly calibrated predictions: confidence = 1.0, accuracy = 1.0
    y_true_perfect = [0, 1]
    y_prob_perfect = [0.0, 1.0]
    ece_perf = compute_ece(y_true_perfect, y_prob_perfect)
    assert ece_perf["ece"] < 0.05
    assert ece_perf["brier_score"] == 0.0

    # 2. Overconfident inaccurate predictions: confidence ~1.0, accuracy = 0.0
    y_true_bad = [0, 0]
    y_prob_bad = [0.99, 0.99]
    ece_bad = compute_ece(y_true_bad, y_prob_bad)
    assert ece_bad["ece"] > 0.80  # Significant calibration gap


def test_reliability_decision_policy():
    """Verify three-tier decision engine: Authentic, Manipulated, and Inconclusive abstention."""
    policy = ReliabilityDecisionPolicy(fake_threshold=0.65, real_threshold=0.35, min_quality_score=0.35)

    # High fake confidence
    res_fake = policy.decide(manipulation_score=0.88, quality_score=0.9)
    assert res_fake["decision"] == "Potentially Manipulated"
    assert res_fake["is_decisive"] is True

    # High authentic confidence
    res_real = policy.decide(manipulation_score=0.15, quality_score=0.9)
    assert res_real["decision"] == "Likely Authentic"
    assert res_real["is_decisive"] is True

    # Ambiguous / uncertain score -> Abstain
    res_uncertain = policy.decide(manipulation_score=0.52, quality_score=0.9)
    assert res_uncertain["decision"] == "Inconclusive"
    assert res_uncertain["is_decisive"] is False

    # Low visual quality -> Automatic Abstention regardless of model score
    res_degraded = policy.decide(manipulation_score=0.99, quality_score=0.20)
    assert res_degraded["decision"] == "Inconclusive"
    assert res_degraded["is_decisive"] is False


def test_selective_risk_evaluation():
    """Test Coverage and Selective Risk calculations across sample cohort."""
    policy = ReliabilityDecisionPolicy(fake_threshold=0.70, real_threshold=0.30)

    # 4 samples: 2 clear real, 1 clear fake, 1 ambiguous (0.50)
    y_true = [0, 0, 1, 1]
    y_prob = [0.10, 0.15, 0.90, 0.50]

    eval_res = policy.evaluate_selective_risk(y_true, y_prob, policy)
    # 3 decisive samples out of 4 -> Coverage = 0.75
    assert eval_res["coverage"] == 0.75
    assert eval_res["abstention_rate"] == 0.25
    assert eval_res["decisive_samples"] == 3
    # All 3 decisive samples were correct -> Selective Risk = 0.0
    assert eval_res["selective_risk"] == 0.0
    assert eval_res["selective_accuracy"] == 1.0


def test_image_perturbations():
    """Test image transformation functions maintain image shapes and data types."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # JPEG
    jpeg_img = apply_jpeg_compression(img, quality=40)
    assert jpeg_img.shape == (100, 100, 3)
    assert jpeg_img.dtype == np.uint8

    # Gaussian Blur
    blur_img = apply_gaussian_blur(img, sigma=2.0)
    assert blur_img.shape == (100, 100, 3)
    assert blur_img.dtype == np.uint8

    # Downsampling
    down_img = apply_downsampling(img, scale=0.5)
    assert down_img.shape == (100, 100, 3)
    assert down_img.dtype == np.uint8

    # Gaussian Noise
    noise_img = apply_gaussian_noise(img, sigma=20.0)
    assert noise_img.shape == (100, 100, 3)
    assert noise_img.dtype == np.uint8
