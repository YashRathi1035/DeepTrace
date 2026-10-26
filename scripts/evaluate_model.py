"""Master evaluation runner: calibration fitting, reliability analysis, robustness, and ablation studies."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import load_config, get_logger, set_seed
from models.predict import DeepfakeDetector
from preprocessing.dataset import DeepfakeFaceDataset
from forensic.calibration import (
    TemperatureScaler,
    compute_ece,
    plot_reliability_diagram,
    ReliabilityDecisionPolicy
)
from evaluation.metrics import compute_metrics
from evaluation.robustness import (
    RobustnessBenchmark,
    apply_jpeg_compression,
    apply_gaussian_blur,
    apply_downsampling,
    apply_gaussian_noise,
    plot_robustness_curves
)


def run_full_evaluation(
    config_path: str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """Execute end-to-end scientific evaluation of DeepTrace detector."""
    config = load_config(config_path)
    logger = get_logger("DeepTrace.Evaluate")
    set_seed(config["system"]["seed"])

    chk_path = checkpoint_path or config["model"].get("default_checkpoint", "artifacts/checkpoints/best_model.pt")
    if not os.path.exists(chk_path):
        raise FileNotFoundError(f"Model checkpoint not found: {chk_path}. Run training first.")

    manifest_path = os.path.join(config["paths"]["manifests_dir"], "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(config["paths"]["manifests_dir"], "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run dataset preparation first.")

    # 1. Initialize detector
    device_name = "cuda" if (config["system"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    detector = DeepfakeDetector(checkpoint_path=chk_path, device=device_name)
    logger.info(f"Loaded detector ({detector.architecture}) on {device_name}.")

    # 2. Extract validation and test samples
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])

    val_samples = [s for s in samples if s.get("split") == "val"]
    test_samples = [s for s in samples if s.get("split") == "test"]
    logger.info(f"Loaded {len(val_samples)} validation samples, {len(test_samples)} test samples.")

    # 3. Fit Temperature Scaler on Validation Split
    logger.info("Fitting Temperature Scaling calibration on validation split...")
    val_logits = []
    val_labels = []

    for s in val_samples:
        p = s["image_path"]
        if os.path.exists(p):
            pred = detector.predict_image(p)
            val_logits.append(pred["raw_scores"])
            val_labels.append(int(s["label"]))

    scaler = TemperatureScaler(initial_temperature=1.0)
    if val_logits:
        t_val = scaler.fit(np.array(val_logits), np.array(val_labels))
        logger.info(f"Optimized calibration temperature: T = {t_val:.4f}")
    else:
        logger.warning("No valid validation samples found; using default temperature T = 1.0.")

    # Save calibration artifact
    cal_path = os.path.join(config["paths"]["checkpoints_dir"], "calibration.json")
    scaler.save(cal_path)

    # 4. Evaluate Calibration on Held-out Test Set
    test_logits = []
    test_labels = []
    test_raw_probs = []
    test_cal_probs = []

    for s in test_samples:
        p = s["image_path"]
        if os.path.exists(p):
            pred = detector.predict_image(p)
            raw_p = pred["manipulation_score"]
            logits_arr = np.array([pred["raw_scores"]])
            cal_p = float(scaler.calibrate_logits(logits_arr)[0][1])

            test_logits.append(pred["raw_scores"])
            test_labels.append(int(s["label"]))
            test_raw_probs.append(raw_p)
            test_cal_probs.append(cal_p)

    if not test_labels:
        raise ValueError("No test samples available for evaluation.")

    # Compute ECE before and after calibration
    ece_uncal = compute_ece(test_labels, test_raw_probs)
    ece_cal = compute_ece(test_labels, test_cal_probs)
    logger.info(f"Test ECE — Uncalibrated: {ece_uncal['ece']:.4f} | Calibrated: {ece_cal['ece']:.4f}")

    # Plot Reliability Diagram
    rel_plot_path = os.path.join(config["paths"]["plots_dir"], "reliability_diagram.png")
    plot_reliability_diagram(ece_cal, rel_plot_path)

    # 5. Evaluate Selective Risk & Abstention Policy
    policy = ReliabilityDecisionPolicy(
        fake_threshold=config["reliability"]["abstention"]["fake_threshold"],
        real_threshold=config["reliability"]["abstention"]["real_threshold"]
    )
    sel_eval = policy.evaluate_selective_risk(test_labels, test_cal_probs, policy)
    logger.info(f"Reliability Policy: Coverage = {sel_eval['coverage']:.2%}, Selective Risk = {sel_eval['selective_risk']:.2%}, Abstention = {sel_eval['abstention_rate']:.2%}")

    # 6. Robustness Stress-Tests
    logger.info("Executing controlled robustness stress-tests...")
    benchmark = RobustnessBenchmark(detector=detector, scaler=scaler, policy=policy)
    perturbation_specs = [
        ("Clean Baseline", lambda img: img),
        ("JPEG Q=90", lambda img: apply_jpeg_compression(img, 90)),
        ("JPEG Q=50", lambda img: apply_jpeg_compression(img, 50)),
        ("JPEG Q=30", lambda img: apply_jpeg_compression(img, 30)),
        ("Blur sigma=1", lambda img: apply_gaussian_blur(img, 1.0)),
        ("Blur sigma=2", lambda img: apply_gaussian_blur(img, 2.0)),
        ("Downsample 0.5x", lambda img: apply_downsampling(img, 0.5)),
        ("Downsample 0.35x", lambda img: apply_downsampling(img, 0.35)),
        ("Noise sigma=15", lambda img: apply_gaussian_noise(img, 15.0)),
    ]

    robustness_results = []
    for label, fn in perturbation_specs:
        res = benchmark.run_perturbation_experiment(test_samples, fn, label)
        robustness_results.append(res)

    robust_plot_path = os.path.join(config["paths"]["plots_dir"], "robustness_curves.png")
    plot_robustness_curves(robustness_results, robust_plot_path)

    # 7. Ablation Study Comparison Table
    base_raw_preds = [1 if p >= 0.5 else 0 for p in test_raw_probs]
    base_metrics = compute_metrics(test_labels, base_raw_preds, test_raw_probs)

    cal_preds = [1 if p >= 0.5 else 0 for p in test_cal_probs]
    cal_metrics = compute_metrics(test_labels, cal_preds, test_cal_probs)

    ablation_rows = [
        {
            "Pipeline Configuration": "1. Baseline (Uncalibrated Binary)",
            "Accuracy": base_metrics["accuracy"],
            "ROC-AUC": base_metrics["roc_auc"],
            "Brier Score": ece_uncal["brier_score"],
            "ECE": ece_uncal["ece"],
            "Coverage": 1.0,
            "Selective Risk": round(1.0 - base_metrics["accuracy"], 4),
            "Abstention Rate": 0.0
        },
        {
            "Pipeline Configuration": "2. Calibrated (Temperature Scaling)",
            "Accuracy": cal_metrics["accuracy"],
            "ROC-AUC": cal_metrics["roc_auc"],
            "Brier Score": ece_cal["brier_score"],
            "ECE": ece_cal["ece"],
            "Coverage": 1.0,
            "Selective Risk": round(1.0 - cal_metrics["accuracy"], 4),
            "Abstention Rate": 0.0
        },
        {
            "Pipeline Configuration": "3. Reliability-Aware Policy (Calibrated + Abstention)",
            "Accuracy": sel_eval["selective_accuracy"],
            "ROC-AUC": cal_metrics["roc_auc"],
            "Brier Score": ece_cal["brier_score"],
            "ECE": ece_cal["ece"],
            "Coverage": sel_eval["coverage"],
            "Selective Risk": sel_eval["selective_risk"],
            "Abstention Rate": sel_eval["abstention_rate"]
        }
    ]

    ablation_df = pd.DataFrame(ablation_rows)
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    ablation_csv_path = os.path.join(reports_dir, "ablation_study.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    logger.info(f"Saved ablation comparison table to {ablation_csv_path}")

    return {
        "temperature": scaler.temperature.item(),
        "uncalibrated_ece": ece_uncal["ece"],
        "calibrated_ece": ece_cal["ece"],
        "selective_eval": sel_eval,
        "ablation_table": ablation_rows,
        "robustness": robustness_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTrace Scientific Evaluation Runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    args = parser.parse_args()

    run_full_evaluation(config_path=args.config, checkpoint_path=args.checkpoint)
