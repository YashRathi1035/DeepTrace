"""Cross-dataset generalization benchmarks: evaluating domain shift and cross-generator robustness."""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from models.predict import DeepfakeDetector
from forensic.calibration import TemperatureScaler, compute_ece, ReliabilityDecisionPolicy
from evaluation.metrics import compute_metrics


def run_cross_dataset_benchmark(
    detector: DeepfakeDetector,
    scaler: TemperatureScaler,
    policy: ReliabilityDecisionPolicy,
    in_domain_samples: List[Dict[str, Any]],
    cross_domain_samples: List[Dict[str, Any]],
    in_domain_name: str = "FaceForensics++ (Held-out Test)",
    cross_domain_name: str = "Celeb-DF v2 (External)"
) -> Dict[str, Any]:
    """Execute comparative generalization benchmark evaluating domain shift degradation.
    
    Args:
        detector: Deepfake detection model inference engine.
        scaler: Calibrated TemperatureScaler instance.
        policy: ReliabilityDecisionPolicy instance.
        in_domain_samples: Test samples from the same training distribution.
        cross_domain_samples: External evaluation samples.
        in_domain_name: Dataset identifier.
        cross_domain_name: Cross-domain dataset identifier.
        
    Returns:
        Structured dictionary comparing discrimination, calibration, and abstention across domains.
    """
    def evaluate_split(samples: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        y_true = []
        raw_probs = []
        cal_probs = []

        for s in samples:
            p = s["image_path"]
            if not os.path.exists(p):
                continue
            pred = detector.predict_image(p)
            raw_p = pred["manipulation_score"]
            logits = np.array([pred["raw_scores"]])
            cal_p = float(scaler.calibrate_logits(logits)[0][1])

            y_true.append(int(s["label"]))
            raw_probs.append(raw_p)
            cal_probs.append(cal_p)

        if not y_true:
            return {"dataset_name": name, "error": "No accessible sample files."}

        preds = [1 if p >= 0.5 else 0 for p in cal_probs]
        base = compute_metrics(y_true, preds, cal_probs)
        ece_res = compute_ece(y_true, cal_probs)
        sel_res = policy.evaluate_selective_risk(y_true, cal_probs, policy)

        return {
            "dataset_name": name,
            "total_samples": len(y_true),
            "accuracy": base["accuracy"],
            "roc_auc": base["roc_auc"],
            "pr_auc": base["pr_auc"],
            "f1_macro": base["f1_macro"],
            "brier_score": ece_res["brier_score"],
            "ece": ece_res["ece"],
            "coverage": sel_res["coverage"],
            "selective_risk": sel_res["selective_risk"],
            "abstention_rate": sel_res["abstention_rate"]
        }

    in_domain_eval = evaluate_split(in_domain_samples, in_domain_name)
    cross_domain_eval = evaluate_split(cross_domain_samples, cross_domain_name)

    # Compute domain shift deltas
    delta_auc = None
    delta_acc = None
    delta_ece = None

    if "error" not in in_domain_eval and "error" not in cross_domain_eval:
        delta_auc = round(cross_domain_eval["roc_auc"] - in_domain_eval["roc_auc"], 4)
        delta_acc = round(cross_domain_eval["accuracy"] - in_domain_eval["accuracy"], 4)
        delta_ece = round(cross_domain_eval["ece"] - in_domain_eval["ece"], 4)

    return {
        "in_domain": in_domain_eval,
        "cross_domain": cross_domain_eval,
        "domain_shift_impact": {
            "delta_roc_auc": delta_auc,
            "delta_accuracy": delta_acc,
            "delta_ece": delta_ece,
            "note": (
                "Deepfake classifiers typically experience substantial cross-dataset performance drops "
                "due to distinct generative artifacts, varying compression pipelines, and identity shifts."
            )
        }
    }
