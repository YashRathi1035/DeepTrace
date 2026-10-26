"""Comprehensive evaluation metrics for digital media deepfake forensics."""

from typing import Dict, Any, List, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray]
) -> Dict[str, Any]:
    """Calculate comprehensive classification and discrimination metrics.
    
    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Manipulated).
        y_pred: Binary predictions (0 or 1).
        y_prob: Predicted manipulation probabilities for class 1 [0.0, 1.0].
        
    Returns:
        Structured dictionary with accuracy, macro/class-specific precision,
        recall, f1, ROC-AUC, Average Precision (PR-AUC), Brier score, and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))

    # Precision, Recall, F1 (macro and per-class)
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Class-specific (0: Real, 1: Fake)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # ROC-AUC & Average Precision (requires both classes present in y_true)
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.50
        pr_auc = 0.50

    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_real": round(float(prec_per_class[0]), 4) if len(prec_per_class) > 0 else 0.0,
        "recall_real": round(float(rec_per_class[0]), 4) if len(rec_per_class) > 0 else 0.0,
        "f1_real": round(float(f1_per_class[0]), 4) if len(f1_per_class) > 0 else 0.0,
        "precision_manipulated": round(float(prec_per_class[1]), 4) if len(prec_per_class) > 1 else 0.0,
        "recall_manipulated": round(float(rec_per_class[1]), 4) if len(rec_per_class) > 1 else 0.0,
        "f1_manipulated": round(float(f1_per_class[1]), 4) if len(f1_per_class) > 1 else 0.0,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "total_samples": int(len(y_true))
    }
