"""Diagnostic plotting utilities: training curves, ROC/PR curves, and confusion matrix heatmaps."""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers and scripts
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> str:
    """Plot training and validation loss and accuracy curves across epochs."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history.get("train_loss", []), "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history.get("val_loss", []), "s--", label="Val Loss", color="#d62728")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history.get("train_acc", []), "o-", label="Train Acc", color="#2ca02c")
    ax2.plot(epochs, history.get("val_acc", []), "s--", label="Val Acc", color="#ff7f0e")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return os.path.abspath(output_path)


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], output_path: str) -> str:
    """Render and save confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return os.path.abspath(output_path)


def plot_roc_pr_curves(
    y_true: List[int],
    y_prob: List[float],
    roc_path: str,
    pr_path: str
) -> None:
    """Plot and save ROC and Precision-Recall curves."""
    os.makedirs(os.path.dirname(os.path.abspath(roc_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(pr_path)), exist_ok=True)

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    if len(np.unique(y_true_arr)) < 2:
        return

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close(fig)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#2ca02c", lw=2, label="PR Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close(fig)
