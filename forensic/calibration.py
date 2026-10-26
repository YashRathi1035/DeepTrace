"""Confidence calibration (Temperature Scaling), ECE estimation, and reliability-aware abstention."""

import os
import json
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss


class TemperatureScaler(nn.Module):
    """Post-hoc confidence calibration using single-parameter Temperature Scaling.
    
    Transforms uncalibrated logits z into calibrated probabilities:
        p_calibrated = softmax(z / T)
    where T > 0 is learned by minimizing Negative Log-Likelihood (NLL) on validation data only.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature parameter."""
        temp = self.temperature.clamp(min=0.01)
        return logits / temp

    def fit(
        self,
        val_logits: Union[torch.Tensor, np.ndarray],
        val_labels: Union[torch.Tensor, np.ndarray],
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """Optimize temperature parameter on held-out validation logits using L-BFGS.
        
        Args:
            val_logits: Validation set unscaled logits of shape (N, 2).
            val_labels: Ground truth binary labels of shape (N,).
            lr: Optimizer learning rate.
            max_iter: Maximum optimization iterations.
            
        Returns:
            Learned optimal temperature float.
        """
        if isinstance(val_logits, np.ndarray):
            val_logits = torch.from_numpy(val_logits).float()
        if isinstance(val_labels, np.ndarray):
            val_labels = torch.from_numpy(val_labels).long()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)

        return float(self.temperature.item())

    def calibrate_logits(self, logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Produce calibrated probabilities for input logits."""
        if isinstance(logits, np.ndarray):
            t_logits = torch.from_numpy(logits).float()
        else:
            t_logits = logits.float()

        with torch.no_grad():
            scaled = self.forward(t_logits)
            probs = torch.softmax(scaled, dim=-1)
        return probs.cpu().numpy()

    def save(self, output_path: str) -> str:
        """Save calibration parameter to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = {
            "method": "temperature_scaling",
            "temperature": round(float(self.temperature.item()), 4),
            "timestamp": str(np.datetime64("now"))
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return os.path.abspath(output_path)

    @classmethod
    def load(cls, file_path: str) -> "TemperatureScaler":
        """Load temperature parameter from saved JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        temp = float(data.get("temperature", 1.0))
        scaler = cls(initial_temperature=temp)
        return scaler


def compute_ece(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    n_bins: int = 10
) -> Dict[str, Any]:
    """Calculate Expected Calibration Error (ECE) and Brier Score using equal-width binning.
    
    ECE measures the absolute discrepancy between predicted confidence and observed empirical accuracy:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted confidence/probability for the predicted class [0.5, 1.0].
        n_bins: Number of equal-width calibration bins (default 10).
        
    Returns:
        Structured dictionary containing ECE, Brier score, and per-bin diagnostics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Determine predicted class and confidence
    # If y_prob is class 1 probability [0, 1], confidence is max(p, 1-p)
    if np.min(y_prob) < 0.5:
        preds = (y_prob >= 0.5).astype(int)
        confidences = np.where(preds == 1, y_prob, 1.0 - y_prob)
    else:
        preds = (y_prob >= 0.5).astype(int)
        confidences = y_prob

    accuracies = (preds == y_true).astype(float)
    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)

    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]

        # Samples within this confidence bin
        if i == n_bins - 1:
            in_bin = (confidences >= b_low) & (confidences <= b_high)
        else:
            in_bin = (confidences >= b_low) & (confidences < b_high)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            weight = count / len(y_true)
            ece += weight * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = float((b_low + b_high) / 2.0)

        bin_data.append({
            "bin_idx": i,
            "range": [round(b_low, 3), round(b_high, 3)],
            "count": count,
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4)
        })

    brier = float(brier_score_loss(y_true, y_prob if np.max(y_prob) <= 1.0 else confidences))

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
        "bin_data": bin_data,
        "total_samples": len(y_true)
    }


def plot_reliability_diagram(ece_dict: Dict[str, Any], output_path: str) -> str:
    """Generate and save publication-quality Reliability Diagram."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    bin_data = ece_dict["bin_data"]

    confs = [b["confidence"] for b in bin_data if b["count"] > 0]
    accs = [b["accuracy"] for b in bin_data if b["count"] > 0]
    counts = [b["count"] for b in bin_data if b["count"] > 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability curve
    ax1.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", label="Perfect Calibration")
    if confs:
        ax1.bar(confs, accs, width=0.04, alpha=0.6, color="#1f77b4", edgecolor="black", label="Empirical Accuracy")
        ax1.scatter(confs, accs, color="#d62728", zorder=5)

    ax1.set_xlim([0.5, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_title(f"Reliability Diagram (ECE = {ece_dict['ece']:.4f}, Brier = {ece_dict['brier_score']:.4f})", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bin sample count histogram
    if confs:
        ax2.bar(confs, counts, width=0.04, color="#2ca02c", alpha=0.7, edgecolor="black")
    ax2.set_xlim([0.5, 1.0])
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return os.path.abspath(output_path)


class ReliabilityDecisionPolicy:
    """Uncertainty-aware forensic decision engine with selective abstention.
    
    Instead of forcing binary predictions on ambiguous or corrupted samples,
    the policy classifies inputs into three evidential tiers:
    - Likely Authentic
    - Potentially Manipulated
    - Inconclusive (Abstain)
    """

    def __init__(
        self,
        fake_threshold: float = 0.65,
        real_threshold: float = 0.35,
        min_quality_score: float = 0.35
    ):
        if real_threshold >= fake_threshold:
            raise ValueError(f"real_threshold ({real_threshold}) must be < fake_threshold ({fake_threshold}).")
        self.fake_threshold = fake_threshold
        self.real_threshold = real_threshold
        self.min_quality_score = min_quality_score

    def decide(
        self,
        manipulation_score: float,
        quality_score: float = 1.0,
        quality_warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Apply decision policy to calibrated manipulation probability and input quality.
        
        Args:
            manipulation_score: Calibrated probability that input is manipulated [0.0, 1.0].
            quality_score: Composite image/face quality score [0.0, 1.0].
            quality_warnings: Diagnostic warnings.
            
        Returns:
            Dictionary with 'decision' label, 'is_decisive' boolean, and explanatory rationale.
        """
        warnings = list(quality_warnings or [])

        # Quality check failure leads to automatic abstention
        if quality_score < self.min_quality_score:
            return {
                "decision": "Inconclusive",
                "is_decisive": False,
                "confidence_tier": "Low Quality / Degraded Input",
                "rationale": f"Input facial crop quality ({quality_score:.2f}) falls below forensic reliability threshold ({self.min_quality_score:.2f}). System abstains.",
                "warnings": warnings + ["Abstained due to low input visual quality."]
            }

        # Score threshold policy
        if manipulation_score >= self.fake_threshold:
            return {
                "decision": "Potentially Manipulated",
                "is_decisive": True,
                "confidence_tier": "Decisive Manipulation",
                "rationale": f"Calibrated manipulation probability ({manipulation_score:.3f}) exceeds suspicion threshold ({self.fake_threshold:.2f}).",
                "warnings": warnings
            }
        elif manipulation_score <= self.real_threshold:
            return {
                "decision": "Likely Authentic",
                "is_decisive": True,
                "confidence_tier": "Decisive Authentic",
                "rationale": f"Calibrated manipulation probability ({manipulation_score:.3f}) falls below authenticity threshold ({self.real_threshold:.2f}).",
                "warnings": warnings
            }
        else:
            return {
                "decision": "Inconclusive",
                "is_decisive": False,
                "confidence_tier": "Ambiguous / High Uncertainty",
                "rationale": f"Calibrated probability ({manipulation_score:.3f}) lies inside the inconclusive margin [{self.real_threshold:.2f}, {self.fake_threshold:.2f}]. System abstains to avoid false decision.",
                "warnings": warnings + ["Decision abstained due to high model uncertainty."]
            }

    @staticmethod
    def evaluate_selective_risk(
        y_true: Union[List[int], np.ndarray],
        y_prob: Union[List[float], np.ndarray],
        policy: "ReliabilityDecisionPolicy"
    ) -> Dict[str, Any]:
        """Measure Coverage and Selective Risk across a dataset.
        
        - Coverage: Fraction of inputs where system returns a decisive prediction.
        - Selective Risk: Error rate among the accepted (decisive) predictions.
        
        Args:
            y_true: Ground truth labels (0 or 1).
            y_prob: Predicted manipulation probabilities [0.0, 1.0].
            policy: Policy instance.
            
        Returns:
            Dictionary with coverage, selective risk, decisive sample count, and error count.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        decisive_indices = []
        decisive_preds = []

        for i, p in enumerate(y_prob):
            res = policy.decide(manipulation_score=p)
            if res["is_decisive"]:
                decisive_indices.append(i)
                decisive_preds.append(1 if res["decision"] == "Potentially Manipulated" else 0)

        total_samples = len(y_true)
        n_decisive = len(decisive_indices)

        if n_decisive > 0:
            accepted_true = y_true[decisive_indices]
            accepted_preds = np.array(decisive_preds)
            errors = int(np.sum(accepted_true != accepted_preds))
            selective_risk = float(errors / n_decisive)
            selective_accuracy = 1.0 - selective_risk
        else:
            errors = 0
            selective_risk = 0.0
            selective_accuracy = 1.0

        coverage = float(n_decisive / max(1, total_samples))
        abstention_rate = 1.0 - coverage

        return {
            "coverage": round(coverage, 4),
            "abstention_rate": round(abstention_rate, 4),
            "selective_risk": round(selective_risk, 4),
            "selective_accuracy": round(selective_accuracy, 4),
            "total_samples": total_samples,
            "decisive_samples": n_decisive,
            "abstention_samples": total_samples - n_decisive,
            "errors_among_decisive": errors
        }
