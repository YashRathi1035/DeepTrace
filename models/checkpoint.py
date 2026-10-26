"""Checkpoint serialization, metadata preservation, and validation for DeepTrace models."""

import os
import hashlib
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from models.detector import build_detector


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    checkpoint_path: str,
    is_best: bool = False
) -> str:
    """Save model checkpoint with complete metadata, training configuration, and metrics.
    
    Args:
        model: PyTorch model instance.
        optimizer: Optimizer state (optional).
        epoch: Current epoch index.
        metrics: Dictionary of validation metrics (loss, accuracy, auc, etc.).
        config: Training configuration dictionary.
        checkpoint_path: Path where checkpoint will be saved.
        is_best: Whether this is the best performing checkpoint so far.
        
    Returns:
        Absolute path to the saved checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    architecture = getattr(model, "architecture_name", "efficientnet_b0")
    num_classes = getattr(model, "num_classes", 2)

    payload = {
        "version": "1.0",
        "architecture": architecture,
        "num_classes": num_classes,
        "labels": config.get("model", {}).get("labels", ["Real", "Manipulated"]),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }

    torch.save(payload, checkpoint_path)

    if is_best:
        dir_name = os.path.dirname(os.path.abspath(checkpoint_path))
        best_path = os.path.join(dir_name, "best_model.pt")
        if best_path != os.path.abspath(checkpoint_path):
            torch.save(payload, best_path)

    return os.path.abspath(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Safely load a DeepTrace model checkpoint and reconstruct the architecture.
    
    Args:
        checkpoint_path: Path to .pt or .pth checkpoint file.
        device: Target torch.device (CUDA or CPU). Auto-detected if None.
        eval_mode: Whether to set model.eval() upon loading.
        
    Returns:
        Tuple of (model, checkpoint_metadata_dict).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint payload safely
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize checkpoint from {checkpoint_path}: {e}")

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format at {checkpoint_path}; missing 'model_state_dict'.")

    arch = payload.get("architecture", "efficientnet_b0")
    num_classes = payload.get("num_classes", 2)
    dropout = payload.get("config", {}).get("model", {}).get("dropout", 0.2)

    # Instantiate model
    model = build_detector(
        architecture=arch,
        num_classes=num_classes,
        pretrained=False,  # Weights loaded from state_dict
        dropout=dropout
    )

    model.load_state_dict(payload["model_state_dict"])
    model.to(device)

    if eval_mode:
        model.eval()

    return model, payload
