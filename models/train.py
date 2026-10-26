"""Training pipeline for deepfake detection models with mixed precision and early stopping."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import load_config, get_logger, set_seed
from models.detector import build_detector
from models.checkpoint import save_checkpoint, load_checkpoint
from preprocessing.dataset import DeepfakeFaceDataset
from evaluation.metrics import compute_metrics
from evaluation.plots import plot_training_history, plot_confusion_matrix, plot_roc_pr_curves


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float]:
    """Train model across one epoch using mixed precision if enabled."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (tensors, labels, _) in enumerate(dataloader):
        tensors = tensors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(tensors)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * tensors.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += tensors.size(0)

    epoch_loss = total_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, Any], List[int], List[float]]:
    """Evaluate model on validation split and compute discrimination metrics."""
    model.eval()
    total_loss = 0.0
    total = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for tensors, labels, _ in dataloader:
            tensors = tensors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(tensors)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * tensors.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total += tensors.size(0)
            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    val_loss = total_loss / max(1, total)
    metrics = compute_metrics(all_targets, all_preds, all_probs)
    return val_loss, metrics, all_targets, all_probs


def train_model(
    config_path: str = "configs/config.yaml",
    arch_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    resume_path: Optional[str] = None
) -> Dict[str, Any]:
    """Execute complete training workflow with validation-based checkpointing."""
    config = load_config(config_path)
    logger = get_logger("DeepTrace.Train", log_file=os.path.join(config["paths"]["logs_dir"], "train.log"))
    set_seed(config["system"]["seed"])

    # Hardware detection
    configured_device = config["system"]["device"].lower()
    if configured_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU execution.")

    # Hyperparameters
    architecture = arch_override or config["model"]["architecture"]
    epochs = epochs_override or config["training"]["epochs"]
    batch_size = batch_size_override or config["training"]["batch_size"]
    lr = lr_override or config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 1e-5)
    patience = config["training"].get("early_stopping_patience", 3)
    use_amp = config["system"].get("mixed_precision", True) and (device.type == "cuda")

    manifest_path = os.path.join(config["paths"]["manifests_dir"], "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(config["paths"]["manifests_dir"], "dataset_manifest.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. Run `python scripts/prepare_dataset.py --create-fixture` first."
        )

    # Datasets and Loaders
    image_size = tuple(config["preprocessing"]["image_size"])
    train_dataset = DeepfakeFaceDataset(manifest_path=manifest_path, split="train", image_size=image_size)
    val_dataset = DeepfakeFaceDataset(manifest_path=manifest_path, split="val", image_size=image_size)

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError("Training dataset has 0 samples.")

    # Class weighting for loss
    if config["training"].get("class_weights", True):
        class_weights = train_dataset.get_class_weights().to(device)
        logger.info(f"Class weights (Real: {class_weights[0]:.2f}, Manipulated: {class_weights[1]:.2f})")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 is safest for Windows multiprocessing
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Model instantiation or resumption
    start_epoch = 1
    if resume_path:
        logger.info(f"Resuming checkpoint from {resume_path}")
        model, chk = load_checkpoint(resume_path, device=device, eval_mode=False)
        start_epoch = chk.get("epoch", 0) + 1
    else:
        logger.info(f"Building detector with architecture: {architecture}")
        model = build_detector(
            architecture=architecture,
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"].get("pretrained", True),
            dropout=config["model"].get("dropout", 0.2)
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Tracking records
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": []
    }

    best_val_auc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = {}

    checkpoints_dir = config["paths"]["checkpoints_dir"]
    plots_dir = config["paths"]["plots_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(f"Starting training for {epochs} epochs (AMP: {use_amp}, Device: {device})...")

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp
        )

        val_loss, val_metrics, val_targets, val_probs = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["roc_auc"]

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_auc"].append(round(val_auc, 4))

        logger.info(
            f"Epoch [{epoch:02d}/{start_epoch + epochs - 1}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Checkpoint selection (prefer higher AUC or lower Loss)
        is_best = (val_auc > best_val_auc) or (val_auc == best_val_auc and val_loss < best_val_loss)
        if is_best:
            best_val_auc = val_auc
            best_val_loss = val_loss
            patience_counter = 0
            best_metrics = val_metrics
            chk_path = os.path.join(checkpoints_dir, f"{architecture}_best.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                checkpoint_path=chk_path,
                is_best=True
            )
            logger.info(f"New best checkpoint saved (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without validation improvement.")
                break

    # Save training curves and diagnostic plots
    curves_path = os.path.join(plots_dir, "training_curves.png")
    plot_training_history(history, curves_path)

    cm_path = os.path.join(plots_dir, "val_confusion_matrix.png")
    plot_confusion_matrix(best_metrics.get("confusion_matrix", [[0, 0], [0, 0]]), ["Real", "Manipulated"], cm_path)

    # Save training history JSON
    history_file = os.path.join(checkpoints_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": architecture,
            "best_metrics": best_metrics,
            "history": history
        }, f, indent=2)

    logger.info("Training pipeline finished successfully.")
    return {
        "best_metrics": best_metrics,
        "history": history,
        "checkpoint": os.path.join(checkpoints_dir, "best_model.pt")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTrace Model Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        arch_override=args.arch,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        resume_path=args.resume
    )
