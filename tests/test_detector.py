"""Automated unit tests for deepfake detection models, checkpoints, metrics, and inference."""

import os
import tempfile
import numpy as np
import torch
import pytest

from models.detector import build_detector, EfficientNetB0Detector, ResNet18Detector
from models.checkpoint import save_checkpoint, load_checkpoint
from models.predict import DeepfakeDetector
from evaluation.metrics import compute_metrics


def test_model_architecture_shapes():
    """Verify forward pass output tensor dimensions for both backbones."""
    # EfficientNet-B0
    eff_model = build_detector("efficientnet_b0", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out_eff = eff_model(x)
    assert out_eff.shape == (2, 2)
    assert hasattr(eff_model, "get_target_layer")

    # ResNet-18
    res_model = build_detector("resnet18", num_classes=2, pretrained=False)
    out_res = res_model(x)
    assert out_res.shape == (2, 2)
    assert hasattr(res_model, "get_target_layer")

    # Invalid architecture rejection
    with pytest.raises(ValueError):
        build_detector("invalid_network_name")


def test_checkpoint_save_and_load(tmp_path):
    """Test checkpoint serialization, parameter preservation, and safe restoration."""
    chk_file = str(tmp_path / "test_model.pt")
    model = build_detector("efficientnet_b0", num_classes=2, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dummy_config = {
        "model": {"labels": ["Real", "Manipulated"], "dropout": 0.2},
        "system": {"seed": 42}
    }
    dummy_metrics = {"accuracy": 0.95, "roc_auc": 0.99}

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        metrics=dummy_metrics,
        config=dummy_config,
        checkpoint_path=chk_file,
        is_best=False
    )
    assert os.path.exists(chk_file)

    loaded_model, meta = load_checkpoint(chk_file, device=torch.device("cpu"), eval_mode=True)
    assert meta["epoch"] == 5
    assert meta["architecture"] == "efficientnet_b0"
    assert meta["metrics"]["accuracy"] == 0.95
    assert meta["labels"] == ["Real", "Manipulated"]

    # Verify model is executable
    x = torch.randn(1, 3, 224, 224)
    out = loaded_model(x)
    assert out.shape == (1, 2)


def test_metrics_computation():
    """Verify precision, recall, f1, ROC-AUC, and confusion matrix calculation."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["total_samples"] == 4


def test_deepfake_detector_inference(tmp_path):
    """Verify DeepfakeDetector structured outputs, inference latency, and batch handling."""
    chk_file = str(tmp_path / "test_detector.pt")
    model = build_detector("efficientnet_b0", num_classes=2, pretrained=False)
    save_checkpoint(
        model=model,
        optimizer=None,
        epoch=1,
        metrics={},
        config={"model": {"labels": ["Real", "Manipulated"]}},
        checkpoint_path=chk_file
    )

    detector = DeepfakeDetector(checkpoint_path=chk_file, device="cpu")

    # Batch prediction
    batch = torch.randn(3, 3, 224, 224)
    preds = detector.predict_batch(batch)
    assert len(preds) == 3

    for p in preds:
        assert p["predicted_class"] in ["Real", "Manipulated"]
        assert 0.0 <= p["manipulation_score"] <= 1.0
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["inference_time_ms"] >= 0.0
        assert len(p["raw_scores"]) == 2
        assert len(p["probabilities"]) == 2
