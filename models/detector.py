"""Deepfake detection neural network architectures with pretrained backbones and custom heads."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0Detector(nn.Module):
    """EfficientNet-B0 backbone for facial deepfake detection.
    
    Replaces ImageNet 1000-class head with binary classification:
    Class 0: Authentic / Real
    Class 1: Manipulated / Deepfake
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # In EfficientNet-B0, backbone.classifier is Sequential(Dropout, Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        self.architecture_name = "efficientnet_b0"
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, num_classes)."""
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Target convolutional layer for Grad-CAM explanation."""
        # Final convolutional block in EfficientNet-B0 features
        return self.backbone.features[-1]


class ResNet18Detector(nn.Module):
    """ResNet-18 lightweight comparison baseline for digital media forensics."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # In ResNet-18, backbone.fc is Linear(512, 1000)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        self.architecture_name = "resnet18"
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, num_classes)."""
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Target convolutional layer for Grad-CAM explanation."""
        # Final residual block in ResNet-18
        return self.backbone.layer4[-1]


def build_detector(
    architecture: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """Factory function instantiating deepfake classifier with validated configuration.
    
    Args:
        architecture: 'efficientnet_b0' or 'resnet18'.
        num_classes: Number of prediction classes (default 2).
        pretrained: Whether to initialize with ImageNet weights.
        dropout: Dropout rate before classification layer.
        
    Returns:
        Configured PyTorch nn.Module.
    """
    arch = architecture.lower().strip()
    if arch == "efficientnet_b0":
        return EfficientNetB0Detector(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif arch == "resnet18":
        return ResNet18Detector(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unsupported architecture '{architecture}'. Choose 'efficientnet_b0' or 'resnet18'.")
