"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability engine for deepfake forensics."""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes class-discriminative localization maps using gradient attribution."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """Initialize GradCAM hook manager.
        
        Args:
            model: PyTorch deepfake classification model.
            target_layer: Convolutional layer to probe. If None, queries model.get_target_layer().
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_target_layer"):
                self.target_layer = model.get_target_layer()
            else:
                raise AttributeError("Model must provide get_target_layer() or target_layer must be specified.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target convolutional layer."""
        def forward_hook(module, inp, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        f_hook = self.target_layer.register_forward_hook(forward_hook)
        b_hook = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [f_hook, b_hook]

    def remove_hooks(self) -> None:
        """Remove PyTorch hook handles to prevent memory leaks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __del__(self):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = 1
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM saliency heatmap for target class.
        
        Args:
            input_tensor: Normalized image tensor of shape (1, 3, H, W).
            target_class: Class index to explain (0 = Real, 1 = Manipulated).
                          If None, uses predicted class with highest score.
                          
        Returns:
            2D numpy float array of shape (H, W) with values normalized to [0.0, 1.0].
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"GradCAM expects input tensor of shape (1, 3, H, W); got {input_tensor.shape}.")

        device = next(self.model.parameters()).device
        x = input_tensor.clone().detach().to(device)
        x.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(x)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad_k)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps: sum_k (alpha_k * A_k)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # Discard negative contributions (features that decrease class confidence)

        # Upsample to input tensor resolution (H, W)
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        min_val = np.min(cam_np)
        max_val = np.max(cam_np)
        if max_val - min_val > 1e-8:
            cam_norm = (cam_np - min_val) / (max_val - min_val)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    @staticmethod
    def overlay_heatmap(
        original_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.40,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend 2D heatmap with original BGR face crop.
        
        Args:
            original_bgr: Source face crop image (BGR uint8).
            heatmap: 2D float array in [0.0, 1.0].
            alpha: Heatmap blend weight (0.0 = only original, 1.0 = only heatmap).
            colormap: OpenCV colormap enum (e.g. cv2.COLORMAP_JET).
            
        Returns:
            Tuple of (overlay_bgr, colored_heatmap_bgr).
        """
        h, w = original_bgr.shape[:2]
        # Resize heatmap if shape differs from original image
        if heatmap.shape[:2] != (h, w):
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap

        # Convert to 8-bit [0, 255]
        uint8_cam = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
        colored_cam = cv2.applyColorMap(uint8_cam, colormap)

        # Blend
        overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, colored_cam, alpha, 0)
        return overlay, colored_cam


def generate_explanation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_bgr: np.ndarray,
    target_class: int = 1,
    output_dir: Optional[str] = None,
    prefix: str = "frame"
) -> Dict[str, Any]:
    """Unified wrapper generating Grad-CAM heatmap, overlay, and saving artifact images.
    
    Args:
        model: Deepfake detection model.
        input_tensor: Preprocessed tensor (1, 3, H, W).
        original_bgr: Source face crop (BGR uint8).
        target_class: Class to explain (default 1 = Manipulated).
        output_dir: Optional directory to save output PNG files.
        prefix: Filename prefix for saved images.
        
    Returns:
        Dictionary containing:
        - 'heatmap': 2D float array in [0.0, 1.0]
        - 'overlay_bgr': BGR blended image
        - 'overlay_rgb': RGB blended image (for UI rendering)
        - 'heatmap_bgr': Colored heatmap alone
        - 'target_class': Target class index explained
        - 'saved_heatmap_path': Optional saved file path
        - 'saved_overlay_path': Optional saved file path
    """
    grad_cam = GradCAM(model=model)
    try:
        heatmap = grad_cam.generate_cam(input_tensor, target_class=target_class)
        overlay_bgr, colored_heatmap_bgr = grad_cam.overlay_heatmap(original_bgr, heatmap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        result: Dict[str, Any] = {
            "heatmap": heatmap,
            "overlay_bgr": overlay_bgr,
            "overlay_rgb": overlay_rgb,
            "colored_heatmap_bgr": colored_heatmap_bgr,
            "target_class": target_class,
            "is_success": True
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            overlay_file = os.path.join(output_dir, f"{prefix}_gradcam_overlay.png")
            heatmap_file = os.path.join(output_dir, f"{prefix}_gradcam_heatmap.png")
            cv2.imwrite(overlay_file, overlay_bgr)
            cv2.imwrite(heatmap_file, colored_heatmap_bgr)
            result["saved_overlay_path"] = overlay_file
            result["saved_heatmap_path"] = heatmap_file

        return result
    finally:
        grad_cam.remove_hooks()
