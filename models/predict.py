"""Inference engine providing structured predictions, probability estimation, and latency tracking."""

import time
from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F

from models.checkpoint import load_checkpoint
from preprocessing.image_processor import ImageProcessor


class DeepfakeDetector:
    """Production-grade forensic inference engine for image and video frame analysis."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        """Initialize detector from saved checkpoint.
        
        Args:
            checkpoint_path: Path to serialized model checkpoint (.pt).
            device: 'cuda' or 'cpu'. Defaults to CUDA if available.
            image_processor: Preprocessing pipeline instance.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model, self.checkpoint_meta = load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
            eval_mode=True
        )

        self.architecture = self.checkpoint_meta.get("architecture", "efficientnet_b0")
        self.labels = self.checkpoint_meta.get("labels", ["Real", "Manipulated"])
        self.image_processor = image_processor or ImageProcessor()

    def predict_batch(self, batch_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """Run batch inference on pre-normalized PyTorch tensors of shape (B, 3, H, W).
        
        Args:
            batch_tensor: Tensor of normalized image crops on CPU or target device.
            
        Returns:
            List of structured prediction dictionaries.
        """
        batch_tensor = batch_tensor.to(self.device)
        start_t = time.perf_counter()

        with torch.no_grad():
            logits = self.model(batch_tensor)  # (B, 2)
            probs = F.softmax(logits, dim=1)    # (B, 2)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        per_item_ms = round(elapsed_ms / max(1, batch_tensor.size(0)), 2)

        logits_np = logits.cpu().numpy()
        probs_np = probs.cpu().numpy()

        results = []
        for i in range(len(logits_np)):
            raw_scores = [round(float(v), 4) for v in logits_np[i]]
            pred_idx = int(np.argmax(probs_np[i]))
            manipulation_prob = float(probs_np[i][1])  # Class 1 is Manipulated

            results.append({
                "predicted_class": self.labels[pred_idx],
                "predicted_label_index": pred_idx,
                "raw_scores": raw_scores,
                "probabilities": [round(float(p), 4) for p in probs_np[i]],
                "manipulation_score": round(manipulation_prob, 4),
                "confidence": round(float(probs_np[i][pred_idx]), 4),
                "model_identifier": self.architecture,
                "inference_time_ms": per_item_ms,
                "is_calibrated": False,
                "warnings": []
            })

        return results

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Process and classify a single OpenCV BGR video frame.
        
        Args:
            frame_bgr: BGR video frame numpy array.
            
        Returns:
            Prediction dictionary with quality diagnostics and original face crop.
        """
        proc_result = self.image_processor.process_face(frame_bgr)
        tensor = proc_result["tensor"]

        pred = self.predict_batch(tensor)[0]
        pred["preprocessing_info"] = proc_result["quality"]
        pred["original_bbox"] = proc_result["original_bbox"]
        pred["detection_confidence"] = proc_result["detection_confidence"]
        pred["num_faces_detected"] = proc_result["num_faces_detected"]
        pred["crop_bgr"] = proc_result["crop_bgr"]
        pred["crop_rgb"] = proc_result["crop_rgb"]
        pred["warnings"].extend(proc_result["quality"].get("warnings", []))

        return pred

    def predict_image(self, image_input: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """Convenience method to predict authenticity of an image file path, bytes, or array."""
        image_bgr = self.image_processor.load_image(image_input)
        return self.predict_frame(image_bgr)
