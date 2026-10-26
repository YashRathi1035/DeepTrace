"""Dataset loading, manifest management, leakage-free splits, and forensic data loaders."""

import os
import json
import random
from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class GaussianBlurTransform:
    """Simulate optical or motion blur augmentation."""
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 1.5)):
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.3:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


class JpegCompressionTransform:
    """Simulate social media and web compression artifacts."""
    def __init__(self, quality_range: Tuple[int, int] = (60, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.35:
            import io
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


def get_training_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Augmentation pipeline designed for deepfake detection.
    
    Includes spatial, color, compression, and blur perturbations while avoiding
    artificial shortcuts that destroy subtle manipulation boundary features.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        GaussianBlurTransform(radius_range=(0.5, 1.2)),
        JpegCompressionTransform(quality_range=(65, 95)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_validation_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Deterministic validation and inference transform."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class DeepfakeFaceDataset(Dataset):
    """PyTorch Dataset loading preprocessed face crops governed by a manifest."""

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """Initialize dataset from manifest file (CSV or JSON).
        
        Args:
            manifest_path: Path to manifest.json or manifest.csv.
            split: Target split ('train', 'val', or 'test').
            transform: Custom PyTorch transform. If None, default split transform is chosen.
            image_size: Target image dimensions.
        """
        self.manifest_path = manifest_path
        self.split = split
        self.image_size = tuple(image_size)

        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        if manifest_path.endswith(".json"):
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data.get("samples", []))
        else:
            df = pd.read_csv(manifest_path)

        if "split" in df.columns:
            self.samples_df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.samples_df = df.reset_index(drop=True)

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_training_transforms(self.image_size)
        else:
            self.transform = get_validation_transforms(self.image_size)

    def __len__(self) -> int:
        return len(self.samples_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.samples_df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label"])  # 0 = Real, 1 = Manipulated

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Sample crop not found at: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)

        metadata = {
            "image_path": image_path,
            "video_id": row.get("video_id", "unknown"),
            "subject_id": row.get("subject_id", "unknown"),
            "manipulation_type": row.get("manipulation_type", "unknown")
        }

        return tensor, label, metadata

    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for class imbalance handling."""
        labels = self.samples_df["label"].values
        counts = np.bincount(labels, minlength=2)
        total = len(labels)
        weights = [total / (2.0 * max(c, 1)) for c in counts]
        return torch.tensor(weights, dtype=torch.float32)


def create_split_manifest(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Assign train/val/test splits at the video/subject level to prevent data leakage.
    
    Frames from the exact same video or subject must NEVER cross split boundaries.
    
    Args:
        samples: List of sample dictionaries containing 'video_id', 'subject_id', etc.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        seed: Random seed for reproducible splitting.
        
    Returns:
        List of sample dicts with updated 'split' field.
    """
    rng = random.Random(seed)

    # Group sample indices by video_id
    video_to_indices: Dict[str, List[int]] = {}
    for idx, s in enumerate(samples):
        vid = s.get("video_id") or s.get("subject_id") or f"item_{idx}"
        video_to_indices.setdefault(vid, []).append(idx)

    unique_videos = list(video_to_indices.keys())
    rng.shuffle(unique_videos)

    total_videos = len(unique_videos)
    n_test = int(round(total_videos * test_ratio))
    n_val = int(round(total_videos * val_ratio))

    test_vids = set(unique_videos[:n_test])
    val_vids = set(unique_videos[n_test:n_test + n_val])
    train_vids = set(unique_videos[n_test + n_val:])

    results = []
    for s in samples:
        vid = s.get("video_id") or s.get("subject_id") or "item"
        s_copy = dict(s)
        if vid in test_vids:
            s_copy["split"] = "test"
        elif vid in val_vids:
            s_copy["split"] = "val"
        else:
            s_copy["split"] = "train"
        results.append(s_copy)

    return results
