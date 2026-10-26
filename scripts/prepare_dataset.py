"""Dataset preparation CLI: extracts face crops, generates leakage-free splits, and exports manifests."""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import cv2
import pandas as pd
from configs import load_config, get_logger, set_seed
from preprocessing.image_processor import ImageProcessor
from preprocessing.video_processor import VideoProcessor
from preprocessing.dataset import create_split_manifest


def generate_synthetic_fixture(config: dict, logger) -> None:
    """Generate a small, clearly labeled synthetic fixture dataset for software testing.
    
    Creates 10 test samples (5 Real, 5 Manipulated) with synthetic facial patterns
    and distinct video IDs to validate the preprocessing, pipeline, and splitting logic.
    """
    raw_dir = config["paths"]["raw_data_dir"]
    fixture_dir = os.path.join(raw_dir, "synthetic_fixture")
    os.makedirs(os.path.join(fixture_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(fixture_dir, "manipulated"), exist_ok=True)

    logger.info(f"Generating synthetic fixture in: {fixture_dir}")

    for i in range(5):
        # Create a synthetic image with a face-like oval and facial features
        img_real = np.full((320, 320, 3), 40, dtype=np.uint8)
        # Face ellipse
        cv2.ellipse(img_real, (160, 160), (70, 95), 0, 0, 360, (190, 210, 230), -1)
        # Eyes
        cv2.circle(img_real, (135, 140), 12, (50, 50, 50), -1)
        cv2.circle(img_real, (185, 140), 12, (50, 50, 50), -1)
        # Nose
        cv2.line(img_real, (160, 145), (160, 175), (120, 130, 150), 3)
        # Mouth
        cv2.ellipse(img_real, (160, 200), (25, 10), 0, 0, 180, (70, 70, 160), 3)

        path_real = os.path.join(fixture_dir, "real", f"real_vid_{i:02d}_frame0.png")
        cv2.imwrite(path_real, img_real)

        # Manipulated version has blending boundary artifact around the facial region
        img_fake = img_real.copy()
        # Insert artificial boundary discontinuity / patch
        cv2.rectangle(img_fake, (120, 125), (200, 215), (180, 170, 210), 2)
        # Subtle color distortion in face center
        img_fake[130:210, 125:195, 0] = np.clip(img_fake[130:210, 125:195, 0].astype(int) + 30, 0, 255)
        path_fake = os.path.join(fixture_dir, "manipulated", f"fake_vid_{i:02d}_frame0.png")
        cv2.imwrite(path_fake, img_fake)

    logger.info("Synthetic fixture generated (5 real, 5 manipulated samples).")


def prepare_dataset(config_path: str = "configs/config.yaml", create_fixture: bool = False) -> None:
    """Preprocess media, extract faces, and build manifests with leakage-free splits."""
    config = load_config(config_path)
    logger = get_logger("DeepTrace.PrepareDataset")
    set_seed(config["system"]["seed"])

    if create_fixture:
        generate_synthetic_fixture(config, logger)

    raw_dir = config["paths"]["raw_data_dir"]
    processed_dir = config["paths"]["processed_data_dir"]
    manifests_dir = config["paths"]["manifests_dir"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    image_proc = ImageProcessor(
        image_size=config["preprocessing"]["image_size"],
        mean=config["preprocessing"]["normalize_mean"],
        std=config["preprocessing"]["normalize_std"],
        face_margin=config["preprocessing"]["face_margin"]
    )
    video_proc = VideoProcessor(
        image_processor=image_proc,
        sampling_fps=config["video"]["sampling_fps"],
        max_frames=config["video"]["max_frames"]
    )

    logger.info(f"Scanning raw media directory: {raw_dir}")
    samples_list = []

    # Traverse directory
    valid_exts = {".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov"}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in valid_exts:
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, raw_dir).lower()

            # Infer label from path
            if "real" in rel_path or "original" in rel_path or "actors" in rel_path:
                label = 0
                manipulation_type = "authentic"
            elif "fake" in rel_path or "manipulated" in rel_path or "deepfakes" in rel_path:
                label = 1
                manipulation_type = "synthetic_or_deepfake"
            else:
                label = 0  # Default assumption
                manipulation_type = "unspecified"

            # Derive unique video/subject ID from filename stem
            stem = os.path.splitext(f)[0]
            video_id = stem.split("_frame")[0] if "_frame" in stem else stem

            if ext in {".mp4", ".avi", ".mov"}:
                out_crops_dir = os.path.join(processed_dir, f"{video_id}_crops")
                res = video_proc.process_video(full_path, output_dir=out_crops_dir)
                for frame_rec in res["frames"]:
                    if "saved_crop_path" in frame_rec:
                        samples_list.append({
                            "image_path": frame_rec["saved_crop_path"],
                            "video_id": video_id,
                            "subject_id": video_id,
                            "label": label,
                            "manipulation_type": manipulation_type,
                            "timestamp": frame_rec["timestamp"],
                            "frame_index": frame_rec["frame_index"],
                            "source_media": full_path
                        })
            else:
                # Static image
                try:
                    res = image_proc.process_face(full_path)
                    crop_filename = f"{stem}_crop.jpg"
                    out_path = os.path.join(processed_dir, crop_filename)
                    cv2.imwrite(out_path, res["crop_bgr"])
                    samples_list.append({
                        "image_path": out_path,
                        "video_id": video_id,
                        "subject_id": video_id,
                        "label": label,
                        "manipulation_type": manipulation_type,
                        "timestamp": 0.0,
                        "frame_index": 0,
                        "source_media": full_path
                    })
                except Exception as e:
                    logger.warning(f"Could not extract face from {full_path}: {e}")

    logger.info(f"Extracted {len(samples_list)} face crop samples.")
    if len(samples_list) == 0:
        logger.warning("No samples found. If datasets are not downloaded, run with --create-fixture.")
        return

    # Split assignment with zero data leakage
    val_ratio = config["training"].get("val_split", 0.15)
    test_ratio = config["training"].get("test_split", 0.15)
    split_samples = create_split_manifest(
        samples_list,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=config["system"]["seed"]
    )

    # Save Manifests
    manifest_json = os.path.join(manifests_dir, "dataset_manifest.json")
    manifest_csv = os.path.join(manifests_dir, "dataset_manifest.csv")

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0",
            "seed": config["system"]["seed"],
            "total_samples": len(split_samples),
            "samples": split_samples
        }, f, indent=2)

    df = pd.DataFrame(split_samples)
    df.to_csv(manifest_csv, index=False)
    logger.info(f"Manifest saved to {manifest_json} and {manifest_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTrace Dataset Preparation Tool")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--create-fixture", action="store_true", help="Generate synthetic fixture dataset for testing")
    args = parser.parse_args()
    prepare_dataset(args.config, create_fixture=args.create_fixture)
