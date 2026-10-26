"""Dataset validation CLI script to audit data integrity, class balance, and prevent data leakage."""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import pandas as pd
from configs import load_config, get_logger
from forensic.metadata import inspect_media_file


def validate_dataset(config_path: str = "configs/config.yaml") -> bool:
    """Audit dataset directory or manifest for file existence, readability, and split isolation."""
    config = load_config(config_path)
    logger = get_logger("DeepTrace.ValidateDataset", log_file=os.path.join(config["paths"]["logs_dir"], "validation.log"))
    logger.info("Starting DeepTrace Dataset Audit...")

    manifests_dir = config["paths"]["manifests_dir"]
    manifest_path = os.path.join(manifests_dir, "dataset_manifest.json")
    csv_manifest_path = os.path.join(manifests_dir, "dataset_manifest.csv")

    target_manifest = None
    if os.path.exists(manifest_path):
        target_manifest = manifest_path
    elif os.path.exists(csv_manifest_path):
        target_manifest = csv_manifest_path

    if not target_manifest:
        logger.warning(f"No manifest found at {manifest_path} or {csv_manifest_path}.")
        raw_dir = config["paths"]["raw_data_dir"]
        logger.info(f"Checking raw dataset folder: {raw_dir}")
        if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
            logger.warning(f"Raw data directory {raw_dir} is empty.")
            logger.info("Tip: Run `python scripts/prepare_dataset.py --create-fixture` to generate a software validation fixture.")
            return False
        return True

    # Validate manifest contents
    logger.info(f"Validating manifest: {target_manifest}")
    if target_manifest.endswith(".json"):
        with open(target_manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        df = pd.DataFrame(samples)
    else:
        df = pd.read_csv(target_manifest)

    total_samples = len(df)
    logger.info(f"Total samples recorded in manifest: {total_samples}")

    if total_samples == 0:
        logger.error("Manifest contains zero samples.")
        return False

    required_cols = {"image_path", "label", "split", "video_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Manifest missing required columns: {missing_cols}")
        return False

    # 1. File existence and integrity check
    missing_files = 0
    corrupted_files = 0
    for idx, row in df.iterrows():
        p = row["image_path"]
        if not os.path.exists(p):
            missing_files += 1
            if missing_files <= 5:
                logger.warning(f"Missing file: {p}")
        else:
            if os.path.getsize(p) == 0:
                corrupted_files += 1

    if missing_files > 0:
        logger.error(f"{missing_files}/{total_samples} sample files could not be found on disk.")
    if corrupted_files > 0:
        logger.error(f"{corrupted_files}/{total_samples} files have 0 byte size.")

    # 2. Class balance
    label_counts = df["label"].value_counts().to_dict()
    logger.info(f"Class distribution: {label_counts} (0: Real, 1: Manipulated)")

    # 3. Split distribution
    split_counts = df["split"].value_counts().to_dict()
    logger.info(f"Split distribution: {split_counts}")

    # 4. Data leakage verification (video_id must never appear in both train and (val or test))
    train_vids = set(df[df["split"] == "train"]["video_id"].unique())
    val_vids = set(df[df["split"] == "val"]["video_id"].unique())
    test_vids = set(df[df["split"] == "test"]["video_id"].unique())

    train_val_leak = train_vids.intersection(val_vids)
    train_test_leak = train_vids.intersection(test_vids)
    val_test_leak = val_vids.intersection(test_vids)

    if train_val_leak:
        logger.error(f"DATA LEAKAGE DETECTED! {len(train_val_leak)} videos overlap between Train and Val: {train_val_leak}")
    if train_test_leak:
        logger.error(f"DATA LEAKAGE DETECTED! {len(train_test_leak)} videos overlap between Train and Test: {train_test_leak}")
    if val_test_leak:
        logger.error(f"DATA LEAKAGE DETECTED! {len(val_test_leak)} videos overlap between Val and Test: {val_test_leak}")

    has_leakage = bool(train_val_leak or train_test_leak or val_test_leak)
    is_valid = (missing_files == 0) and (corrupted_files == 0) and not has_leakage

    if is_valid:
        logger.info("Dataset validation passed! Manifest and data are consistent and leakage-free.")
    else:
        logger.warning("Dataset validation completed with issues flagged above.")

    return is_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTrace Dataset Validation Tool")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    success = validate_dataset(args.config)
    exit(0 if success else 1)
