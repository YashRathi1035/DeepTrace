"""Configuration loader, reproducibility utilities, and structured logger for DeepTrace."""

import os
import random
import logging
from typing import Any, Dict, Optional
import yaml
import numpy as np
import torch


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to config.yaml. Defaults to configs/config.yaml relative
                     to workspace root.
                     
    Returns:
        Nested dictionary of configuration options.
    """
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "DeepTrace", log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Configure and return a structured logger with both console and optional file handlers.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to log file.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if requested
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
