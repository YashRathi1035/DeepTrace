# DeepTrace — Reliability-Aware Deepfake Forensics and Evidence Analyzer

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeepTrace is a digital media forensics platform and research system designed to inspect images and videos for facial manipulation while estimating prediction reliability, uncertainty, and evidential integrity.

> **Important Forensic Notice**:
> DeepTrace is a scientific decision-support tool, not a definitive authenticity verifier. A model prediction is not absolute proof of manipulation. Saliency maps (Grad-CAM) explain internal model activations rather than verified manipulation masks, and metadata anomalies or absent EXIF tags do not constitute conclusive evidence of tampering.

---

## 1. Project Overview & Research Motivation

### The Problem
Most contemporary deepfake detection models output overconfident binary predictions (`REAL` vs. `FAKE`), yet degrade significantly when confronted with:
1. Real-world social media compression and optical blur.
2. Cross-generator domain shift (e.g. tested on algorithms unseen during training).
3. Degraded or low-resolution facial crops.

### Core Research Direction
DeepTrace investigates whether combining **confidence calibration** (temperature scaling, Expected Calibration Error minimization), **input-quality verification**, **selective abstention** (flagging inconclusive inputs rather than forcing a guess), and **temporal consistency analysis** can make deepfake forensics substantially more reliable and robust in practice.

---

## 2. System Architecture

```text
DeepTrace/
├── configs/
│   ├── config.yaml               # Central pipeline & hyperparameter configuration
│   └── __init__.py               # Config loader, logger, and reproducibility seeds
├── models/
│   ├── detector.py               # EfficientNet-B0 & ResNet18 classifier architectures
│   ├── train.py                  # Training pipeline with AMP & early stopping
│   ├── predict.py                # Reusable inference engine
│   └── checkpoint.py             # Checkpoint serialization & validation
├── preprocessing/
│   ├── face_detection.py         # MediaPipe BlazeFace & fallback detectors with context margin
│   ├── image_processor.py        # Normalization, quality scoring (blur/brightness/resolution)
│   ├── video_processor.py        # Frame streaming & temporal sampling at target FPS
│   └── dataset.py                # PyTorch dataset & leakage-free split partitioning
├── explainability/
│   └── gradcam.py                # Grad-CAM visual saliency heatmaps
├── forensic/
│   ├── metadata.py               # SHA-256 cryptographic hashing & container inspection
│   ├── aggregation.py            # Video-level frame score aggregation policies
│   ├── calibration.py            # Temperature scaling & ECE calculation
│   ├── quality.py                # Quality assessment & thresholding
│   └── report.py                 # Structured PDF and JSON report generator
├── evaluation/
│   ├── metrics.py                # Accuracy, ROC-AUC, PR-AUC, Brier score, ECE
│   ├── robustness.py             # Perturbation evaluations (JPEG, blur, resize)
│   └── plots.py                  # Reliability diagrams & metric curves
├── ui/
│   ├── components.py             # Reusable UI widgets & evidence cards
│   └── pages.py                  # Dashboard multi-page views
├── scripts/
│   ├── prepare_dataset.py        # Face extraction, fixture generation, manifest builder
│   ├── validate_dataset.py       # Data integrity & cross-split leakage auditor
│   └── evaluate_model.py         # Evaluation benchmark runner
├── tests/                        # Automated pytest test suites
└── data/ & artifacts/            # Local data manifests, weights, logs, and checkpoints
```

---

## 3. Technology Stack

- **Core & Runtime**: Python 3.11, PyTorch 2.5.1 + CUDA 12.1 (NVIDIA RTX 4050 Laptop GPU support, with automatic CPU fallback).
- **Computer Vision**: OpenCV (video decoding, quality assessment), Google MediaPipe (facial landmark detection).
- **Explainability**: Grad-CAM (visual feature attribution).
- **Interface**: Streamlit (forensic evidence dashboard).
- **Reporting**: ReportLab (tamper-evident PDF reports), Python `hashlib` (SHA-256 integrity hashes).
- **Testing**: `pytest`.

---

## 4. Setup and Installation

### 1. Clone the repository
```powershell
git clone https://github.com/your-username/DeepTrace.git
cd DeepTrace
```

### 2. Create and activate a Python 3.11 virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

---

## 5. Quickstart & Verification Commands

### Step 1: Run automated tests
```powershell
python -m pytest tests/ -v
```

### Step 2: Generate a test dataset fixture and manifests
```powershell
python scripts/prepare_dataset.py --create-fixture
```

### Step 3: Audit dataset integrity and verify zero data leakage
```powershell
python scripts/validate_dataset.py --config configs/config.yaml
```

---

## 6. Dataset Acquisition

### FaceForensics++ (Primary Training/Validation)
- **Repository**: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- Submit the official access request form.
- Place downloaded video sequences into `data/raw/FaceForensics/`.
- Run `python scripts/prepare_dataset.py` to extract face crops and build manifests.

### Celeb-DF v2 (External Generalization Evaluation)
- **Repository**: [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- Place downloaded external evaluation videos into `data/raw/Celeb-DF-v2/`.

---

## 7. Current Project Status
- [x] **Phase 1**: Project Setup, Architecture, Dataset Pipeline, and Baseline Preparation
- [ ] **Phase 2**: Deepfake Detection Model and Training Pipeline
- [ ] **Phase 3**: Video Forensics, Frame Analysis, and Temporal Evidence
- [ ] **Phase 4**: Explainable AI and Forensic Evidence Collection
- [ ] **Phase 5**: Reliability, Calibration, and Research Experiments
- [ ] **Phase 6**: Streamlit Application and Forensic Report Generation
- [ ] **Phase 7**: Integration, Testing, Optimization, Documentation, and Release
