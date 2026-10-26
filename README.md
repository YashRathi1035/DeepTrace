# DeepTrace — Reliability-Aware Deepfake Forensics and Evidence Analyzer

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeepTrace is a research-grade digital media forensics platform designed to inspect images and videos for facial manipulation while estimating prediction reliability, uncertainty, and evidential integrity.

> **Important Forensic Notice**:
> DeepTrace is a scientific decision-support tool, not a definitive authenticity verifier. Model predictions reflect probabilistic patterns in neural feature representations. Saliency maps (Grad-CAM) explain internal model activations rather than verified manipulation masks. Missing EXIF tags or container re-compression are normal in web-distributed media and do not constitute conclusive evidence of tampering.

---

## 1. Project Overview & Research Motivation

### The Problem
Most contemporary deepfake detection models output overconfident binary predictions (`REAL` vs. `FAKE`), yet degrade significantly when confronted with:
1. Real-world social media compression and optical blur.
2. Cross-generator domain shift (tested on manipulation algorithms unseen during training).
3. Degraded or low-resolution facial crops.

### Core Research Direction
DeepTrace investigates whether combining **confidence calibration** (temperature scaling, ECE minimization), **input-quality verification**, **selective abstention** (flagging inconclusive inputs rather than forcing a guess), and **temporal consistency analysis** can make deepfake forensics substantially more reliable in practice.

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
│   ├── predict.py                # Reusable inference engine with batch & frame APIs
│   └── checkpoint.py             # Checkpoint serialization & validation
├── preprocessing/
│   ├── face_detection.py         # MediaPipe BlazeFace & fallback detectors
│   ├── image_processor.py        # Normalization, quality scoring, tensor preparation
│   ├── video_processor.py        # Frame streaming & temporal sampling at target FPS
│   └── dataset.py                # PyTorch dataset & leakage-free split partitioning
├── explainability/
│   └── gradcam.py                # Grad-CAM visual saliency heatmaps
├── forensic/
│   ├── metadata.py               # SHA-256 cryptographic hashing & container inspection
│   ├── aggregation.py            # Video-level frame score aggregation policies
│   ├── calibration.py            # Temperature scaling, ECE calculation & abstention policy
│   ├── quality.py                # Composite quality assessment & thresholding
│   └── report.py                 # Tamper-evident PDF and JSON forensic report generator
├── evaluation/
│   ├── metrics.py                # Accuracy, ROC-AUC, PR-AUC, Brier score, ECE
│   ├── robustness.py             # Perturbation evaluations (JPEG, blur, resize, noise)
│   ├── cross_dataset.py          # Cross-dataset generalization evaluation
│   └── plots.py                  # Reliability diagrams & metric curves
├── ui/
│   ├── components.py             # Premium reusable UI widgets & evidence cards
│   └── pages.py                  # Multi-page Streamlit dashboard views
├── scripts/
│   ├── prepare_dataset.py        # Face extraction, fixture generation, manifest builder
│   ├── validate_dataset.py       # Data integrity & cross-split leakage auditor
│   └── evaluate_model.py         # Evaluation benchmark runner
├── tests/                        # 40-test automated pytest suite (100% pass rate)
│   ├── test_preprocessing.py     # SHA-256, metadata, face crop, dataset leakage
│   ├── test_detector.py          # Architecture shapes, checkpoint I/O, inference
│   ├── test_aggregation.py       # IoU tracking, aggregation strategies, temporal analysis
│   ├── test_calibration.py       # ECE, temperature scaling, abstention policy
│   ├── test_gradcam.py           # Heatmap generation, overlay blending, evidence records
│   ├── test_video_processing.py  # Frame sampling, metadata, corrupted video handling
│   ├── test_reports.py           # JSON export, PDF generation
│   └── test_integration.py       # End-to-end pipeline integration (10 tests)
├── app.py                        # Streamlit forensic dashboard entry point
└── data/ & artifacts/            # Local data manifests, weights, logs, and checkpoints
```

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Core & Runtime | Python 3.11, PyTorch 2.5.1 + CUDA 12.1 |
| Computer Vision | OpenCV, Google MediaPipe BlazeFace |
| Backbone | EfficientNet-B0, ResNet-18 (torchvision) |
| Explainability | Grad-CAM (class-discriminative saliency) |
| Calibration | Temperature Scaling (post-hoc, validation-tuned) |
| Interface | Streamlit (premium dark forensic dashboard) |
| Reporting | ReportLab (multi-section PDF), hashlib SHA-256 |
| Testing | pytest (40 tests, 100% pass rate) |

---

## 4. Setup and Installation

### 1. Clone the repository
```powershell
git clone https://github.com/YashRathi1035/DeepTrace.git
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

## 5. Quickstart

### Step 1: Run the full automated test suite
```powershell
python -m pytest tests/ -v
# Expected: 40 passed
```

### Step 2: Generate a test dataset fixture and manifests
```powershell
python scripts/prepare_dataset.py --create-fixture
```

### Step 3: Audit dataset integrity (zero data leakage check)
```powershell
python scripts/validate_dataset.py --config configs/config.yaml
```

### Step 4: Launch the Streamlit forensic dashboard
```powershell
streamlit run app.py
```

---

## 6. Forensic Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home & Overview | Architecture overview, pipeline visualization, capability list |
| 🖼️ Image Analysis | File upload → SHA-256 → calibrated inference → Grad-CAM → PDF/JSON export |
| 🎥 Video Forensics | Frame sampling → temporal analysis → suspicious interval localization → report |
| 📑 Forensic Dossier | Archived evidence records browser (JSON viewer + PDF download) |
| 📊 Evaluation Dashboard | Ablation tables, reliability diagrams, robustness curves, confusion matrices |

---

## 7. Dataset Acquisition

### FaceForensics++ (Primary Training/Validation)
- **Repository**: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- Submit the official access request form.
- Place downloaded video sequences into `data/raw/FaceForensics/`.
- Run `python scripts/prepare_dataset.py` to extract face crops and build manifests.

### Celeb-DF v2 (External Generalization Evaluation)
- **Repository**: [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- Place downloaded evaluation videos into `data/raw/Celeb-DF-v2/`.

---

## 8. Project Status

- [x] **Phase 1**: Project Setup, Architecture, Dataset Pipeline, and Baseline Preparation
- [x] **Phase 2**: Deepfake Detection Model and Training Pipeline
- [x] **Phase 3**: Video Forensics, Frame Analysis, and Temporal Evidence
- [x] **Phase 4**: Explainable AI and Forensic Evidence Collection
- [x] **Phase 5**: Reliability, Calibration, and Research Experiments
- [x] **Phase 6**: Streamlit Application and Forensic Report Generation
- [ ] **Phase 7**: Integration, Testing, Optimization, Documentation, and Release
