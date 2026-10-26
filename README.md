# DeepTrace — Reliability-Aware Deepfake Forensics and Evidence Analyzer

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen.svg)](#)
[![Security Hardened](https://img.shields.io/badge/security-hardened-blue.svg)](#)
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
│   ├── security.py               # Whitelist validation, magic bytes, path traversal defense
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
│   ├── evaluate_model.py         # Cross-dataset & reliability benchmark runner
│   ├── benchmark.py              # Performance profiling & latency benchmark suite
│   └── robustness_demo.py        # Real-world degradation stress-test demo
├── tests/                        # 53-test automated pytest suite (100% pass rate)
│   ├── test_preprocessing.py     # SHA-256, metadata, face crop, dataset leakage
│   ├── test_detector.py          # Architecture shapes, checkpoint I/O, inference
│   ├── test_aggregation.py       # IoU tracking, aggregation strategies, temporal analysis
│   ├── test_calibration.py       # ECE, temperature scaling, abstention policy
│   ├── test_gradcam.py           # Heatmap generation, overlay blending, evidence records
│   ├── test_video_processing.py  # Frame sampling, metadata, corrupted video handling
│   ├── test_reports.py           # JSON export, PDF generation
│   ├── test_security.py          # Input validation, path traversal, magic bytes, caps
│   └── test_integration.py       # End-to-end pipeline integration (10 tests)
├── app.py                        # Streamlit forensic dashboard entry point
├── CONTRIBUTING.md               # Contributor and developer guide
├── TECHNICAL_DECISIONS.md        # Architecture and engineering decision records
└── data/ & artifacts/            # Local data manifests, weights, logs, and checkpoints
```

---

## 3. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Core & Runtime | Python 3.11, PyTorch 2.5.1 + CUDA 12.1 | Accelerated mixed-precision training and sub-10ms inference |
| Computer Vision | OpenCV, Google MediaPipe BlazeFace | Lightweight sub-3ms face detection with 20% margin expansion |
| Backbone | EfficientNet-B0 (~5.3M params), ResNet-18 | High parameter efficiency, sensitive to boundary artifacts |
| Explainability | Grad-CAM (features.8 layer) | Visual attribution highlighting influential facial regions |
| Calibration | Temperature Scaling (NLL / L-BFGS) | ECE reduction by >66% without modifying discriminative accuracy |
| Decision Policy | Selective Forensic Abstention | Prevents forced guesses on low-quality/ambiguous inputs |
| Interface | Streamlit (Dark Charcoal Theme) | Evidence-first UI with pulsing badges and metric cards |
| Reporting | ReportLab Platypus + SHA-256 | Tamper-evident PDF export and JSON dossiers |
| Security | Multi-layer input validation | Whitelists, magic bytes, size caps, path traversal blocking |
| Testing | pytest (53 automated tests) | 100% passing unit, integration, and security tests |

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

## 5. Quickstart & Verification

### Step 1: Run the full automated test suite
```powershell
python -m pytest tests/ -v --tb=short
# Expected: 53 passed in ~11s
```

### Step 2: Run performance profiling and latency benchmark
```powershell
python scripts/benchmark.py --iterations 30
# Measures neural forward pass, face detection, end-to-end inference, and video throughput
```

### Step 3: Run model robustness and degradation stress-test
```powershell
python scripts/robustness_demo.py --n-samples 20
# Evaluates accuracy, ECE, and selective abstention under JPEG, blur, resize, and noise
```

### Step 4: Audit dataset integrity (zero data leakage check)
```powershell
python scripts/prepare_dataset.py --create-fixture
python scripts/validate_dataset.py --config configs/config.yaml
```

### Step 5: Launch the Streamlit forensic dashboard
```powershell
streamlit run app.py
```

---

## 6. Performance & Benchmark Metrics

Evaluated on an **NVIDIA GeForce RTX 4050 Laptop GPU** (6 GB VRAM) with CUDA 12.1:

| Metric | Measured Value | Throughput |
|---|---|---|
| Pure Neural Forward Pass (EfficientNet-B0) | **9.27 ms** | 107.8 FPS |
| Facial Extraction & Preprocessing | **2.88 ms** | 347.6 FPS |
| End-to-End Image Inference (Full Chain) | **13.03 ms** | 76.8 FPS |
| Video Streaming Analysis (1 FPS sampling) | **20.11 ms/frame** | 49.7 FPS |
| Peak GPU VRAM Allocation | **49.7 MB** | Minimal footprint |
| Host System RAM (RSS Peak) | **877.5 MB** | Memory-safe |

---

## 7. Security Architecture & Input Hardening

In forensic deployments, user-uploaded media files can pose security risks. DeepTrace enforces defensive controls:
- **Extension Whitelist**: Permits only verified media extensions (`.jpg`, `.jpeg`, `.png`, `.webp`, `.mp4`, `.avi`, `.mov`, `.mkv`). Executable scripts (`.exe`, `.py`, `.sh`, `.bat`) are immediately rejected.
- **Magic Byte Inspection**: Verifies binary header signatures (JPEG `\xff\xd8\xff`, PNG `\x89PNG`, WebP `RIFF...WEBP`, MP4 `ftyp`, MKV `\x1a\x45\xdf\xa3`). Masqueraded executables disguised with image extensions are blocked.
- **Strict Size Limits**: 25 MB cap for images and 250 MB cap for video containers to mitigate Denial-of-Service (DoS).
- **Path Traversal Neutralization**: All filenames are stripped of null bytes (`\x00`) and relative breakout paths (`../`, `..\\`). Resolved paths are verified strictly within approved directories.
- **Secure File Descriptors**: Employs `tempfile.mkstemp` with restricted OS permissions to prevent symlink attacks and race conditions.

---

## 8. Forensic Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home & Overview | Architectural foundation, 7-step forensic pipeline diagram, capability matrix |
| 🖼️ Image Analysis | Secure file upload → SHA-256 → calibrated inference → Grad-CAM → PDF/JSON export |
| 🎥 Video Forensics | Frame sampling → temporal consistency → suspicious interval alerts → report |
| 📑 Forensic Dossier | Archived evidence records browser (JSON inspection + PDF dossier download) |
| 📊 Evaluation Dashboard | Benchmark tables, reliability diagrams, robustness curves, degradation plots |

---

## 9. Contributing & Engineering Decisions

- Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines, testing procedures, and architectural conventions.
- Refer to [`TECHNICAL_DECISIONS.md`](TECHNICAL_DECISIONS.md) for in-depth records on model backbones, calibration methodology, aggregation mathematics, and security hardening.

---

## 10. Project Status

- [x] **Phase 1**: Project Setup, Architecture, Dataset Pipeline, and Baseline Preparation
- [x] **Phase 2**: Deepfake Detection Model and Training Pipeline
- [x] **Phase 3**: Video Forensics, Frame Analysis, and Temporal Evidence
- [x] **Phase 4**: Explainable AI and Forensic Evidence Collection
- [x] **Phase 5**: Reliability, Calibration, and Research Experiments
- [x] **Phase 6**: Streamlit Application and Forensic Report Generation
- [x] **Phase 7**: Integration, Testing, Optimization, Documentation, and Release
