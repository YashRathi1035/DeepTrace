# Contributing to DeepTrace

Thank you for your interest in contributing to **DeepTrace** — a reliability-aware digital media forensics and deepfake evidence analyzer. We welcome contributions from computer vision researchers, digital forensics engineers, machine learning developers, and open-source contributors.

---

## Code of Conduct & Forensic Integrity

In forensic applications, computational transparency, reproducibility, and cryptographic integrity are paramount.
1. **No False Confidence**: Never design heuristics or thresholds that force binary decisions on ambiguous media without calibration or uncertainty metrics.
2. **Defensive Programming**: All external media inputs must pass strict security validation (extension filtering, magic bytes inspection, size caps, and path traversal mitigation).
3. **Reproducibility**: All experiments, training splits, and evaluations must enforce fixed random seeds and prevent data leakage across subjects/videos.

---

## Development Environment Setup

### Prerequisites
- **Python**: 3.10+ (tested on Python 3.11.6)
- **PyTorch**: 2.0+ (CUDA recommended for training and high-throughput inference)
- **Operating Systems**: Windows 11, Linux (Ubuntu 20.04+), macOS (Apple Silicon via MPS)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/DeepTrace.git
   cd DeepTrace
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -m pytest tests/ -v --tb=short
   ```
   All 53 unit and integration tests should pass cleanly without needing external dataset downloads.

---

## Project Structure & Architecture

```
DeepTrace/
├── configs/               # YAML configuration files (models, calibration, video, quality)
├── data/                  # Local datasets, manifests, and synthetic fixtures
├── preprocessing/         # MediaPipe BlazeFace detection, alignment, context margin cropping
├── models/                # EfficientNet-B0 & ResNet-18 architectures, training engine, predictor
├── explainability/        # Grad-CAM attribution, heatmap generation, overlay composition
├── forensic/              # Security hardening, calibration, selective abstention, aggregation, reports
├── evaluation/            # Cross-dataset generalization, robustness curves, metric computation
├── ui/                    # Streamlit components and 5-page forensic analyst dashboard
├── scripts/               # CLI tools for training, evaluation, benchmarking, and robustness testing
├── tests/                 # Full unit, integration, and security test suites
└── artifacts/             # Generated checkpoints, plots, dossiers, and PDF reports
```

---

## Adding New Features

### 1. Adding a New Neural Backbone Detector
1. Subclass `nn.Module` in [`models/detector.py`](file:///models/detector.py).
2. Implement:
   - `forward(self, x: torch.Tensor) -> torch.Tensor`: Outputs raw binary logits of shape `(B, 2)`.
   - `get_target_layer(self) -> nn.Module`: Returns the final convolutional layer for Grad-CAM attribution.
3. Register your architecture in `build_detector(architecture: str, ...)` in [`models/detector.py`](file:///models/detector.py).
4. Add corresponding test case to [`tests/test_detector.py`](file:///tests/test_detector.py).

### 2. Adding a New Video Score Aggregation Strategy
1. Open [`forensic/aggregation.py`](file:///forensic/aggregation.py).
2. Add your function under `# --- Score Aggregation Strategies ---`:
   ```python
   def aggregate_my_strategy(scores: List[float], **kwargs) -> float:
       ...
   ```
3. Register the strategy name in `AGGREGATION_REGISTRY` in [`forensic/aggregation.py`](file:///forensic/aggregation.py).
4. Add unit tests in [`tests/test_aggregation.py`](file:///tests/test_aggregation.py).

### 3. Adding a New UI Analysis Page
1. Define your page renderer in [`ui/pages.py`](file:///ui/pages.py) (e.g. `render_audio_forensics_page(...)`).
2. Add the navigation option to `PAGES` in [`app.py`](file:///app.py) and update `_render_sidebar()`.
3. Use existing design primitives from [`ui/components.py`](file:///ui/components.py) (`render_header`, `render_evidence_card`, `render_verdict_badge`).

---

## Testing & Quality Assurance

Before submitting any code or opening a pull request:

1. **Run full test suite**:
   ```bash
   python -m pytest tests/ -v --tb=short
   ```
   Ensure 100% pass rate.

2. **Run latency and throughput profiler**:
   ```bash
   python scripts/benchmark.py --iterations 30
   ```
   Verify that new changes do not introduce latency regressions or memory leaks.

3. **Run model robustness check**:
   ```bash
   python scripts/robustness_demo.py --n-samples 20
   ```

4. **Verify Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```
   Verify all 5 pages render without UI layout errors or uncaught exceptions.

---

## Pull Request Checklist

- [ ] Clear, descriptive pull request title and summary.
- [ ] Code adheres to clean, typed Python 3 standards with comprehensive docstrings.
- [ ] All existing comments and docstrings preserved.
- [ ] No hardcoded secrets, external API keys, or absolute system paths.
- [ ] All new functions include corresponding unit tests under `tests/`.
- [ ] All 53+ test suite cases pass.
- [ ] Any architectural tradeoffs documented in [`TECHNICAL_DECISIONS.md`](file:///TECHNICAL_DECISIONS.md).

Thank you for helping advance reliability, transparency, and evidential rigor in digital media forensics!
