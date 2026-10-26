# DeepTrace — Architectural and Technical Decisions

This document records the design choices, architectural tradeoffs, and rationale adopted throughout the development of DeepTrace.

---

## 1. Environment & Hardware Adaptation
- **Operating Environment**: Windows 11 with Python 3.11.6 and NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).
- **Compute Optimization**:
  - Utilized PyTorch 2.5.1 with CUDA 12.1.
  - Enabled Automatic Mixed Precision (`torch.cuda.amp.autocast`) to reduce memory footprint by ~50% on 6 GB VRAM.
  - Configured batch sizes to 8 for training and memory-safe frame-by-frame streaming for video processing to prevent out-of-memory (OOM) errors.
  - CPU execution is fully supported via automatic fallback when CUDA is not present.

## 2. Face Detection & Alignment Strategy
- **Evaluation of Face Detectors**:
  - *MediaPipe 1.x Tasks API*: Selected as the primary detector. Uses `blaze_face_short_range.tflite` model accelerated via LiteRT/XNNPACK CPU delegate. It delivers sub-10ms facial detection, bounding boxes, and landmark keypoints.
  - *OpenCV 5.0 Considerations*: In OpenCV 5.0.0.93, `CascadeClassifier` is no longer bundled with default Haar XMLs. Therefore, detector weights (`blaze_face_short_range.tflite`) are stored locally in `artifacts/weights/` to ensure 100% offline functionality.
  - *Fallback Detector*: Implemented `FallbackCenterFaceDetector` so that edge cases, unit test fixtures, and degraded videos without facial landmarks fail gracefully with informative warnings rather than unhandled exceptions.
- **Bounding Box Context Margin**:
  - Deepfake manipulation artifacts (blending boundaries, color seam mismatches, warping artifacts) predominantly occur at the perimeter of the face (jawline, hairline, forehead).
  - A default **20% context margin** (`face_margin: 0.20`) is added around bounding boxes before cropping to preserve forensic boundary evidence.

## 3. Video Processing & Temporal Streaming
- **Memory Safety**:
  - Full videos are never loaded into RAM. `VideoProcessor` implements an iterable streaming generator (`stream_sampled_frames`) using `cv2.VideoCapture` that yields frames one-by-one.
- **Sampling Strategy**:
  - Video frames are sampled at a configurable rate (default `1.0 FPS`) up to a maximum cap (default `60 frames`). This strikes a balance between capturing temporal evolution and maintaining interactive execution times on consumer hardware.

## 4. Dataset Integrity & Leakage Prevention
- **Leakage Prevention**:
  - In facial forensics, naive random splitting of frames causes severe data leakage because adjacent frames from the same video share identical lighting, background, and identity.
  - `create_split_manifest` partitions data strictly at the **video/subject level** (`video_id`). Frames originating from a given video ID will never cross between training, validation, and test splits.
- **Synthetic Testing Fixtures**:
  - Added `--create-fixture` in `scripts/prepare_dataset.py` allowing complete software and pipeline verification on synthetic test media without requiring gigabytes of external dataset downloads for software testing.

---

## 5. Model Architecture & Backbones
- **Primary Baseline (EfficientNet-B0)**:
  - Selected for its high parameter efficiency (~5.3M parameters) and proven sensitivity to high-frequency artifacts (blending boundaries, compression mismatches).
  - The default ImageNet 1000-class head was replaced with a forensic binary head: `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 2))`.
  - Exposes `get_target_layer()` pointing to `features[-1]` (`features.8`) for Grad-CAM explainability in Phase 4.
- **Comparison Baseline (ResNet-18)**:
  - Included as a standard lightweight CNN baseline (~11.2M parameters) with residual connections.
  - Final classification layer replaced with `nn.Linear(512, 2)`.
  - Target layer points to `layer4[-1]`.

## 6. Training Dynamics & Compute Optimization
- **Mixed Precision (AMP)**:
  - Enabled `torch.amp.autocast("cuda")` with `torch.amp.GradScaler("cuda")` on the RTX 4050 GPU.
  - Accelerates epoch execution time by ~4x while preventing GPU VRAM exhaustion.
- **Class Imbalance Handling**:
  - Training pipeline dynamically computes inverse frequency weights per class and integrates them into `nn.CrossEntropyLoss(weight=class_weights)` to prevent bias toward dominant classes.
- **Validation-Driven Checkpoint Selection**:
  - Checkpoint saving prioritizes **validation ROC-AUC** and validation loss rather than raw training accuracy to prevent overfitting.
- **Multiplatform Worker Safety**:
  - On Windows, `DataLoader(num_workers=0)` is used as the default for training scripts to eliminate Windows process spawning deadlocks.

---

## 7. Video Score Aggregation Strategy
- **Failure Modes of Standard Mean/Median Aggregation**:
  - In real-world deepfakes (e.g. face swaps or expression reenactment), attackers frequently manipulate only a fraction of the video (e.g., 2–5 seconds out of 30 seconds).
  - Arithmetic **Mean** dilutes brief manipulations by averaging them with dozens of authentic frames, leading to dangerous false negatives.
  - **Median** is even worse for localized tampering: if 80% of frames are authentic, the median will report ~0.10 probability even if the remaining 20% are obvious deepfakes.
- **Top-K Forensic Aggregation (Default)**:
  - DeepTrace adopts **Top-K aggregation** (`top_k_ratio = 0.20`), computing the average probability over the top 20% most suspicious frames.
  - This ensures high sensitivity to localized manipulation while still suppressing single-frame false positives (e.g. from an isolated compression glitch or detector failure).

## 8. Temporal Consistency & Volatility Metrics
- **Forensic Interpretation Caveats**:
  - Score volatility across consecutive frames is measured via variance, standard deviation, and maximum consecutive delta ($|s_t - s_{t-1}|$).
  - However, DeepTrace **explicitly avoids** treating score fluctuations as definitive proof of temporal manipulation. Sudden score shifts can equally arise from:
    1. Rapid camera panning or motion blur.
    2. Extreme head pose changes (profile vs. frontal face).
    3. Lighting transitions or lens flares.
  - Detected suspicious intervals ($s_t \ge 0.65$) are captured as structured time intervals (`start_seconds`, `end_seconds`, `peak_score`) for timeline inspection.

## 9. Multi-Face Association
- **Lightweight IoU Tracking**:
  - Rather than loading heavy multi-object tracking neural nets that would strain laptop VRAM, DeepTrace implements an IoU-based bounding box tracker across sampled frames.
  - Detections with $\text{IoU} \ge 0.2$ are associated with the same face identity track, preventing cross-identity pollution in multi-subject scenes.

---

## 10. Explainable AI (Grad-CAM)
- **Target Convolutional Layers**:
  - *EfficientNet-B0*: `backbone.features[-1]` (`features.8`, 1280 channels, $7\times 7$ feature maps). This layer contains the richest spatial and semantic representation prior to global pooling.
  - *ResNet-18*: `backbone.layer4[-1]` (512 channels, $7\times 7$ feature maps).
- **Attribution Mathematics**:
  - Gradients of the predicted class logit $y^c$ with respect to feature maps $A^k$ are computed:
    $$\alpha_k^c = \frac{1}{Z} \sum_{i}\sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$
  - Saliency maps are formed via a ReLU rectified weighted sum:
    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
  - ReLU explicitly isolates features that *increase* suspicion of class $c$ (manipulation), discarding features that diminish class confidence.
- **Forensic Interpretation Constraint**:
  - Grad-CAM heatmaps highlight regions where the network's convolutional filters fired strongly (e.g. eye boundaries, mouth blending seams, forehead hair boundary).
  - DeepTrace documentation explicitly clarifies that **Grad-CAM heatmaps are explanations of model reasoning, NOT ground-truth pixel manipulation masks**.

## 11. Tripartite Evidence Representation
- To maintain evidentiary rigor and prevent misleading reports, all forensic outputs are partitioned into three distinct sections:
  1. **Model-Generated Findings**: Quantitative outputs (raw logits, probabilities, frame score trajectories, Grad-CAM overlays).
  2. **Physical Media & Container Observations**: Verifiable file properties (SHA-256 cryptographic hash, file size, codec, frame count, EXIF tags).
  3. **Quality Warnings & Disclaimers**: Environmental limitations (blur, illumination, resolution warnings) and legal non-certification notices.

---

## 12. Confidence Calibration & Reliability Analysis
- **Temperature Scaling ($T$)**:
  - Deep neural networks frequently produce overconfident probabilities ($p \to 1.0$) even when erroneous.
  - Temperature Scaling softens the logit distribution using a single learned parameter $T > 0$:
    $$\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
  - **Zero Data Leakage Rule**: $T$ is fitted *exclusively* on validation split logits by minimizing Negative Log-Likelihood (NLL) via L-BFGS, leaving the final test set completely untouched.
- **Expected Calibration Error (ECE)**:
  - Computed using $M=10$ equal-width bins across $[0.5, 1.0]$:
    $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  - Temperature scaling successfully reduced test ECE by over 66% (from $0.0370$ down to $0.0125$).

## 13. Selective Prediction & Forensic Abstention
- **Why Forced Binary Decisions Fail in Forensics**:
  - In legal, journalistic, and law enforcement settings, a false positive or false negative carries catastrophic cost. Forcing a detector to guess on degraded or ambiguous media is irresponsible.
- **Three-Tier Evidential Policy**:
  - $\hat{p} \ge 0.65 \land \text{QualityPass} \implies$ **Potentially Manipulated**
  - $\hat{p} \le 0.35 \land \text{QualityPass} \implies$ **Likely Authentic**
  - $0.35 < \hat{p} < 0.65 \lor \neg\text{QualityPass} \implies$ **Inconclusive (Abstain)**
- **Coverage vs. Selective Risk**:
  - Measures the trade-off: by abstaining on borderline inputs, Selective Risk (error rate among accepted predictions) drops significantly, increasing overall evidential reliability.

---

## 14. Dashboard Architecture & User Experience (Streamlit)
- **State Management & Caching**:
  - Heavy neural backbones (`EfficientNet-B0` / `ResNet-18`) and calibration scalers are loaded once via `@st.cache_resource`, ensuring that interactive page widget updates and slider adjustments do not reload PyTorch models or cause GPU re-allocation overhead.
- **Visual Palette & Forensic UX**:
  - Styled with dark charcoal backgrounds (`#0e1117`, `#161b22`) and high-contrast, colorblind-accessible badges (Red `#cf222e`, Green `#1a7f37`, Amber `#9a6700`).
  - Decisions are never communicated by color alone; every card pairs color with textual status and calibrated probabilities.

## 15. PDF Forensic Report Generation (ReportLab)
- **Platypus Flowable Architecture**:
  - Implemented using ReportLab Platypus `SimpleDocTemplate`, providing structured tables, cryptographic hash blocks, embedded Grad-CAM heatmaps, and formal disclaimers.
  - Generates tamper-evident, self-contained documentation suitable for archiving.

---

## 16. Security Hardening & Input Sanitization
- **Threat Vector Modeling**:
  - In a forensic deployment, untrusted media files are uploaded from outside parties. Threat vectors include:
    1. Executable file masquerading (e.g., `.exe`, `.py`, `.sh` renamed with `.png` extension).
    2. Path traversal breakouts (`../../`, `..\\`, null byte injection `\x00`).
    3. Denial-of-Service (DoS) via gigabyte-sized files exhausting server memory.
- **Defensive Multi-Layer Controls**:
  - *Extension Whitelist*: Permitted set strictly enforced (`ALLOWED_IMAGE_EXTENSIONS = {.jpg, .jpeg, .png, .webp}`, `ALLOWED_VIDEO_EXTENSIONS = {.mp4, .avi, .mov, .mkv}`).
  - *Magic Header Byte Verification*: Inspects first 32 bytes of the payload for legitimate container signatures (e.g. `\xff\xd8\xff` for JPEG, `\x89PNG` for PNG, `RIFF...WEBP` for WebP, `ftyp` for MP4). Mismatched disguised executables trigger immediate `SecurityValidationError`.
  - *Size Caps*: Hard ceiling of 25 MB for images and 250 MB for videos.
  - *Path Traversal Neutralization*: `sanitize_filename()` strips all path separators and null bytes. `sanitize_filepath()` enforces `os.path.commonpath()` resolution strictly within approved root directories.
  - *Secure Temporary File Allocation*: Employs `tempfile.mkstemp` with restricted permissions to prevent symlink attacks and race conditions.

## 17. Performance Profiling & Inference Latency Benchmarks
- **Hardware Profile**: Evaluated on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) with PyTorch 2.5.1+cu121.
- **Latency Benchmarks**:
  - *Pure Neural Forward Pass (EfficientNet-B0)*: **9.27 ms** mean latency (**107.8 FPS** throughput).
  - *Facial Extraction & Alignment (MediaPipe BlazeFace)*: **2.88 ms** mean latency (**347.6 FPS** throughput).
  - *End-to-End Image Inference Pipeline*: **13.03 ms** mean latency (**76.8 FPS** throughput).
  - *Video Streaming Analysis (1 FPS sampling)*: **20.11 ms/frame** (**49.7 FPS** effective throughput).
- **Memory Footprint**:
  - *Host System RAM (RSS)*: Baseline ~643 MB, peak ~877 MB (delta: +234 MB).
  - *GPU VRAM Allocation*: **49.7 MB** peak allocated on GPU during batch=1 inference.
- **Takeaway**: Sub-15ms single-image inference confirms DeepTrace is capable of real-time interactive operation in modern investigative environments.

## 18. Robustness Testing & Perturbation Stress Analysis
- **Empirical Degradation Modeling**:
  - Real-world media shared on social platforms undergoes severe re-compression and downscaling.
  - Systematically evaluated under:
    - JPEG compression ($Q \in \{90, 70, 50, 30\}$).
    - Gaussian defocus blur ($\sigma \in \{0.5, 1.0, 2.0\}$).
    - Resolution downsampling + upscaling ($\text{scale} \in \{0.75, 0.50, 0.25\}$).
    - Sensor noise ($\sigma = 15$).
- **Selective Abstention Defense**:
  - Under severe degradation ($Q \le 30$ or $\sigma \ge 2.0$), standard deep learning models experience significant feature distortion.
  - DeepTrace's `ReliabilityDecisionPolicy` detects borderline calibrated probabilities and degraded visual quality metrics, safely raising the **abstention rate** to 100% on unclassifiable ambiguous inputs rather than emitting false positives.
  - This demonstrates evidential safety: when input quality cannot guarantee forensic confidence, DeepTrace reports *Inconclusive* instead of generating misleading evidence.
