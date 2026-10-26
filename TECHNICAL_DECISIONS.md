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
