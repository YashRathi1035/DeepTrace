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
