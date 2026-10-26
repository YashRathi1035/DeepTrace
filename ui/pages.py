"""Multi-page view renderers for the DeepTrace Streamlit Forensic Dashboard."""

import os
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    render_header,
    render_verdict_badge,
    render_legal_disclaimer,
    render_hash_value,
    render_section_divider,
    render_evidence_card,
    render_pipeline_overview,
    render_metric_row,
)
from forensic.metadata import inspect_media_file
from forensic.quality import evaluate_face_quality
from forensic.report import build_evidence_record, export_evidence_json, generate_forensic_pdf
from explainability.gradcam import generate_explanation
from evaluation.plots import plot_video_score_timeline


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Home & Forensic Overview
# ─────────────────────────────────────────────────────────────────────────────
def render_home_page():
    render_header(
        "DeepTrace — Digital Media Forensics & Evidence Analyzer",
        "Reliability-aware deepfake detection · confidence calibration · visual explainability · tamper-evident reporting"
    )

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        ### Project Objective

        **DeepTrace** is a research-grade digital media forensics platform that assesses whether facial content
        in images and videos exhibits signatures consistent with synthetic manipulation (deepfakes).

        Rather than producing a single overconfident binary label, DeepTrace implements a full reliability-aware
        forensic pipeline that returns calibrated probabilities, quality-gated verdicts, and tamper-evident
        evidence records.
        """)

        st.markdown("#### Key Capabilities")

        capabilities = [
            ("🔐 Cryptographic File Auditing", "SHA-256 hash verification and full container metadata inspection."),
            ("👤 Context-Aware Face Extraction", "20% margin expansion around facial boundaries to preserve manipulation seam artifacts."),
            ("📊 Confidence Calibration", "Temperature Scaling to minimize Expected Calibration Error (ECE) and produce reliable probabilities."),
            ("🎯 Selective Abstention", "Three-tier outcome: *Likely Authentic*, *Potentially Manipulated*, or *Inconclusive*."),
            ("🔥 Visual Attribution", "Grad-CAM saliency heatmaps highlighting regions most influential toward the manipulated class."),
            ("⏱️ Temporal Analysis", "Per-frame timeline, suspicious interval localization, and Top-K aggregation for video forensics."),
            ("📄 Forensic Reporting", "Multi-page PDF and machine-readable JSON evidence dossiers with embedded visuals."),
        ]
        for icon_label, desc in capabilities:
            st.markdown(f"- **{icon_label}**: {desc}")

        st.info("""
        🔬 **Core Research Question**:
        Can combining confidence calibration, input quality gates, selective abstention, and temporal consistency
        analysis significantly reduce false positives and silent failures when detecting deepfakes in degraded
        real-world media?
        """)

    with col2:
        st.markdown("#### Forensic Pipeline")
        render_pipeline_overview()

        st.markdown("#### Supported Media Formats")
        col_img, col_vid = st.columns(2)
        with col_img:
            st.markdown("**Images**")
            st.code(".jpg  .jpeg\n.png  .webp", language="text")
        with col_vid:
            st.markdown("**Videos**")
            st.code(".mp4  .avi\n.mov  .mkv", language="text")

        st.markdown("#### Model Architecture")
        st.markdown("""
        | Component | Detail |
        |---|---|
        | Backbone | EfficientNet-B0 |
        | Baseline | ResNet-18 |
        | Calibration | Temperature Scaling |
        | Explainability | Grad-CAM |
        | Face Detector | MediaPipe BlazeFace |
        """)

    render_legal_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Image Analysis
# ─────────────────────────────────────────────────────────────────────────────
def render_image_analysis_page(detector, scaler, policy):
    render_header(
        "Image Forensics & Saliency Attribution",
        "Single-image face detection · calibrated inference · Grad-CAM visual explanation · evidence export"
    )

    uploaded_file = st.file_uploader(
        "Upload target image for forensic analysis",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPEG, PNG, WebP"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        col_img, col_info = st.columns([1, 1], gap="medium")

        with col_img:
            st.image(file_bytes, caption="Uploaded Media Preview", use_container_width=True)

        with col_info:
            meta = inspect_media_file(tmp_path)
            st.markdown("#### 🔐 File Integrity")

            render_evidence_card("File Identification", f"""
            <div style="color:#c9d1d9;font-size:0.88rem;line-height:1.8;">
                <strong style="color:#8b949e;">Name:</strong> {meta.get('file_name')}<br/>
                <strong style="color:#8b949e;">Size:</strong> {meta.get('file_size_bytes', 0):,} bytes<br/>
                <strong style="color:#8b949e;">Dimensions:</strong> {meta.get('width')} × {meta.get('height')} px
            </div>
            """)

            render_hash_value("SHA-256 Integrity Hash", meta.get('sha256', 'N/A'))

            exif = meta.get("exif", {})
            if exif:
                with st.expander("📋 EXIF Metadata", expanded=False):
                    for k, v in list(exif.items())[:12]:
                        st.markdown(f"- **{k}**: `{v}`")
            else:
                st.caption("ℹ️ No EXIF metadata present (common for web-distributed images).")

            st.markdown("")
            analyze_btn = st.button(
                "🔍 Run Forensic Analysis",
                type="primary",
                use_container_width=True,
                key="img_analyze_btn"
            )

        if analyze_btn:
            with st.spinner("Executing face extraction · calibrated inference · Grad-CAM attribution..."):
                try:
                    # 1. Raw inference
                    pred = detector.predict_image(tmp_path)
                    raw_prob = pred["manipulation_score"]
                    logits_arr = np.array([pred["raw_scores"]])
                    cal_prob = float(scaler.calibrate_logits(logits_arr)[0][1])

                    # 2. Quality check
                    q_eval = evaluate_face_quality(pred["crop_bgr"])
                    quality_score = q_eval["quality_score"]

                    # 3. Decision policy
                    decision_res = policy.decide(
                        manipulation_score=cal_prob,
                        quality_score=quality_score,
                        quality_warnings=q_eval["warnings"]
                    )

                    # 4. Grad-CAM
                    cam_res = generate_explanation(
                        model=detector.model,
                        input_tensor=detector.image_processor.process_face(tmp_path)["tensor"],
                        original_bgr=pred["crop_bgr"],
                        target_class=1
                    )

                    st.session_state["image_analysis_result"] = {
                        "pred": pred,
                        "cal_prob": cal_prob,
                        "raw_prob": raw_prob,
                        "decision_res": decision_res,
                        "q_eval": q_eval,
                        "cam_res": cam_res,
                        "tmp_path": tmp_path,
                        "meta": meta,
                    }
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "image_analysis_result" in st.session_state:
            res = st.session_state["image_analysis_result"]
            render_section_divider()

            st.markdown("### Forensic Assessment Results")

            render_verdict_badge(
                verdict=res["decision_res"]["decision"],
                score=res["cal_prob"],
                is_calibrated=True
            )
            st.markdown(
                f'<div style="color:#8b949e;font-size:0.88rem;margin-top:-6px;margin-bottom:14px;">'
                f'<em>{res["decision_res"]["rationale"]}</em></div>',
                unsafe_allow_html=True
            )

            # Metrics row
            render_metric_row([
                {"label": "Manipulation Score", "value": f"{res['cal_prob']:.4f}",
                 "help": "Calibrated (temperature-scaled) probability of manipulation"},
                {"label": "Raw Model Score", "value": f"{res['raw_prob']:.4f}",
                 "help": "Uncalibrated softmax probability"},
                {"label": "Sharpness (Laplacian)", "value": f"{res['q_eval']['sharpness_score']:.1f}",
                 "help": "Variance of Laplacian (>40 = adequately sharp for forensics)"},
                {"label": "Inference Latency", "value": f"{res['pred']['inference_time_ms']:.1f} ms",
                 "help": "Per-frame neural network inference time"},
            ])

            # Quality warnings
            warnings = res["decision_res"].get("warnings", [])
            if warnings:
                for w in warnings:
                    st.warning(f"⚠️ {w}")

            # Visual Evidence
            render_section_divider()
            st.markdown("#### 🔥 Visual Saliency Attribution (Grad-CAM)")
            c_left, c_right = st.columns(2, gap="small")
            with c_left:
                st.image(
                    res["pred"]["crop_rgb"],
                    caption="Preprocessed Face Crop (20% Context Margin)",
                    use_container_width=True
                )
            with c_right:
                st.image(
                    res["cam_res"]["overlay_rgb"],
                    caption="Grad-CAM Manipulation Saliency Overlay (JET Colormap)",
                    use_container_width=True
                )

            st.caption(
                "ℹ️ Red/yellow regions indicate facial features that most strongly activated the "
                "'Manipulated' class neurons. These are neural attribution maps, NOT verified manipulation masks."
            )

            # Export Section
            render_section_divider()
            st.markdown("#### 📋 Export Forensic Dossier")

            reports_dir = "artifacts/reports"
            os.makedirs(reports_dir, exist_ok=True)

            evidence_rec = build_evidence_record(
                media_path=res["tmp_path"],
                analysis_result={
                    "predicted_class": res["decision_res"]["decision"],
                    "manipulation_score": res["cal_prob"],
                    "confidence": res["pred"]["confidence"],
                    "raw_scores": res["pred"]["raw_scores"],
                    "inference_time_ms": res["pred"]["inference_time_ms"],
                    "quality_warnings": res["decision_res"]["warnings"],
                    "is_calibrated": True
                },
                model_name=detector.architecture
            )

            # Save temp images for PDF embedding
            crop_path = os.path.join(reports_dir, "report_crop.png")
            cam_path = os.path.join(reports_dir, "report_cam.png")
            cv2.imwrite(crop_path, res["pred"]["crop_bgr"])
            cv2.imwrite(cam_path, res["cam_res"]["overlay_bgr"])

            pdf_path = os.path.join(reports_dir, "DeepTrace_Image_Forensic_Report.pdf")
            json_path = os.path.join(reports_dir, "DeepTrace_Image_Evidence.json")
            generate_forensic_pdf(evidence_rec, pdf_path, visual_paths=[crop_path, cam_path])
            export_evidence_json(evidence_rec, json_path)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download Forensic PDF Report",
                        data=f.read(),
                        file_name="DeepTrace_Image_Forensic_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_img_pdf"
                    )
            with btn_col2:
                with open(json_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "💾 Download Machine-Readable JSON",
                        data=f.read(),
                        file_name="DeepTrace_Image_Evidence.json",
                        mime="application/json",
                        use_container_width=True,
                        key="dl_img_json"
                    )

    render_legal_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Video Analysis
# ─────────────────────────────────────────────────────────────────────────────
def render_video_analysis_page(analyzer, scaler, policy):
    render_header(
        "Video Forensics & Temporal Evidence Timeline",
        "Multi-frame sampling · score aggregation · anomaly trajectory tracking · temporal interval detection"
    )

    uploaded_vid = st.file_uploader(
        "Upload target video for forensic analysis",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        sampling_fps = st.slider(
            "Sampling Rate (FPS)",
            min_value=0.5, max_value=4.0, value=1.0, step=0.5,
            help="Number of frames sampled per video second. Higher rates are more thorough but slower."
        )
    with col_opt2:
        agg_method = st.selectbox(
            "Score Aggregation Strategy",
            ["top_k", "mean", "median", "trimmed_mean"],
            index=0,
            help="Top-K averages the most suspicious frames — ideal for localized deepfakes."
        )

    if uploaded_vid is not None:
        vid_bytes = uploaded_vid.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_vid.name}") as tmp_v:
            tmp_v.write(vid_bytes)
            tmp_vid_path = tmp_v.name

        # Correct Streamlit video API
        st.video(tmp_vid_path)

        meta = inspect_media_file(tmp_vid_path)
        duration = meta.get('duration_seconds', 'N/A')
        fps_val = meta.get('fps', 'N/A')
        sha_preview = (meta.get('sha256') or 'N/A')[:20]

        render_evidence_card("Video Container Metadata", f"""
        <div style="color:#c9d1d9;font-size:0.88rem;line-height:2.0;">
            <strong style="color:#8b949e;">Codec:</strong> {meta.get('fourcc', 'N/A')} &nbsp;|&nbsp;
            <strong style="color:#8b949e;">Duration:</strong> {duration}s &nbsp;|&nbsp;
            <strong style="color:#8b949e;">FPS:</strong> {fps_val} &nbsp;|&nbsp;
            <strong style="color:#8b949e;">Frames:</strong> {meta.get('frame_count', 'N/A')}<br/>
            <strong style="color:#8b949e;">SHA-256:</strong>
            <span style="font-family:monospace;color:#79c0ff;">{sha_preview}...</span>
        </div>
        """)

        run_vid_btn = st.button(
            "🎞️ Run Full Video Forensics",
            type="primary",
            use_container_width=True,
            key="vid_analyze_btn"
        )

        if run_vid_btn:
            analyzer.video_processor.sampling_fps = sampling_fps
            with st.spinner("Streaming frames · detecting faces · computing temporal dynamics..."):
                try:
                    vid_res = analyzer.analyze_video(tmp_vid_path, aggregation_method=agg_method)
                    st.session_state["video_analysis_result"] = vid_res
                    st.session_state["video_temp_path"] = tmp_vid_path
                except Exception as e:
                    st.error(f"Video analysis failed: {e}")

        if "video_analysis_result" in st.session_state:
            v_res = st.session_state["video_analysis_result"]
            render_section_divider()

            st.markdown("### Video Forensic Verdict")
            render_verdict_badge(
                verdict=v_res["predicted_class"],
                score=v_res["video_level_score"],
                is_calibrated=True
            )

            # Key metrics
            render_metric_row([
                {"label": "Video-Level Score", "value": f"{v_res['video_level_score']:.4f}",
                 "help": f"Aggregated via {v_res['selected_aggregation_method']}"},
                {"label": "Frames Analyzed", "value": str(v_res["num_analyzed_frames"])},
                {"label": "Frames Skipped", "value": str(v_res["num_frames_without_faces"]),
                 "help": "Frames where no face was detected"},
                {"label": "Score Variance", "value": f"{v_res['temporal_consistency']['variance']:.4f}",
                 "help": "High variance may indicate temporal manipulation"},
            ])

            # Suspicious intervals
            susp_intervals = v_res["temporal_consistency"]["suspicious_intervals"]
            if susp_intervals:
                st.warning(
                    f"⚠️ **{len(susp_intervals)} Suspicious Temporal Segment(s) Detected** — "
                    f"Continuous sequences exceeding the manipulation threshold:"
                )
                for s in susp_intervals:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;▸ **{s['start_seconds']}s – {s['end_seconds']}s** "
                        f"| Peak Score: **{s['peak_manipulation_score']:.4f}**"
                    )
            else:
                st.success("✅ **Temporal Stability**: No continuous manipulation intervals detected.")

            if v_res["temporal_consistency"].get("volatility_warning"):
                st.warning("⚠️ " + v_res["temporal_consistency"]["notes"])

            # Timeline plot
            render_section_divider()
            st.markdown("#### 📈 Manipulation Probability Timeline")
            plots_dir = "artifacts/plots"
            os.makedirs(plots_dir, exist_ok=True)
            timeline_img = os.path.join(plots_dir, "ui_video_timeline.png")

            plot_video_score_timeline(
                timestamps=v_res["timestamps"],
                scores=[r["manipulation_score"] for r in v_res["per_frame_scores"]],
                video_level_score=v_res["video_level_score"],
                threshold=analyzer.fake_threshold,
                suspicious_intervals=susp_intervals,
                output_path=timeline_img
            )
            st.image(timeline_img, caption="Per-frame manipulation trajectory across video timeline", use_container_width=True)

            # Frame Inspector
            render_section_divider()
            st.markdown("#### 🔍 Frame Inspector")
            frames = v_res["per_frame_scores"]
            if frames:
                frame_options = [
                    f"Frame {f['frame_index']} | t={f['timestamp']}s | Score={f['manipulation_score']:.3f}"
                    for f in frames
                ]
                chosen_idx = st.selectbox(
                    "Select sampled frame to inspect",
                    range(len(frames)),
                    format_func=lambda i: frame_options[i],
                    key="frame_inspector"
                )
                chosen_frame = frames[chosen_idx]

                col_f1, col_f2 = st.columns([1, 1], gap="medium")
                with col_f1:
                    if chosen_frame.get("crop_rgb") is not None:
                        st.image(
                            chosen_frame["crop_rgb"],
                            caption=f"Face Crop at t={chosen_frame['timestamp']}s",
                            use_container_width=True
                        )
                    else:
                        st.info("No face crop available for this frame.")
                with col_f2:
                    st.markdown(f"""
                    | Property | Value |
                    |---|---|
                    | Assigned Face ID | `{chosen_frame['face_id']}` |
                    | Manipulation Score | `{chosen_frame['manipulation_score']:.4f}` |
                    | Predicted Class | `{chosen_frame['predicted_class']}` |
                    | Detection Confidence | `{chosen_frame['detection_confidence']:.2f}` |
                    | Sharpness (Laplacian) | `{chosen_frame['quality'].get('blur_score', 'N/A')}` |
                    """)

            # Export
            render_section_divider()
            st.markdown("#### 📋 Export Video Forensic Report")

            rec = build_evidence_record(
                media_path=st.session_state["video_temp_path"],
                analysis_result=v_res,
                model_name=analyzer.detector.architecture,
                heatmap_paths=[timeline_img]
            )

            reports_dir = "artifacts/reports"
            os.makedirs(reports_dir, exist_ok=True)
            pdf_out = os.path.join(reports_dir, "DeepTrace_Video_Forensic_Report.pdf")
            json_out = os.path.join(reports_dir, "DeepTrace_Video_Evidence.json")
            generate_forensic_pdf(rec, pdf_out, visual_paths=[timeline_img])
            export_evidence_json(rec, json_out)

            b1, b2 = st.columns(2)
            with b1:
                with open(pdf_out, "rb") as f:
                    st.download_button(
                        "📄 Download Video PDF Report",
                        data=f.read(),
                        file_name="DeepTrace_Video_Forensic_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_vid_pdf"
                    )
            with b2:
                with open(json_out, "r", encoding="utf-8") as f:
                    st.download_button(
                        "💾 Download Video JSON Evidence",
                        data=f.read(),
                        file_name="DeepTrace_Video_Evidence.json",
                        mime="application/json",
                        use_container_width=True,
                        key="dl_vid_json"
                    )

    render_legal_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Forensic Report Dossier Viewer
# ─────────────────────────────────────────────────────────────────────────────
def render_report_page():
    render_header(
        "Forensic Dossier & Evidence Record Viewer",
        "Audit cryptographic signatures, model parameters, and chain-of-custody metadata"
    )

    reports_dir = "artifacts/reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_files = sorted([
        f for f in os.listdir(reports_dir)
        if f.endswith(".json") or f.endswith(".pdf")
    ])

    if not report_files:
        st.info(
            "📂 No forensic evidence records found yet.\n\n"
            "Run an image or video analysis first to populate the dossier."
        )
        render_legal_disclaimer()
        return

    st.markdown(f"### 📁 Archived Forensic Records ({len(report_files)} files)")
    render_section_divider()

    json_files = [f for f in report_files if f.endswith(".json")]
    pdf_files = [f for f in report_files if f.endswith(".pdf")]

    # JSON viewer section
    if json_files:
        st.markdown("#### 📄 JSON Evidence Records")
        for rf in json_files:
            full_p = os.path.join(reports_dir, rf)
            sz = os.path.getsize(full_p)
            with st.expander(f"🗂️ {rf}  ({sz:,} bytes)", expanded=False):
                try:
                    with open(full_p, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Key summary
                    findings = data.get("model_findings", {})
                    file_obs = data.get("file_observations", {})
                    verdict = findings.get("media_classification", "N/A")
                    score = findings.get("manipulation_score") or findings.get("video_level_score", 0.0)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Verdict", verdict)
                    c2.metric("Manipulation Score", f"{score:.4f}" if isinstance(score, float) else str(score))
                    c3.metric("Record ID", data.get("record_id", "N/A")[:20] + "...")

                    st.markdown(f"**SHA-256**: `{file_obs.get('sha256', 'N/A')}`")
                    st.markdown(f"**Generated**: {data.get('timestamp_utc', 'N/A')}")

                    with st.expander("Raw JSON Data", expanded=False):
                        st.json(data)

                    with open(full_p, "rb") as f:
                        st.download_button(
                            f"⬇️ Download {rf}",
                            data=f.read(),
                            file_name=rf,
                            mime="application/json",
                            key=f"dl_{rf}"
                        )
                except Exception as e:
                    st.error(f"Could not parse {rf}: {e}")

    # PDF downloads
    if pdf_files:
        st.markdown("#### 📋 PDF Forensic Reports")
        for rf in pdf_files:
            full_p = os.path.join(reports_dir, rf)
            sz = os.path.getsize(full_p)
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                st.markdown(f"📌 **{rf}** &nbsp;—&nbsp; {sz:,} bytes")
            with col_btn:
                with open(full_p, "rb") as f:
                    st.download_button(
                        "⬇️ Download",
                        data=f.read(),
                        file_name=rf,
                        mime="application/pdf",
                        key=f"dl_{rf}"
                    )

    render_legal_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: Research & Evaluation Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def render_evaluation_dashboard():
    render_header(
        "Research Evaluation & Benchmark Dashboard",
        "Empirical validation metrics · calibration diagrams · perturbation stress-tests · ablation results"
    )

    plots_dir = "artifacts/plots"
    reports_dir = "artifacts/reports"
    ablation_csv = os.path.join(reports_dir, "ablation_study.csv")
    rel_diagram = os.path.join(plots_dir, "reliability_diagram.png")
    robust_curves = os.path.join(plots_dir, "robustness_curves.png")
    cm_plot = os.path.join(plots_dir, "val_confusion_matrix.png")
    train_curves = os.path.join(plots_dir, "training_curves.png")

    if not os.path.exists(ablation_csv):
        st.warning("""
        ⚠️ **No evaluation benchmark records found.**

        Generate scientific benchmarks by running:
        ```powershell
        python scripts/evaluate_model.py --config configs/config.yaml
        ```
        This will produce ablation tables, reliability diagrams, robustness curves, and confusion matrices.
        """)
        render_legal_disclaimer()
        return

    # ── Section 1: Ablation Study ──
    st.markdown("### 1. Comparative Ablation Study")
    st.caption("Model variants evaluated across accuracy, AUC-ROC, Brier Score, and Expected Calibration Error.")
    df_ablation = pd.read_csv(ablation_csv)

    # Highlight best per column
    numeric_cols = df_ablation.select_dtypes(include=[float, int]).columns.tolist()
    st.dataframe(
        df_ablation.style.highlight_max(subset=numeric_cols, color="#1f3a5c").highlight_min(
            subset=[c for c in numeric_cols if "ECE" in c or "Brier" in c or "ece" in c.lower()],
            color="#1a3d2a"
        ),
        use_container_width=True
    )

    render_section_divider()

    # ── Section 2: Calibration & Reliability ──
    st.markdown("### 2. Confidence Calibration & Reliability Diagrams")
    col_rel, col_robust = st.columns(2, gap="medium")
    with col_rel:
        if os.path.exists(rel_diagram):
            st.image(
                rel_diagram,
                caption="Reliability Diagram: Empirical Accuracy vs. Predicted Confidence (10 bins)",
                use_container_width=True
            )
        else:
            st.info("Reliability diagram not yet generated. Run evaluate_model.py.")

    with col_robust:
        if os.path.exists(robust_curves):
            st.image(
                robust_curves,
                caption="Robustness Degradation: JPEG Compression, Gaussian Blur, Downscaling & Sensor Noise",
                use_container_width=True
            )
        else:
            st.info("Robustness curves not yet generated.")

    render_section_divider()

    # ── Section 3: Training History & Confusion Matrix ──
    st.markdown("### 3. Training Performance & Confusion Matrix")
    col_tr, col_cm = st.columns(2, gap="medium")
    with col_tr:
        if os.path.exists(train_curves):
            st.image(
                train_curves,
                caption="Training & Validation Loss / Accuracy Curves",
                use_container_width=True
            )
        else:
            st.info("Training curves not yet rendered.")
    with col_cm:
        if os.path.exists(cm_plot):
            st.image(
                cm_plot,
                caption="Validation Confusion Matrix",
                use_container_width=True
            )
        else:
            st.info("Confusion matrix not yet rendered.")

    # ── Section 4: Any extra saved plots ──
    if os.path.exists(plots_dir):
        extra_plots = [
            f for f in os.listdir(plots_dir)
            if f.endswith(".png") and f not in {
                "reliability_diagram.png", "robustness_curves.png",
                "val_confusion_matrix.png", "training_curves.png",
                "ui_video_timeline.png"
            }
        ]
        if extra_plots:
            render_section_divider()
            st.markdown("### 4. Additional Experiment Plots")
            cols = st.columns(min(3, len(extra_plots)))
            for i, plot_file in enumerate(extra_plots[:6]):
                with cols[i % 3]:
                    plot_path = os.path.join(plots_dir, plot_file)
                    st.image(plot_path, caption=plot_file.replace("_", " ").replace(".png", ""), use_container_width=True)

    render_legal_disclaimer()
