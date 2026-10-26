"""DeepTrace — Reliability-Aware Deepfake Forensics and Evidence Analyzer.

Main Streamlit application entry point.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import torch

from configs import load_config
from models.predict import DeepfakeDetector
from forensic.calibration import TemperatureScaler, ReliabilityDecisionPolicy
from forensic.aggregation import VideoForensicAnalyzer
from ui.components import apply_custom_theme
from ui.pages import (
    render_home_page,
    render_image_analysis_page,
    render_video_analysis_page,
    render_report_page,
    render_evaluation_dashboard,
)


# ─────────────────────────────────────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepTrace — Digital Media Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/YashRathi1035/DeepTrace",
        "About": "DeepTrace v1.0 — Reliability-Aware Deepfake Forensics & Evidence Analyzer",
    }
)

apply_custom_theme()


# ─────────────────────────────────────────────────────────────────────────────
# Cached Pipeline Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading forensic inference pipeline...")
def load_forensic_pipeline():
    """Load and cache detector, calibration scaler, decision policy, and video analyzer.

    Uses @st.cache_resource so heavy neural models are instantiated exactly once
    across all re-runs of the Streamlit app.

    Returns:
        Tuple of (DeepfakeDetector, TemperatureScaler, ReliabilityDecisionPolicy,
                  VideoForensicAnalyzer) or (None, None, None, None) if no checkpoint.
    """
    config = load_config()

    # 1. Locate checkpoint
    chk_path = config["model"].get("default_checkpoint", "artifacts/checkpoints/best_model.pt")
    if not os.path.exists(chk_path):
        chk_dir = config["paths"]["checkpoints_dir"]
        if os.path.exists(chk_dir):
            pts = sorted([f for f in os.listdir(chk_dir) if f.endswith(".pt")])
            chk_path = os.path.join(chk_dir, pts[0]) if pts else None
        else:
            chk_path = None

    if chk_path is None or not os.path.exists(chk_path):
        return None, None, None, None

    # 2. Device selection
    use_cuda = config["system"]["device"] == "cuda" and torch.cuda.is_available()
    device_name = "cuda" if use_cuda else "cpu"

    # 3. Calibration scaler
    cal_path = os.path.join(config["paths"]["checkpoints_dir"], "calibration.json")
    scaler = (
        TemperatureScaler.load(cal_path)
        if os.path.exists(cal_path)
        else TemperatureScaler(initial_temperature=1.0)
    )

    # 4. Neural detector
    detector = DeepfakeDetector(checkpoint_path=chk_path, device=device_name)

    # 5. Decision policy
    policy = ReliabilityDecisionPolicy(
        fake_threshold=config["reliability"]["abstention"]["fake_threshold"],
        real_threshold=config["reliability"]["abstention"]["real_threshold"],
    )

    # 6. Video analyzer
    analyzer = VideoForensicAnalyzer(
        detector=detector,
        default_aggregation="top_k",
        top_k_ratio=0.20,
        fake_threshold=config["reliability"]["abstention"]["fake_threshold"],
    )

    return detector, scaler, policy, analyzer


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar(detector, scaler) -> str:
    """Render navigation sidebar with system telemetry. Returns selected page name."""
    with st.sidebar:
        st.markdown("""
        <div style="padding:14px 0 8px; text-align:center;">
            <div style="font-size:2rem;">🛡️</div>
            <div style="font-size:1.2rem;font-weight:700;color:#f0f6fc;margin-top:4px;">DeepTrace</div>
            <div style="font-size:0.78rem;color:#8b949e;margin-top:2px;">Digital Media Forensics</div>
        </div>
        <hr style="border-color:#21262d;margin:10px 0 16px;"/>
        """, unsafe_allow_html=True)

        nav_options = [
            "🏠 Home & Overview",
            "🖼️ Image Analysis",
            "🎥 Video Forensics",
            "📑 Forensic Dossier",
            "📊 Evaluation Dashboard",
        ]
        selected_page = st.radio("Navigate", nav_options, index=0, label_visibility="collapsed")

        st.markdown("<hr style='border-color:#21262d;margin:16px 0 12px;'/>", unsafe_allow_html=True)
        st.markdown("**System Status**")

        if detector is not None:
            st.success(f"✅ Model loaded: `{detector.architecture}`")
            dev_label = (
                f"CUDA ({torch.cuda.get_device_name(0).split(' ')[-1]})"
                if detector.device.type == "cuda"
                else "CPU (no CUDA)"
            )
            st.info(f"🖥️ Compute: `{dev_label}`")
            st.info(f"🌡️ Calibration T: `{scaler.temperature.item():.4f}`")
        else:
            st.error("❌ No model checkpoint found")
            st.caption("Train a model first:\n```\npython -m models.train\n```")

        st.markdown("<hr style='border-color:#21262d;margin:12px 0 8px;'/>", unsafe_allow_html=True)
        st.caption("DeepTrace v1.0 · MIT License\nFor research use only.")

    return selected_page


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    detector, scaler, policy, analyzer = load_forensic_pipeline()
    selected_page = _render_sidebar(detector, scaler)

    if selected_page == "🏠 Home & Overview":
        render_home_page()

    elif selected_page == "🖼️ Image Analysis":
        if detector is None:
            st.error(
                "**No model checkpoint found** in `artifacts/checkpoints/`.\n\n"
                "Please run training first: `python -m models.train`"
            )
        else:
            render_image_analysis_page(detector, scaler, policy)

    elif selected_page == "🎥 Video Forensics":
        if detector is None:
            st.error(
                "**No model checkpoint found** in `artifacts/checkpoints/`.\n\n"
                "Please run training first: `python -m models.train`"
            )
        else:
            render_video_analysis_page(analyzer, scaler, policy)

    elif selected_page == "📑 Forensic Dossier":
        render_report_page()

    elif selected_page == "📊 Evaluation Dashboard":
        render_evaluation_dashboard()


if __name__ == "__main__":
    main()
