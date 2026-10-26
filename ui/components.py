"""Reusable forensic UI widgets, badges, layout components, and styling for DeepTrace Streamlit app."""

import streamlit as st
from typing import Dict, Any, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Theme / Global CSS
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_theme():
    """Apply professional, dark-neutral forensic dashboard styling with premium micro-animations."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Global overrides ── */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }
        /* Sidebar polish */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1117 0%, #111722 100%);
            border-right: 1px solid #21262d;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stCaption { color: #8b949e; }

        /* ── Top Banner ── */
        .forensic-header {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 60%, #0d1623 100%);
            border-left: 4px solid #1f6feb;
            padding: 20px 28px;
            border-radius: 8px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
            position: relative;
            overflow: hidden;
        }
        .forensic-header::before {
            content: '';
            position: absolute;
            top: 0; right: 0; bottom: 0; width: 200px;
            background: linear-gradient(90deg, transparent, rgba(31,111,235,0.04));
            pointer-events: none;
        }
        .forensic-header h2 {
            margin: 0;
            color: #f0f6fc;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.6px;
        }
        .forensic-header p {
            margin: 5px 0 0;
            color: #8b949e;
            font-size: 0.875rem;
            font-weight: 400;
        }

        /* ── Stat / Metric Cards ── */
        .stat-card {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 12px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .stat-card:hover {
            border-color: #30363d;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .stat-card .label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #8b949e;
            margin-bottom: 5px;
            font-weight: 500;
        }
        .stat-card .value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #f0f6fc;
        }

        /* ── Verdict Badges ── */
        .badge-fake {
            display: inline-flex; align-items: center; gap: 7px;
            background: rgba(248,81,73,0.12);
            color: #f85149;
            border: 1px solid rgba(248,81,73,0.35);
            padding: 8px 18px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 1.0rem;
            letter-spacing: 0.03em;
            animation: badge-pulse-fake 2.5s infinite;
        }
        @keyframes badge-pulse-fake {
            0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.0); }
            50% { box-shadow: 0 0 0 6px rgba(248,81,73,0.08); }
        }
        .badge-real {
            display: inline-flex; align-items: center; gap: 7px;
            background: rgba(46,160,67,0.12);
            color: #3fb950;
            border: 1px solid rgba(46,160,67,0.35);
            padding: 8px 18px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 1.0rem;
            letter-spacing: 0.03em;
        }
        .badge-inconclusive {
            display: inline-flex; align-items: center; gap: 7px;
            background: rgba(210,153,34,0.12);
            color: #d29922;
            border: 1px solid rgba(210,153,34,0.35);
            padding: 8px 18px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 1.0rem;
            letter-spacing: 0.03em;
        }

        /* ── Info / Notice boxes ── */
        .forensic-notice {
            background: #161b22;
            border: 1px solid #21262d;
            border-left: 3px solid #d29922;
            padding: 13px 18px;
            border-radius: 6px;
            font-size: 0.83rem;
            color: #8b949e;
            margin-top: 20px;
            line-height: 1.55;
        }
        .info-box {
            background: linear-gradient(135deg,#0d1624 0%,#101c2e 100%);
            border: 1px solid #1f3a5c;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 10px 0;
        }
        .evidence-card {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 14px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .evidence-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        /* ── Hash / mono values ── */
        .hash-value {
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 0.82rem;
            color: #79c0ff;
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 4px;
            padding: 4px 10px;
            display: inline-block;
            word-break: break-all;
        }

        /* ── Pipeline step badges ── */
        .pipeline-step {
            display: flex; align-items: flex-start; gap: 14px;
            padding: 10px 0;
            border-bottom: 1px solid #21262d;
        }
        .pipeline-step:last-child { border-bottom: none; }
        .step-number {
            min-width: 28px; height: 28px;
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.78rem; font-weight: 700; color: #fff;
        }
        .step-text { color: #c9d1d9; font-size: 0.9rem; line-height: 1.45; }
        .step-label { font-weight: 600; color: #f0f6fc; }

        /* ── Stmetric overrides ── */
        [data-testid="metric-container"] {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 12px 16px;
        }

        /* ── Section divider ── */
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #30363d 30%, #30363d 70%, transparent);
            margin: 22px 0;
        }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Component Functions
# ─────────────────────────────────────────────────────────────────────────────

def render_header(title: str, subtitle: str):
    """Render top forensic page banner with title and subtitle."""
    st.markdown(f"""
    <div class="forensic-header">
        <h2>🛡️ {title}</h2>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_verdict_badge(verdict: str, score: float, is_calibrated: bool = True):
    """Render animated color-coded verdict banner with probability display."""
    cal_label = "Calibrated" if is_calibrated else "Raw Probability"

    if "Manipulated" in verdict or "Fake" in verdict:
        badge_cls = "badge-fake"
        icon = "⚠️"
    elif "Authentic" in verdict or "Real" in verdict:
        badge_cls = "badge-real"
        icon = "🛡️"
    else:
        badge_cls = "badge-inconclusive"
        icon = "❓"

    st.markdown(f"""
    <div style="margin: 14px 0 18px;">
        <span class="{badge_cls}">{icon} {verdict.upper()}</span>
        <span style="margin-left: 16px; font-size: 0.9rem; color: #8b949e;">
            Manipulation Probability:&nbsp;
            <strong style="color: #f0f6fc; font-size: 1.0rem;">{score:.4f}</strong>
            &nbsp;<span style="font-size: 0.78rem; color: #484f58;">({cal_label})</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_hash_value(label: str, value: str):
    """Render a cryptographic hash with monospace styling."""
    st.markdown(f"""
    <div style="margin: 6px 0;">
        <span style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em;">{label}</span><br/>
        <span class="hash-value">{value}</span>
    </div>
    """, unsafe_allow_html=True)


def render_section_divider():
    """Render a subtle horizontal gradient divider."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def render_evidence_card(title: str, body_html: str):
    """Render an evidence card with hover effect."""
    st.markdown(f"""
    <div class="evidence-card">
        <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; color:#8b949e; margin-bottom:8px; font-weight:600;">{title}</div>
        {body_html}
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_overview():
    """Render the visual forensic pipeline overview steps."""
    steps = [
        ("Ingestion & Integrity Audit", "SHA-256 cryptographic hashing + full container metadata inspection."),
        ("Facial Region Extraction", "MediaPipe BlazeFace detection with 20% context margin expansion and OpenCV fallback."),
        ("Input Quality Gate", "Sharpness (Laplacian), brightness, and resolution diagnostics with automatic abstention below threshold."),
        ("Calibrated Neural Inference", "EfficientNet-B0 / ResNet-18 inference followed by Temperature Scaling to minimize ECE."),
        ("Reliability Decision Policy", "Three-tier outcome: <em>Likely Authentic</em>, <em>Potentially Manipulated</em>, or <em>Inconclusive</em>."),
        ("Visual Attribution (Grad-CAM)", "Class-discriminative saliency maps over convolutional feature maps."),
        ("Forensic Report Generation", "Tamper-evident PDF + machine-readable JSON evidence dossier."),
    ]
    html = ""
    for i, (label, desc) in enumerate(steps, 1):
        html += f"""
        <div class="pipeline-step">
            <div class="step-number">{i}</div>
            <div class="step-text"><span class="step-label">{label}</span><br/>{desc}</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_legal_disclaimer():
    """Render the mandatory forensic decision-support legal disclaimer."""
    st.markdown("""
    <div class="forensic-notice">
        <strong>⚖️ Forensic Decision-Support Notice:</strong> DeepTrace is an academic research platform, not a
        court-certified or legally admissible authenticity verifier. Model predictions reflect probabilistic patterns
        in convolutional feature representations; Grad-CAM overlays explain internal neural activations rather than
        verified manipulation ground-truth masks. Missing EXIF tags or media recompression are standard artifacts
        in web-distributed content and do not independently constitute evidence of tampering.
        All findings must be corroborated with independent chain-of-custody provenance evidence.
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(metrics: List[Dict[str, Any]]):
    """Render a row of metric cards.

    Args:
        metrics: List of dicts with keys: label, value, help (optional), delta (optional).
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        col.metric(
            label=m["label"],
            value=m["value"],
            delta=m.get("delta"),
            help=m.get("help")
        )
