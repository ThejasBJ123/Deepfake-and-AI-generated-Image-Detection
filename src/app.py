#!/usr/bin/env python3
"""
src/app.py
-----------
Streamlit Web Application for Deepfake and AI-Generated Image Detection.

Interactive forensic investigation dashboard providing:
1. Real-time Deep Learning classification (Real vs. Fake / AI-Generated).
2. Explainable AI (XAI) Forensic Suite:
   - Error Level Analysis (ELA) JPEG compression discrepancy mapping
   - 2D Fast Fourier Transform (FFT) periodic generative artifact spectrum
   - High-pass sensor noise residual analysis (PRNU)
3. Quantitative Evaluation reports, confusion matrices, and ROC curves.
"""

import io
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

# Import local forensic engine
from forensics import compute_ela, compute_fft_spectrum, compute_noise_residual

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deepfake & AI Image Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-num {
        font-size: 24px;
        font-weight: 700;
        color: #1565c0;
    }
    .metric-lbl {
        font-size: 12px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .verdict-fake {
        background-color: #ffebee;
        border: 2px solid #e53935;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        color: #c62828;
        font-weight: bold;
        font-size: 22px;
    }
    .verdict-real {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        color: #2e7d32;
        font-weight: bold;
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model Loading Helpers
# ---------------------------------------------------------
DEFAULT_MODEL_DIR = Path("outputs/deepfake_detector/weights")


def discover_available_models():
    """Discover available classification model checkpoints."""
    models = []
    if DEFAULT_MODEL_DIR.exists():
        for pt in DEFAULT_MODEL_DIR.glob("*.pt"):
            models.append(str(pt))
    for pt in Path(".").glob("*cls*.pt"):
        models.append(str(pt))
    if not models:
        models.append("yolov8n-cls.pt")
    return sorted(list(set(models)))


@st.cache_resource
def load_detection_model(model_path_str: str):
    """Load and cache YOLO classification model."""
    return YOLO(model_path_str)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Deepfake Detector")
    st.caption("AI-Generated & Deepfake Forensics")
    st.divider()

    available_models = discover_available_models()
    selected_model_path = st.selectbox(
        "Model Checkpoint",
        options=available_models,
        index=0 if available_models else 0,
        help="Select fine-tuned detector weights",
    )

    decision_threshold = st.slider(
        "Fake Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold required to flag an image as AI-Generated/Fake",
    )

    imgsz = st.select_slider(
        "Processing Resolution",
        options=[224, 256, 384, 512],
        value=224,
        help="Input resolution for model forward pass",
    )

    st.divider()
    st.markdown("### Forensic Settings")
    ela_quality = st.slider("ELA JPEG Quality", 50, 98, 90, 2)
    ela_scale = st.slider("ELA Intensity Scale", 5, 30, 15, 1)

    st.divider()
    st.markdown("### Hardware Accelerator")
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    st.write(f"**Device:** {device_name}")
    st.write(f"**PyTorch:** `{torch.__version__}`")
    st.write(f"**CUDA:** `{'Active' if cuda_avail else 'Unavailable'}`")


# ---------------------------------------------------------
# Load Model Instance
# ---------------------------------------------------------
try:
    model = load_detection_model(selected_model_path)
except Exception as e:
    st.error(f"Error loading model from `{selected_model_path}`: {e}")
    st.stop()


# ---------------------------------------------------------
# Main Page Title & Tabs
# ---------------------------------------------------------
st.title("🛡️ Deepfake & AI-Generated Image Detection Platform")
st.markdown(
    "Detect authentic human photographs vs. synthetically generated / manipulated images "
    "using Deep Neural Network feature classification and Explainable AI (XAI) computer vision forensics."
)

tab_inspect, tab_batch, tab_metrics, tab_theory = st.tabs(
    [
        "🔍 Single Image Inspector",
        "📁 Sample Gallery & Benchmarks",
        "📊 Model Evaluation & Metrics",
        "🧠 Forensic Theory & Architecture",
    ]
)


# ---------------------------------------------------------
# TAB 1: Single Image Inspector
# ---------------------------------------------------------
with tab_inspect:
    col_upload, col_btn = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload image to analyze (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload any real photograph, deepfake face swap, or AI-generated image (Midjourney, Stable Diffusion, StyleGAN)",
        )
    with col_btn:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔬 Run Forensic Analysis", use_container_width=True, type="primary")

    if uploaded_file is not None:
        raw_pil = Image.open(uploaded_file).convert("RGB")
        w_orig, h_orig = raw_pil.size

        # Deep Learning Inference
        t0 = time.perf_counter()
        results = model.predict(source=raw_pil, imgsz=imgsz, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        res = results[0]
        probs = res.probs.data.cpu().numpy()
        model_classes = model.names

        fake_idx = 1
        real_idx = 0
        for idx, name in model_classes.items():
            if "fake" in name.lower():
                fake_idx = idx
            elif "real" in name.lower():
                real_idx = idx

        fake_prob = float(probs[fake_idx])
        real_prob = float(probs[real_idx])
        is_fake = fake_prob >= decision_threshold

        # Forensic Analysis
        ela_map, ela_score = compute_ela(raw_pil, quality=ela_quality, scale=ela_scale)
        fft_map, hf_ratio = compute_fft_spectrum(raw_pil)
        noise_map, noise_var = compute_noise_residual(raw_pil)

        # Verdict Header
        st.markdown("<br>", unsafe_allow_html=True)
        if is_fake:
            st.markdown(
                f'<div class="verdict-fake">🚨 VERDICT: FAKE / AI-GENERATED IMAGE (Confidence: {fake_prob:.1%})</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-real">✅ VERDICT: AUTHENTIC / REAL PHOTOGRAPH (Confidence: {real_prob:.1%})</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # Key Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-lbl">Fake Probability</div>'
                f'<div class="metric-num" style="color:{"#e53935" if fake_prob > 0.5 else "#6c757d"}">{fake_prob:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-lbl">Real Probability</div>'
                f'<div class="metric-num" style="color:{"#43a047" if real_prob > 0.5 else "#6c757d"}">{real_prob:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-lbl">ELA Discrepancy</div>'
                f'<div class="metric-num">{ela_score:.2f}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-box"><div class="metric-lbl">FFT HF Energy</div>'
                f'<div class="metric-num">{hf_ratio:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f'<div class="metric-box"><div class="metric-lbl">Inference Latency</div>'
                f'<div class="metric-num">{latency_ms:.1f} ms</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Multi-Panel Forensic Visuals
        st.subheader("🔬 Multi-Modal Forensic Analysis Panels")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)

        with p_col1:
            st.markdown("**1. Original Input**")
            st.image(raw_pil, use_container_width=True, caption=f"Resolution: {w_orig}x{h_orig}px")

        with p_col2:
            st.markdown("**2. Error Level Analysis (ELA)**")
            st.image(
                ela_map,
                use_container_width=True,
                caption=f"Compression Variance: {ela_score:.1f}",
            )
            st.caption(
                "Bright, inconsistent regions highlight synthetic splicing or digital re-compression differences."
            )

        with p_col3:
            st.markdown("**3. 2D-FFT Power Spectrum**")
            st.image(
                fft_map,
                use_container_width=True,
                caption=f"High-Freq Energy: {hf_ratio:.1%}",
            )
            st.caption(
                "Detects periodic grid/checkerboard peaks created by GAN upsampling and diffusion decoders."
            )

        with p_col4:
            st.markdown("**4. Sensor Noise Residual**")
            st.image(
                noise_map,
                use_container_width=True,
                caption=f"Noise Variance: {noise_var:.1f}",
            )
            st.caption(
                "Authentic photos display natural physical sensor PRNU shot noise; AI images show synthetic uniformity."
            )

        # Tabular Forensic Report
        st.divider()
        st.subheader("📋 Forensic Evidence Breakdown")
        breakdown_data = [
            {
                "Forensic Indicator": "Deep Convolutional Neural Feature Score",
                "Metric / Value": f"Fake: {fake_prob:.2%} | Real: {real_prob:.2%}",
                "Observation": "Deep visual feature embedding confirms high synthetic correlation" if is_fake else "Visual feature embedding aligns with natural photographic manifold",
                "Weight": "High (Primary Classifier)",
            },
            {
                "Forensic Indicator": "Error Level Analysis (ELA)",
                "Metric / Value": f"Mean Discrepancy = {ela_score:.2f}",
                "Observation": "Localized compression error spikes detected across face/object boundary" if ela_score > 6.0 else "Uniform compression dissipation consistent with single-source camera file",
                "Weight": "Medium (Spatial Consistency)",
            },
            {
                "Forensic Indicator": "2D-FFT Frequency Domain Spectrum",
                "Metric / Value": f"High-Frequency Energy = {hf_ratio:.2%}",
                "Observation": "Anomalous periodic frequency spikes or radial symmetry break detected" if hf_ratio > 0.35 else "Smooth radial 1/f decay typical of natural photographic optics",
                "Weight": "Medium (Spectral Artifacts)",
            },
            {
                "Forensic Indicator": "PRNU Camera Sensor Noise Residual",
                "Metric / Value": f"Variance = {noise_var:.2f}",
                "Observation": "Low or unnatural sensor noise variance; indicative of neural rendering" if noise_var < 30.0 else "Organic shot noise pattern observed",
                "Weight": "Supporting (Hardware Footprint)",
            },
        ]
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)

        # Download Report Panel
        st.divider()
        report_buf = io.BytesIO()
        # Create consolidated image
        c_h, c_w = 300, 300
        r1 = cv2.resize(np.array(raw_pil), (c_w, c_h))
        r2 = cv2.resize(ela_map, (c_w, c_h))
        r3 = cv2.resize(fft_map, (c_w, c_h))
        r4 = cv2.resize(noise_map, (c_w, c_h))
        comb = np.hstack([r1, r2, r3, r4])
        comb_pil = Image.fromarray(comb)
        comb_pil.save(report_buf, format="JPEG", quality=95)

        st.download_button(
            label="📥 Download Forensic Evidence Panel (JPEG)",
            data=report_buf.getvalue(),
            file_name=f"forensic_report_{uploaded_file.name}",
            mime="image/jpeg",
        )
    else:
        st.info("👆 Please upload an image file above to begin forensic deepfake detection.")


# ---------------------------------------------------------
# TAB 2: Sample Gallery & Benchmarks
# ---------------------------------------------------------
with tab_batch:
    st.subheader("🖼️ Benchmark Test Gallery")
    st.write("Test pre-loaded real photographs and AI-generated sample images.")

    test_dir = Path("test_images")
    sample_files = [p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}] if test_dir.exists() else []

    if sample_files:
        chosen_sample = st.selectbox("Select Benchmark Image:", sample_files, format_func=lambda x: x.name)
        if st.button("Run Detection on Selected Sample", type="primary"):
            sample_pil = Image.open(chosen_sample).convert("RGB")
            t0 = time.perf_counter()
            s_results = model.predict(source=sample_pil, imgsz=imgsz, verbose=False)
            s_lat = (time.perf_counter() - t0) * 1000.0

            s_res = s_results[0]
            s_probs = s_res.probs.data.cpu().numpy()
            s_model_classes = model.names

            s_fake_idx = 1
            s_real_idx = 0
            for idx, name in s_model_classes.items():
                if "fake" in name.lower():
                    s_fake_idx = idx
                elif "real" in name.lower():
                    s_real_idx = idx

            s_fake_prob = float(s_probs[s_fake_idx])
            s_real_prob = float(s_probs[s_real_idx])
            s_is_fake = s_fake_prob >= decision_threshold

            if s_is_fake:
                st.markdown(
                    f'<div class="verdict-fake">🚨 VERDICT: FAKE / AI-GENERATED ({s_fake_prob:.1%})</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict-real">✅ VERDICT: AUTHENTIC REAL PHOTO ({s_real_prob:.1%})</div>',
                    unsafe_allow_html=True,
                )

            s_ela, s_ela_score = compute_ela(sample_pil)
            s_fft, s_hf = compute_fft_spectrum(sample_pil)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(sample_pil, caption=f"Input: {chosen_sample.name}", use_container_width=True)
            with c2:
                st.image(s_ela, caption=f"ELA Map (Discrepancy: {s_ela_score:.1f})", use_container_width=True)
            with c3:
                st.image(s_fft, caption=f"2D-FFT Spectrum (HF: {s_hf:.1%})", use_container_width=True)
    else:
        st.warning("No sample files found in test_images/. Run dataset preparation to populate test samples.")


# ---------------------------------------------------------
# TAB 3: Model Evaluation & Performance Metrics
# ---------------------------------------------------------
with tab_metrics:
    st.subheader("📊 Quantitative Model Performance & Test Split Results")
    eval_dir = Path("outputs/evaluation/test_set")
    summary_json = eval_dir / "metrics_summary.json"
    summary_txt = eval_dir / "metrics_summary.txt"

    if summary_json.exists():
        try:
            m_data = json.loads(summary_json.read_text(encoding="utf-8"))
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Overall Accuracy", f"{m_data.get('accuracy', 0):.2%}")
            e2.metric("ROC AUC Score", f"{m_data.get('roc_auc', 0):.2%}")
            e3.metric("Fake Detection F1-Score", f"{m_data.get('f1_fake', 0):.4f}")
            e4.metric("Inference Throughput", f"{m_data.get('fps', 0):.1f} FPS")

            st.markdown("#### Evaluation Visualizations")
            p_cm = eval_dir / "confusion_matrix.png"
            p_roc = eval_dir / "roc_curve.png"

            col_cm, col_roc = st.columns(2)
            if p_cm.exists():
                col_cm.image(str(p_cm), caption="Confusion Matrix", use_container_width=True)
            if p_roc.exists():
                col_roc.image(str(p_roc), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)

            if summary_txt.exists():
                st.markdown("#### Detailed Quantitative Report")
                st.code(summary_txt.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Error parsing evaluation summary: {e}")
    else:
        st.info("Run `python src/evaluate.py --split test` to generate quantitative metrics and view them here.")


# ---------------------------------------------------------
# TAB 4: Forensic Theory & Architecture
# ---------------------------------------------------------
with tab_theory:
    st.subheader("🧠 System Architecture & Forensic Theory")
    st.markdown(
        """
    ### 1. Generation Paradigms:
    * **Generative Adversarial Networks (StyleGAN / ProGAN):** Use transpose convolution and bilinear upsampling layers, producing subtle periodic checkerboard grid artifacts in the frequency domain.
    * **Latent Diffusion Models (Stable Diffusion / Midjourney / DALL-E):** Operate in latent space through continuous denoising, resulting in ultra-smooth high-frequency texture transitions and synthetic edge gradients.
    * **Face Swap / Deepfakes:** Replace face identities with blend seams, creating localized compression inconsistencies between face boundary and outer head.

    ### 2. Detection Methodology:
    1. **Deep Convolutional Feature Backbone:**
       - Extracts high-level spatial semantic features and subtle micro-texture inconsistencies.
       - Optimized via binary cross-entropy loss with Automatic Mixed Precision (AMP).
    2. **Error Level Analysis (ELA):**
       - Digital images undergo uniform loss during JPEG encoding.
       - Manipulated or AI-rendered regions show distinct compression decay compared to physical sensor capture.
    3. **2D Fast Fourier Transform (FFT) Power Spectrum:**
       - Computes $F(u, v) = \iint f(x, y) e^{-j 2\pi (ux + vy)} dx dy$.
       - Exposes non-natural high-frequency peaks and symmetry breaks caused by neural deconvolution.
    4. **Physical Camera Sensor Noise (PRNU):**
       - Physical camera sensors leave microscopic Photo-Response Non-Uniformity patterns.
       - AI generation creates sterile or synthetic noise lacking physical hardware signatures.
    """
    )
