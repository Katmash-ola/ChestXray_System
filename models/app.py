import os
# FORCE CPU EXECUTION
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CXR Diagnostic Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Clinical Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b1120;
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0f1929;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stHeader"] {
    background-color: #0b1120;
}

/* ── Header banner ── */
.clinical-header {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 50%, #0d2137 100%);
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.clinical-header h1 {
    color: #f0f9ff;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.clinical-header p {
    color: #93c5fd;
    margin: 4px 0 0 0;
    font-size: 0.9rem;
}
.header-icon { font-size: 3rem; }

/* ── Section cards ── */
.section-card {
    background: #0f1e30;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.section-title {
    color: #60a5fa;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
}

/* ── Diagnosis banner ── */
.dx-critical {
    background: linear-gradient(135deg, #3b0a0a, #7f1d1d);
    border: 2px solid #ef4444;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.dx-normal {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 2px solid #22c55e;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.dx-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.dx-critical .dx-label { color: #fca5a5; }
.dx-normal  .dx-label { color: #86efac; }
.dx-disease {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}
.dx-conf {
    font-size: 0.85rem;
    margin-top: 4px;
}
.dx-critical .dx-conf { color: #fca5a5; }
.dx-normal  .dx-conf { color: #86efac; }

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-tile {
    flex: 1;
    background: #0d2137;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.metric-tile .m-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 4px;
}
.metric-tile .m-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f9ff;
}

/* ── Additional findings ── */
.finding-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1e2d40;
}
.finding-row:last-child { border-bottom: none; }
.finding-name { color: #cbd5e1; font-size: 0.9rem; }
.finding-conf { color: #60a5fa; font-weight: 600; font-size: 0.9rem; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #1e3a5f !important;
    border-radius: 10px !important;
    background: #0d1f30 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563eb !important;
}

/* ── Sidebar items ── */
.sidebar-section {
    background: #0d2137;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.sidebar-section p, .sidebar-section li {
    color: #94a3b8;
    font-size: 0.84rem;
    line-height: 1.6;
}
.sidebar-label {
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}

/* ── Heatmap label ── */
.heatmap-note {
    background: #0d2137;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    color: #93c5fd;
    font-size: 0.83rem;
    margin-top: 10px;
}

/* ── Footer ── */
.footer-bar {
    border-top: 1px solid #1e3a5f;
    padding-top: 14px;
    margin-top: 28px;
    color: #475569;
    font-size: 0.78rem;
    text-align: center;
}

/* ── Streamlit overrides ── */
div[data-testid="stMetricValue"] { color: #f0f9ff !important; }
h3 { color: #e2e8f0 !important; }
.stSlider label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODEL_PATH = "best_chest_xray_model.keras"
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia", "Normal"
]

DISEASE_INFO = {
    "Atelectasis":       "Partial or complete lung collapse. Can occur after surgery or due to blockage.",
    "Cardiomegaly":      "Enlarged heart. May indicate heart failure, hypertension, or cardiomyopathy.",
    "Effusion":          "Fluid accumulation around the lung. Associated with infection, heart failure, or cancer.",
    "Infiltration":      "Abnormal substance (fluid, pus, blood) within lung tissue.",
    "Mass":              "A lesion greater than 3 cm. Requires urgent follow-up to rule out malignancy.",
    "Nodule":            "A rounded lesion less than 3 cm. May indicate early-stage tumour or infection.",
    "Pneumonia":         "Lung infection causing inflammation. Can be bacterial, viral, or fungal in origin.",
    "Pneumothorax":      "Air in the pleural space causing lung collapse. Can be life-threatening.",
    "Consolidation":     "Lung tissue filled with fluid instead of air. Common in pneumonia.",
    "Edema":             "Fluid build-up in the lungs. Often associated with heart failure.",
    "Emphysema":         "Damage to air sacs in the lungs. Common in long-term smokers.",
    "Fibrosis":          "Scarring of lung tissue reducing breathing capacity.",
    "Pleural Thickening":"Thickening of the lung lining. Can follow infection or asbestos exposure.",
    "Hernia":            "Abdominal organs protruding into the chest cavity through the diaphragm.",
    "Normal":            "No significant pathology detected. Lung fields appear clear."
}

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_diagnostic_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found.")
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# ─────────────────────────────────────────────
# GRAD-CAM — HEATMAP OVERLAY
# ─────────────────────────────────────────────
def generate_gradcam_heatmap(img_array, model, class_index):
    """Compute Grad-CAM activation map for the given class."""
    base_model = model.layers[0]
    # Build head sub-model
    head_input = tf.keras.Input(shape=base_model.output_shape[1:])
    x = head_input
    for layer in model.layers[1:]:
        x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_array, training=False)
        tape.watch(conv_outputs)
        preds = head_model(conv_outputs, training=False)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.45, colormap="jet"):
    """
    Overlay a Grad-CAM heatmap onto the original X-ray as a
    semi-transparent colour overlay — the standard clinical
    heatmap visualisation used in published medical AI research.
    """
    # Convert original to RGB numpy
    img_rgb = np.array(original_image.convert("RGB"))
    h, w = img_rgb.shape[:2]

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap (jet: blue=low, green=mid, red=high activation)
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]          # drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)  # scale to 0-255

    # Blend: original X-ray + heatmap
    img_float   = img_rgb.astype(np.float32)
    heat_float  = heatmap_colored.astype(np.float32)
    blended     = (1 - alpha) * img_float + alpha * heat_float
    blended     = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def add_colorbar(fig, ax, colormap="jet"):
    """Add a Grad-CAM intensity colorbar to the matplotlib figure."""
    sm = plt.cm.ScalarMappable(
        cmap=colormap,
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Activation Intensity", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size:2.4rem;">🫁</div>
            <div style="color:#60a5fa; font-weight:700; font-size:1rem; margin-top:4px;">
                CXR Diagnostic Assistant
            </div>
            <div style="color:#475569; font-size:0.75rem; margin-top:2px;">
                Powered by ResNet50 + Grad-CAM
            </div>
        </div>
        <hr style="border-color:#1e3a5f; margin: 12px 0;">
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">⚙️ Analysis Settings</div>', unsafe_allow_html=True)
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05, max_value=0.95,
            value=0.30, step=0.05,
            help="Predictions above this value are reported as findings."
        )

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">🌡️ Heatmap Settings</div>', unsafe_allow_html=True)
        heatmap_alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.2, max_value=0.8,
            value=0.45, step=0.05,
            help="How strongly the heatmap is blended over the X-ray."
        )
        colormap_choice = st.selectbox(
            "Colour Map",
            ["jet", "hot", "plasma", "inferno"],
            index=0,
            help="Colour palette used to display activation regions."
        )

        st.markdown('<hr style="border-color:#1e3a5f; margin: 16px 0;">', unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-label">ℹ️ How to Use</div>
        <div class="sidebar-section">
            <p>1. Upload a frontal chest X-ray image (JPG or PNG).</p>
            <p>2. The model analyses the image across 15 disease categories.</p>
            <p>3. The Grad-CAM heatmap shows which region of the X-ray drove the primary prediction.</p>
            <p>4. Adjust the confidence threshold to control sensitivity.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-label">🎨 Heatmap Guide</div>
        <div class="sidebar-section">
            <p>🔴 <b>Red / Yellow</b> — High activation (strongest disease signal)</p>
            <p>🟢 <b>Green</b> — Moderate activation</p>
            <p>🔵 <b>Blue</b> — Low activation (background regions)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <hr style="border-color:#1e3a5f; margin: 12px 0;">
        <div style="color:#475569; font-size:0.72rem; text-align:center; line-height:1.6;">
            Sol Plaatje University<br>
            NCPS730 Capstone Project<br>
            Mashala Katlego · 202470697
        </div>
        """, unsafe_allow_html=True)

    return threshold, heatmap_alpha, colormap_choice

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    threshold, heatmap_alpha, colormap_choice = render_sidebar()

    # ── Header ──────────────────────────────
    st.markdown("""
    <div class="clinical-header">
        <div class="header-icon">🫁</div>
        <div>
            <h1>Chest X-Ray Diagnostic Assistant</h1>
            <p>AI-powered multi-label pathology detection · ResNet50 Transfer Learning · Grad-CAM Explainability</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ───────────────────────────
    st.info(
        "⚕️ **Clinical Disclaimer:** This system is a research prototype developed as part of an academic "
        "capstone project. It is intended for **demonstration and educational purposes only** and must not "
        "be used as a substitute for professional medical diagnosis.",
        icon=None
    )

    # ── Load model ───────────────────────────
    with st.spinner("Initialising diagnostic model..."):
        model = load_diagnostic_model()
    if model is None:
        st.stop()

    # ── Upload zone ──────────────────────────
    st.markdown('<div class="section-title">📂 Image Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a frontal chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Accepted formats: JPEG, PNG. Recommended: frontal PA or AP view."
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center; padding: 48px 0; color:#334155;">
            <div style="font-size:3rem; margin-bottom:12px;">📤</div>
            <div style="font-size:1rem; color:#475569;">Upload a chest X-ray image to begin analysis</div>
            <div style="font-size:0.8rem; color:#334155; margin-top:6px;">
                Supported formats: JPG · PNG
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    original_image = Image.open(uploaded_file)

    # ── Run inference ────────────────────────
    with st.spinner("Analysing scan..."):
        img_array        = preprocess_image(original_image)
        raw_predictions  = model.predict(img_array, verbose=0)[0]

    detected = [
        (CLASS_NAMES[i], float(raw_predictions[i]), i)
        for i in range(len(CLASS_NAMES))
        if raw_predictions[i] >= threshold
    ]
    detected.sort(key=lambda x: x[1], reverse=True)

    primary_name = detected[0][0] if detected else "No Finding"
    primary_conf = detected[0][1] if detected else 0.0
    primary_idx  = detected[0][2] if detected else -1
    is_normal    = (primary_name == "Normal")

    # ════════════════════════════════════════
    # ROW 1: Original X-ray  |  Diagnosis
    # ════════════════════════════════════════
    col_img, col_dx = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-title">🩻 Uploaded X-Ray</div>', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True, caption="Original uploaded image")

    with col_dx:
        st.markdown('<div class="section-title">🏥 Diagnostic Report</div>', unsafe_allow_html=True)

        # Primary diagnosis banner
        if detected:
            if is_normal:
                st.markdown(f"""
                <div class="dx-normal">
                    <div class="dx-label">✅ Primary Assessment</div>
                    <div class="dx-disease">{primary_name}</div>
                    <div class="dx-conf">Confidence: {primary_conf:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="dx-critical">
                    <div class="dx-label">⚠️ Primary Finding</div>
                    <div class="dx-disease">{primary_name}</div>
                    <div class="dx-conf">Confidence: {primary_conf:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dx-normal">
                <div class="dx-label">✅ Assessment</div>
                <div class="dx-disease">No Findings Detected</div>
                <div class="dx-conf">No conditions exceeded the confidence threshold</div>
            </div>
            """, unsafe_allow_html=True)

        # Metric tiles
        total_detected = len(detected)
        top_score      = f"{raw_predictions.max():.1%}"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="m-label">Conditions Found</div>
                <div class="m-value">{total_detected}</div>
            </div>
            <div class="metric-tile">
                <div class="m-label">Peak Confidence</div>
                <div class="m-value">{top_score}</div>
            </div>
            <div class="metric-tile">
                <div class="m-label">Classes Evaluated</div>
                <div class="m-value">15</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Clinical description of primary finding
        if detected and primary_name in DISEASE_INFO:
            st.markdown(f"""
            <div style="background:#0d2137; border:1px solid #1e3a5f; border-left: 3px solid #2563eb;
                        border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:14px;">
                <div style="color:#60a5fa; font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
                    Clinical Note
                </div>
                <div style="color:#cbd5e1; font-size:0.88rem; line-height:1.6;">
                    {DISEASE_INFO[primary_name]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Additional findings
        if len(detected) > 1:
            st.markdown('<div class="section-title" style="margin-top:8px;">📋 Additional Findings</div>',
                        unsafe_allow_html=True)
            for name, conf, _ in detected[1:]:
                st.markdown(f"""
                <div class="finding-row">
                    <span class="finding-name">{name}</span>
                    <span class="finding-conf">{conf:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    # ROW 2: Probability Chart
    # ════════════════════════════════════════
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Prediction Probabilities — All 15 Classes</div>',
                unsafe_allow_html=True)

    top_idx    = np.argsort(raw_predictions)[-10:][::-1]
    top_names  = [CLASS_NAMES[i] for i in top_idx]
    top_scores = [raw_predictions[i] for i in top_idx]
    colors     = []
    for i, name in zip(top_idx, top_names):
        if name == "Normal":
            colors.append("#22c55e")
        elif raw_predictions[i] >= threshold:
            colors.append("#ef4444")
        else:
            colors.append("#1e3a5f")

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.barh(top_names[::-1], top_scores[::-1], color=colors[::-1],
                   height=0.6, edgecolor="none")

    for bar, score in zip(bars, top_scores[::-1]):
        ax.text(score + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{score:.1%}", va="center", ha="left",
                color="white", fontweight="bold", fontsize=9.5)

    ax.axvline(x=threshold, color="#3b82f6", linestyle="--",
               linewidth=1.5, label=f"Threshold ({threshold:.0%})")
    ax.set_xlim(0, 1.18)
    ax.tick_params(axis="y", colors="#94a3b8", labelsize=10)
    ax.tick_params(axis="x", colors="#64748b", labelsize=9)
    ax.spines["bottom"].set_color("#1e3a5f")
    ax.spines["left"].set_color("#1e3a5f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9,
              labelcolor="white", framealpha=0.2,
              edgecolor="#1e3a5f")
    fig.patch.set_facecolor("#0f1929")
    ax.set_facecolor("#0f1929")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ════════════════════════════════════════
    # ROW 3: Grad-CAM Heatmap
    # ════════════════════════════════════════
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 Explainable AI — Grad-CAM Activation Heatmap</div>',
                unsafe_allow_html=True)

    if primary_idx == -1 or is_normal:
        st.markdown("""
        <div style="background:#0d2137; border:1px solid #1e3a5f; border-radius:8px;
                    padding:20px; text-align:center; color:#64748b;">
            ℹ️ Grad-CAM localisation is not generated for Normal assessments or when no
            conditions are detected above the confidence threshold.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner(f"Generating Grad-CAM heatmap for {primary_name}..."):
            heatmap      = generate_gradcam_heatmap(img_array, model, primary_idx)
            overlay_img  = overlay_heatmap_on_image(
                original_image, heatmap,
                alpha=heatmap_alpha,
                colormap=colormap_choice
            )

        hm_col1, hm_col2, hm_col3 = st.columns([1, 1, 1], gap="large")

        with hm_col1:
            st.markdown('<div style="text-align:center; color:#64748b; font-size:0.8rem; '
                        'margin-bottom:6px;">Original X-Ray</div>', unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)

        with hm_col2:
            st.markdown(f'<div style="text-align:center; color:#60a5fa; font-size:0.8rem; '
                        f'margin-bottom:6px;">Grad-CAM Overlay — {primary_name}</div>',
                        unsafe_allow_html=True)
            st.image(overlay_img, use_container_width=True)

        with hm_col3:
            st.markdown('<div style="text-align:center; color:#64748b; font-size:0.8rem; '
                        'margin-bottom:6px;">Activation Intensity Scale</div>',
                        unsafe_allow_html=True)
            # Standalone colorbar
            fig_cb, ax_cb = plt.subplots(figsize=(1.2, 5))
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            ax_cb.imshow(gradient[::-1], aspect="auto",
                         cmap=colormap_choice, extent=[0, 1, 0, 1])
            ax_cb.set_xticks([])
            ax_cb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_cb.set_yticklabels(["Low", "", "Med", "", "High"],
                                   color="white", fontsize=9)
            ax_cb.spines[:].set_visible(False)
            fig_cb.patch.set_facecolor("#0f1929")
            ax_cb.set_facecolor("#0f1929")
            st.pyplot(fig_cb, use_container_width=True)
            plt.close(fig_cb)

        st.markdown(f"""
        <div class="heatmap-note">
            🔬 <b>Reading the heatmap:</b> Red and yellow regions indicate where the model focused most
            strongly when predicting <b>{primary_name}</b>. These are the anatomical areas that had the
            greatest influence on the diagnosis. Blue regions had little to no influence.
            The overlay opacity can be adjusted in the sidebar.
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════
    st.markdown("""
    <div class="footer-bar">
        ⚕️ For educational and demonstration purposes only · Not for clinical use ·
        NCPS730 Capstone Project · Sol Plaatje University · Mashala Katlego · 202470697
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()