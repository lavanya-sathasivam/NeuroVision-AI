import streamlit as st
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt

from utils import load_model_and_labels, preprocess_image, predict_image
from gradcam import generate_gradcam, overlay_heatmap

try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Page content */
    .css-18e3th9 { padding-top: 1rem; }
    .main > div { padding: 1rem 1.5rem; }

    /* Card style */
    .app-card {
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        padding: 1rem;
        background: #ffffff;
        margin-bottom: 1rem;
    }
    .app-h2 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #012a4a;
        margin-bottom: 0.25rem;
    }
    .app-text {
        color: #334e68;
        margin-bottom: 0.75rem;
    }
    .card-border {
        border-left: 5px solid #0077b6;
        padding-left: 0.9rem;
    }
    .small-muted {
        color: #627d98;
        font-size: 0.9rem;
    }
    .big-highlight {
        color: #03396c;
        font-size: 1.3rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Clinical Intake Panel
st.sidebar.markdown("## 🧾 Clinical Intake Panel", unsafe_allow_html=True)
patient_name = st.sidebar.text_input("Patient Name", value="John Doe")
patient_id = st.sidebar.text_input("Patient ID", value="NV-2026-001")
uploaded_file = st.sidebar.file_uploader(
    "Upload MRI Image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "#### ⚠️ Disclaimer\n"
    "This tool is for research purposes only and not a substitute for professional diagnosis.",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_resources():
    return load_model_and_labels()

model, class_labels = load_resources()
CLASS_ORDER = ["glioma", "meningioma", "pituitary", "no tumor"]

st.markdown("# 🧠 NeuroVision AI", unsafe_allow_html=True)
st.markdown("### Brain MRI Tumor Detection · EfficientNet · Grad-CAM explainability")
st.markdown(
    "<div class='small-muted'>"
    "Upload an MRI image and receive a diagnosis estimate with probability and explainability."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not uploaded_file:
    st.warning("Please upload an MRI image from the sidebar to begin classification.")
    st.stop()

# Load image
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Failed to load the image. Ensure this is a valid JPG/PNG image.")
    st.stop()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

with st.spinner("🔍 Analyzing MRI scan with EfficientNet... Please wait"):
    label, confidence, preds = predict_image(img_bgr, model, class_labels)
    idx_proba = {class_labels[i]: float(preds[i]) for i in range(len(preds))}
    ordered_probs = [
        {"class": c, "probability": idx_proba.get(c, 0.0)} for c in CLASS_ORDER
    ]
    sorted_probs = sorted(ordered_probs, key=lambda x: x["probability"], reverse=True)

    processed_img = preprocess_image(img_bgr)
    heatmap = generate_gradcam(model, processed_img)
    overlay = overlay_heatmap(cv2.resize(img_bgr, (224, 224)), heatmap)

st.success("✅ Prediction complete")

pred_color = "#2d6a4f" if label == "no tumor" else "#9d0208"
pred_desc = "No tumor detected (normal)" if label == "no tumor" else f"{label.title()} tumor detected"

# Top-level result cards
top_cols = st.columns([1.5, 1])
with top_cols[0]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='app-h2'>📊 Prediction Summary</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='big-highlight' style='color:{pred_color};'>{pred_desc}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='app-text'>Predicted class: <strong>{label.title()}</strong><br>"
        f"Confidence: <strong>{confidence * 100:.2f}%</strong></div>",
        unsafe_allow_html=True,
    )
    st.progress(int(confidence * 100))
    st.markdown(
        "<div class='small-muted'>"
        "Confidence represents model certainty based on output probabilities."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with top_cols[1]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='app-h2'>📈 Class Probability Distribution</div>", unsafe_allow_html=True)
    df_chart = pd.DataFrame(sorted_probs)
    fig, ax = plt.subplots(figsize=(5, 3))
    bar_colors = [
        "#ff4d6d" if row["class"] != "no tumor" else "#2d6a4f"
        for _, row in df_chart.iterrows()
    ]
    highlight = [row["class"] == label for _, row in df_chart.iterrows()]
    for i, (class_name, prob, h) in enumerate(zip(df_chart["class"], df_chart["probability"], highlight)):
        c = "#023e8a" if h else ("#249e76" if class_name == "no tumor" else "#d00000")
        ax.barh(class_name, prob, color=c, alpha=0.85, height=0.55)
        ax.text(prob + 0.01, i, f"{prob * 100:.1f}%", va="center", fontsize=9, color="#0b0c10")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔍 Grad-CAM Explainability")
grad_cols = st.columns(3)
with grad_cols[0]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### Original MRI")
    st.image(img_rgb, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with grad_cols[1]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### Grad-CAM Heatmap")
    heatmap_vis = np.uint8(255 * heatmap)
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
    st.image(heatmap_vis, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with grad_cols[2]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### Overlay (MRI + Grad-CAM)")
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    st.image(overlay_rgb, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📄 Downloadable Medical Report")

def create_pdf_report(
    patient_name,
    patient_id,
    original_image_rgb,
    predicted_label,
    confidence,
    probability_data,
    timestamp,
):
    if not _FPDF_AVAILABLE:
        raise RuntimeError("FPDF library not available. Install with `pip install fpdf`.")
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "NeuroVision AI - MRI Tumor Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Date/Time: {timestamp}", ln=True)
    pdf.cell(0, 6, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 6, f"Patient ID: {patient_id}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 6, "Prediction Results", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Predicted Class: {predicted_label.title()}", ln=True)
    pdf.cell(0, 6, f"Confidence: {confidence * 100:.2f}%", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Class Probabilities", ln=True)
    pdf.set_font("Arial", "", 11)
    for cls, prob in probability_data.items():
        pdf.cell(0, 5, f"{cls.title():<12} {prob * 100:.2f}%", ln=True)

    pil_img = Image.fromarray(original_image_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    image_path = "temp_report_image.png"
    with open(image_path, "wb") as f:
        f.write(buffer.read())

    pdf.ln(4)
    pdf.image(image_path, x=30, w=150)

    if Path(image_path).exists():
        Path(image_path).unlink()

    return pdf.output(dest="S").encode("latin-1")

report_bytes = None
report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
probability_data = {x["class"]: x["probability"] for x in ordered_probs}

if _FPDF_AVAILABLE:
    try:
        report_bytes = create_pdf_report(
            patient_name,
            patient_id,
            img_rgb,
            label,
            confidence,
            probability_data,
            report_timestamp,
        )
        st.success("📄 PDF report generated successfully.")
    except Exception as e:
        st.error(f"Could not generate PDF report: {e}")

if report_bytes:
    st.download_button(
        label="📥 Download Report",
        data=report_bytes,
        file_name=f"NeuroVision_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        key="download_report",
    )
else:
    st.info("Install `fpdf` (`pip install fpdf`) to enable report download.")