import streamlit as st
import numpy as np
import cv2
import pandas as pd
import json
import uuid
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
    .css-18e3th9 { padding-top: 1rem; }
    .main > div { padding: 1rem 1.5rem; }

    .app-card {
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        padding: 1rem;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    .app-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #023e8a;
    }

    .app-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #334e68;
        margin-bottom: 0.8rem;
    }

    .status-chip {
        border-radius: 8px;
        color: white;
        padding: 0.3rem 0.65rem;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        margin-top: 0.2rem;
    }

    .label-bold {
        font-weight: 700;
        font-size: 1.1rem;
    }

    .small-muted {
        color: #627d98;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

HISTORY_FILE = Path("patient_history.json")
IMG_FOLDER = Path("history_images")
IMG_FOLDER.mkdir(exist_ok=True)

def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

def get_patient_history(patient_id):
    history = load_history()
    return history.get(patient_id, [])

def append_scan_record(patient_id, record):
    history = load_history()
    existing = history.get(patient_id, [])
    existing.append(record)
    history[patient_id] = existing
    save_history(history)

@st.cache_resource
def load_resources():
    return load_model_and_labels()

model, class_labels = load_resources()
CLASS_ORDER = ["glioma", "meningioma", "pituitary", "no tumor"]

# Sidebar: Clinical intake
st.sidebar.markdown("## 🧾 Clinical Intake Panel")
doctor_name = st.sidebar.text_input("Doctor Name", value="Dr. Maya Rao")
doctor_dept = st.sidebar.text_input("Department", value="Radiology")
clinic_name = st.sidebar.text_input("Clinic/Hospital", value="NeuroHealth Center")

st.sidebar.markdown("---")
patient_name = st.sidebar.text_input("Patient Name", value="John Doe")
patient_id = st.sidebar.text_input("Patient ID", value="NV-2026-001")

uploaded_file = st.sidebar.file_uploader("Upload MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")

selected_history = None
if patient_id:
    patient_history = get_patient_history(patient_id)
    if patient_history:
        history_options = ["New scan"] + [f"{item['timestamp']} | {item['prediction']}" for item in patient_history]
        selected = st.sidebar.selectbox("Select previous scan", history_options)
        if selected != "New scan":
            idx = history_options.index(selected) - 1
            selected_history = patient_history[idx]
    else:
        st.sidebar.info("No history yet for this patient.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "#### ⚠️ Disclaimer\n"
    "This tool is for research purposes only and not a substitute for professional diagnosis.",
    unsafe_allow_html=True,
)

st.markdown("# 🧠 NeuroVision AI", unsafe_allow_html=True)
st.markdown("### Diagnostic Output · Scan Analysis · Patient Record")
st.markdown(f"<div class='app-subtitle'>Doctor: {doctor_name} | {doctor_dept} | {clinic_name}</div>", unsafe_allow_html=True)

if not patient_id:
    st.warning("Enter Patient ID to load/save scan history.")

if selected_history and not uploaded_file:
    record = selected_history
    is_historical = True
else:
    record = None
    is_historical = False

if uploaded_file and (st.sidebar.button("Run Analysis") or not selected_history):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Unable to decode image. Please upload a valid MRI scan.")
            st.stop()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("🔍 Analyzing MRI scan with EfficientNet... Please wait"):
            label, confidence, preds = predict_image(img_bgr, model, class_labels)
            idx_proba = {class_labels[i]: float(preds[i]) for i in range(len(preds))}
            pred_status = "no tumor" if label == "no tumor" else "tumor"
            status_color = "#2d6a4f" if pred_status == "no tumor" else "#d00000"
            ordered_probs = [
                {"class": c, "probability": idx_proba.get(c, 0.0)} for c in CLASS_ORDER
            ]
            sorted_probs = sorted(ordered_probs, key=lambda x: x["probability"], reverse=True)

            processed_img = preprocess_image(img_bgr)
            heatmap = generate_gradcam(model, processed_img)
            overlay = overlay_heatmap(cv2.resize(img_bgr, (224, 224)), heatmap)

        st.success("✅ Scan analysis completed successfully.")

        image_filename = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        image_path = IMG_FOLDER / image_filename
        Image.fromarray(img_rgb).save(image_path)

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_name": patient_name,
            "patient_id": patient_id,
            "prediction": label,
            "confidence": float(confidence),
            "status": "No Tumor" if label == "no tumor" else "Tumor Detected",
            "status_color": status_color,
            "probabilities": {item["class"]: item["probability"] for item in ordered_probs},
            "image_path": str(image_path),
            "heatmap": True,
        }

        if patient_id:
            append_scan_record(patient_id, record)

        is_historical = False

    except Exception as e:
        st.error(f"Analysis error: {e}")
        record = None

if record is None:
    st.info("Upload a scan and click 'Run Analysis' to see results.")
    st.stop()

status_class = "No Tumor" if record.get("status") == "No Tumor" else "Tumor Detected"
status_marker = "#2d6a4f" if status_class == "No Tumor" else "#d00000"

# Main dashboard display
main_cols = st.columns([1.2, 1])
with main_cols[0]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("## 🔎 Scan Diagnostic Output")
    st.markdown(f"<div class='status-chip' style='background-color: {status_marker};'>{status_class}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='label-bold'>Patient:</p> {record.get('patient_name', '')} ({record.get('patient_id', '')})")
    st.markdown(f"<p class='label-bold'>Analysis time:</p> {record.get('timestamp')}")
    st.markdown("---")
    st.markdown(f"<p class='label-bold'>Predicted class:</p> <strong>{record.get('prediction', '').title()}</strong>")
    st.markdown(f"<p class='label-bold'>Confidence:</p> {record.get('confidence', 0) * 100:.2f}%")
    st.progress(int(record.get('confidence', 0) * 100))
    st.markdown("<p class='small-muted'>Confidence represents model certainty based on output probabilities.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Patient Scan History"):
        patient_history = get_patient_history(patient_id) if patient_id else []
        if patient_history:
            history_df = pd.DataFrame([{
                "Timestamp": h["timestamp"],
                "Prediction": h["prediction"],
                "Confidence": f"{h['confidence'] * 100:.1f}%",
                "Status": h["status"],
            } for h in reversed(patient_history)])
            st.table(history_df)
        else:
            st.write("No history records available.")

with main_cols[1]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("## 📊 Probability Distribution")

    probs = [(k, v) for k, v in record.get("probabilities", {}).items()]
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    df_probs = pd.DataFrame(probs, columns=["class", "probability"])

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(df_probs["class"], df_probs["probability"], color=["#2d6a4f" if c == 'no tumor' else '#d00000' for c in df_probs['class']])
    for i, (cls, prob) in enumerate(probs):
        if cls == record.get('prediction'):
            bars[i].set_color('#023e8a')
        ax.text(prob + 0.015, i, f"{prob*100:.1f}%", va="center", fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔍 Grad-CAM Explainability")

grad_cols = st.columns(3)
if record.get('image_path') and Path(record.get('image_path')).exists():
    origin = np.array(Image.open(record['image_path']))
else:
    origin = np.zeros((224, 224, 3), dtype=np.uint8)

heatmap = generate_gradcam(model, preprocess_image(cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)))
overlay = overlay_heatmap(cv2.resize(cv2.cvtColor(origin, cv2.COLOR_RGB2BGR), (224, 224)), heatmap)

with grad_cols[0]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### Original MRI")
    st.image(origin, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with grad_cols[1]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### Heatmap")
    heatmap_vis = np.uint8(255 * heatmap)
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
    st.image(heatmap_vis, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with grad_cols[2]:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### Overlay")
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    st.image(overlay_rgb, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 📄 Downloadable Diagnostic Report")

if not _FPDF_AVAILABLE:
    st.warning("Install fpdf (pip install fpdf) to enable downloadable reports.")

if _FPDF_AVAILABLE:
    def create_pdf_report(patient_name, patient_id, doctor_name, doctor_dept, clinic_name, record):
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, "NeuroVision AI - MRI Diagnostic Report", ln=True, align="C")
        pdf.ln(4)

        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6, f"Date/Time: {record.get('timestamp')}", ln=True)
        pdf.cell(0, 6, f"Doctor: {doctor_name} ({doctor_dept})", ln=True)
        pdf.cell(0, 6, f"Clinic: {clinic_name}", ln=True)
        pdf.cell(0, 6, f"Patient: {patient_name} ({patient_id})", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Diagnostic Output", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 6, f"Prediction: {record.get('prediction', '').title()}", ln=True)
        pdf.cell(0, 6, f"Confidence: {record.get('confidence', 0)*100:.2f}%", ln=True)
        pdf.cell(0, 6, f"Status: {record.get('status', '')}", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 6, "Class Probabilities", ln=True)
        pdf.set_font("Arial", "", 11)
        for cls, prob in record.get("probabilities", {}).items():
            pdf.cell(0, 6, f"{cls.title()}: {prob*100:.2f}%", ln=True)
        pdf.ln(4)

        if record.get('image_path') and Path(record.get('image_path')).exists():
            pdf.image(record['image_path'], x=25, w=160)
        return pdf.output(dest="S").encode("latin-1")

    if st.button("Generate PDF Report"):
        try:
            pdf_bytes = create_pdf_report(patient_name, patient_id, doctor_name, doctor_dept, clinic_name, record)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"NeuroVision_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Could not create PDF report: {e}")
