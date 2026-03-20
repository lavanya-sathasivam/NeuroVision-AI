import streamlit as st
import numpy as np
import cv2
import pandas as pd
import uuid
from pathlib import Path
from datetime import datetime
from PIL import Image
from matplotlib import pyplot as plt

from utils import load_model_and_labels, preprocess_image, predict_image
import patient_manager as pm
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

st.markdown("""
<style>
body { background: #f6f9fc; }
.app-card { border-radius: 14px; background: #ffffff; box-shadow: 0 6px 20px rgba(0,0,0,0.08); padding: 1rem; margin-bottom: 1rem; }
.title { font-size: 2.2rem; font-weight: 800; color: #023e8a; margin-bottom: 0.2rem; }
.subtitle { color: #334e68; margin-bottom: 1rem; }
.status-chip { color: white; padding: 0.35rem 0.8rem; border-radius: 10px; font-weight: 700; }
.small-muted { color: #647d98; font-size: 0.9rem; }
.highlight { font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# caching model
@st.cache_resource
def get_model_resources():
    return load_model_and_labels()

model, class_labels = get_model_resources()
CLASS_ORDER = ["glioma", "meningioma", "pituitary", "no tumor"]

# SESSION state defaults
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = ""
if "show_add_patient" not in st.session_state:
    st.session_state.show_add_patient = False
if "analysis_record" not in st.session_state:
    st.session_state.analysis_record = None

# --- Sidebar (minimal) ---
with st.sidebar:
    st.header("Patient Registry")
    search_input = st.text_input("Search patient", value="", key="sidebar_search")

    patients = pm.search_patients(search_input)
    options = [f"{p['id']} | {p.get('name', '(no name)')}" for p in patients]
    selected = st.selectbox("Select patient", options=["-- select patient --"] + options, index=0, key="sidebar_select")

    if selected != "-- select patient --":
        st.session_state.selected_patient_id = selected.split("|")[0].strip()
        st.session_state.show_add_patient = False

    if st.button("Add new patient"):
        st.session_state.show_add_patient = True
        st.session_state.selected_patient_id = ""

# --- Main dashboard ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("<div class='title'>🧠 NeuroVision AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Clinical MRI tumor analysis and decision support</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown("### Doctor Info")
    doctor_name = st.text_input("Doctor Name", value="Dr. Maya Rao", key="doc_name")
    doctor_dept = st.text_input("Department", value="Radiology", key="doc_dept")
    clinic_name = st.text_input("Clinic / Hospital", value="NeuroHealth Center", key="doc_clinic")

st.markdown("---")

# Patient panel
if st.session_state.show_add_patient:
    with st.container():
        st.markdown("<div class='app-card'><h3>Add Patient</h3></div>", unsafe_allow_html=True)
        with st.form(key="add_patient_form"):
            new_id = st.text_input("Patient ID")
            new_name = st.text_input("Name")
            new_age = st.number_input("Age", min_value=0, max_value=130, value=0)
            new_gender = st.selectbox("Gender", ["", "Female", "Male", "Other"])
            submitted = st.form_submit_button("Save Patient")
            if submitted:
                try:
                    if not new_id or not new_name:
                        st.warning("Patient ID and name are required")
                    else:
                        existing = pm.get_patient(new_id)
                        if existing:
                            pm.update_patient(new_id, name=new_name, age=new_age, gender=new_gender)
                            st.success("Patient updated")
                        else:
                            pm.add_patient(new_id, new_name, age=new_age, gender=new_gender)
                            st.success("Patient added")
                        st.session_state.selected_patient_id = new_id
                        st.session_state.show_add_patient = False
                except Exception as ex:
                    st.error(f"Could not add patient: {ex}")

if not st.session_state.selected_patient_id and not st.session_state.show_add_patient:
    st.markdown("<div class='app-card'><h2>Select or add a patient to begin</h2></div>", unsafe_allow_html=True)
    st.stop()

patient_id = st.session_state.selected_patient_id
patient = pm.get_patient(patient_id) if patient_id else None

if patient:
    with st.container():
        st.markdown("<div class='app-card'><h3>Patient Details</h3></div>", unsafe_allow_html=True)
        st.write({
            "ID": patient_id,
            "Name": patient.get("name"),
            "Age": patient.get("age", "N/A"),
            "Gender": patient.get("gender", "N/A"),
        })
else:
    st.warning("No patient selected. Use the sidebar to select or add one.")
    st.stop()

st.markdown("---")

# Scan workflow
workflow = st.container()
with workflow:
    st.markdown("<div class='app-card'><h3>1. Upload Scan</h3></div>", unsafe_allow_html=True)
    upload_file = st.file_uploader("Drag & drop MRI scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
    run_analysis = st.button("2. Run Analysis")

    if not upload_file:
        st.info("Upload a scan to enable analysis")

record = st.session_state.analysis_record
history = pm.get_patient_history(patient_id)

# load history selection
if history:
    history_options = ["Current session"] + [f"{r['timestamp']} | {r['prediction']} ({r['confidence']*100:.1f}%)" for r in history]
    chosen = st.selectbox("Select scan history to load", history_options)
    if chosen != "Current session":
        idx = history_options.index(chosen) - 1
        record = history[idx]
        st.session_state.analysis_record = record

# analyze
if run_analysis and upload_file:
    try:
        bytes_data = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Invalid image. Please upload a valid MRI scan.")
            st.stop()

        with st.spinner("Analyzing... please wait"):
            label, confidence, preds = predict_image(img_bgr, model, class_labels)
            prob_map = {class_labels[i]: float(preds[i]) for i in range(len(preds))}
            status_class = "no tumor" if label == "no tumor" else "tumor"
            status_color = "#2d6a4f" if status_class == "no tumor" else "#d00000"
            interpretation = "No abnormality detected. Continue routine monitoring." if status_class == "no tumor" else "Further radiological evaluation recommended."

            record = {
                "record_id": str(uuid.uuid4()),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_id": patient_id,
                "patient_name": patient.get("name"),
                "prediction": label,
                "confidence": confidence,
                "status": "No Tumor" if status_class == "no tumor" else "Tumor Detected",
                "status_class": status_class,
                "status_color": status_color,
                "interpretation": interpretation,
                "probabilities": prob_map,
                "image_path": pm.save_scan_image(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"),
            }
            pm.add_scan_record(patient_id, record)
            st.success("Analysis complete")
            st.session_state.analysis_record = record
    except Exception as err:
        st.error(f"Analysis error: {err}")

if not record:
    st.info("No analysis record available yet")
    st.stop()

# Results section
st.markdown("---")
st.markdown("<div class='app-card'><h3>3. Diagnostic Results</h3></div>", unsafe_allow_html=True)

status_color = record.get("status_color", "#2d6a4f")
st.markdown(f"<div class='status-chip' style='background:{status_color};'>{record.get('status')}</div>", unsafe_allow_html=True)
st.markdown(f"**Prediction:** {record.get('prediction', '').title()}")
st.markdown(f"**Confidence:** {record.get('confidence',0)*100:.1f}%")
st.progress(int(record.get('confidence',0)*100))
st.markdown(f"**Clinical Interpretation:** {record.get('interpretation')}")

# Probability chart
probs = sorted(record.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)
df_prob = pd.DataFrame(probs, columns=["Class", "Probability"])
fig, ax = plt.subplots(figsize=(6,3))
colors = ["#2d6a4f" if c=="no tumor" else "#d00000" for c in df_prob["Class"]]
ax.barh(df_prob["Class"], df_prob["Probability"], color=colors)
ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("Probability")
for i,v in enumerate(df_prob["Probability"]): ax.text(v+0.01, i, f"{v*100:.1f}%", va='center')
st.pyplot(fig)

# Grad-CAM
st.markdown("---")
st.markdown("<div class='app-card'><h3>4. Grad-CAM Explainability</h3></div>", unsafe_allow_html=True)
img_path = record.get("image_path")
img_arr = np.array(Image.open(img_path).convert("RGB")) if img_path and Path(img_path).exists() else np.zeros((224,224,3), dtype=np.uint8)
heatmap = generate_gradcam(model, preprocess_image(cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)))
overlay = overlay_heatmap(cv2.resize(cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR), (224,224)), heatmap)
cols = st.columns(3)
with cols[0]:
    st.image(img_arr, caption="Original", use_column_width=True)
with cols[1]:
    heatmap_vis = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
    st.image(heatmap_vis, caption="Heatmap", use_column_width=True)
with cols[2]:
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay", use_column_width=True)
st.markdown("*Highlighted regions indicate areas influencing the model decision.*")

# History
st.markdown("---")
st.markdown("<div class='app-card'><h3>5. Scan History</h3></div>", unsafe_allow_html=True)
if history:
    history_df = pd.DataFrame(history)
    if "confidence" in history_df.columns:
        history_df["confidence"] = (history_df["confidence"]*100).round(1)
    st.dataframe(history_df[["timestamp","prediction","confidence","status"]].sort_values("timestamp", ascending=False))
else:
    st.write("No scan history for this patient yet.")

# PDF report
st.markdown("---")
if _FPDF_AVAILABLE:
    def create_pdf(record):
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0,10,'NeuroVision AI Clinical Report', ln=True, align='C')
        pdf.ln(4)
        pdf.set_font('Arial','',11)
        pdf.cell(0,6,f"Doctor: {doctor_name} ({doctor_dept})", ln=True)
        pdf.cell(0,6,f"Clinic: {clinic_name}", ln=True)
        pdf.cell(0,6,f"Patient: {record.get('patient_name')} ({record.get('patient_id')})", ln=True)
        pdf.cell(0,6,f"Timestamp: {record.get('timestamp')}", ln=True)
        pdf.ln(4)
        pdf.cell(0,6,f"Prediction: {record.get('prediction')}", ln=True)
        pdf.cell(0,6,f"Confidence: {record.get('confidence',0)*100:.2f}%", ln=True)
        pdf.cell(0,6,f"Interpretation: {record.get('interpretation')}", ln=True)
        pdf.ln(4)
        for k,v in record.get('probabilities', {}).items():
            pdf.cell(0,6,f"{k.title()}: {v*100:.1f}%", ln=True)
        pdf.ln(4)
        if record.get('image_path') and Path(record.get('image_path')).exists():
            pdf.image(record.get('image_path'), x=25, w=160)
        return pdf.output(dest='S').encode('latin-1')

    if st.button('Generate PDF Report'):
        try:
            pdf_bytes = create_pdf(record)
            st.download_button('Download PDF Report', pdf_bytes, file_name=f"NeuroVision_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime='application/pdf')
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
else:
    st.warning('Install fpdf to enable PDF report generation (pip install fpdf)')
