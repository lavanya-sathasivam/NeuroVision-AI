import streamlit as st
import numpy as np
import cv2
import pandas as pd
import uuid
from datetime import datetime
from PIL import Image
import altair as alt

from utils import load_model_and_labels, preprocess_image, safe_predict, classify_confidence, clinical_decision
import patient_manager as pm
from gradcam import generate_gradcam, overlay_heatmap

# ================= CONFIG =================
st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def get_model():
    return load_model_and_labels()

model, class_labels = get_model()

# ================= SESSION =================
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None

if "record" not in st.session_state:
    st.session_state.record = None

if "img" not in st.session_state:
    st.session_state.img = None

if "show_add" not in st.session_state:
    st.session_state.show_add = False

# ================= SIDEBAR =================
with st.sidebar:
    st.title("👤 Patients")

    search = st.text_input("Search")
    patients = pm.search_patients(search)

    options = [""] + [f"{p['id']} | {p['name']}" for p in patients]
    selected = st.selectbox("Select", options)

    if selected:
        st.session_state.patient_id = selected.split("|")[0].strip()
        st.session_state.record = None

    if st.button("➕ Add Patient"):
        st.session_state.show_add = True

# ================= HEADER =================
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🧠 NeuroVision Clinical AI")
    st.caption("MRI Tumor Detection & Decision Support")

with col2:
    st.markdown("**Doctor**")
    doctor_name = st.text_input("Name", "Dr. Maya Rao")
    doctor_dept = st.text_input("Dept", "Radiology")

st.divider()

# ================= ADD PATIENT =================
if st.session_state.show_add:
    with st.form("add_patient"):
        st.subheader("Add Patient")

        pid = st.text_input("Patient ID")
        name = st.text_input("Name")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        if st.form_submit_button("Save"):
            if not pid or not name:
                st.error("Required fields missing")
            else:
                pm.add_patient(pid, name, age=age, gender=gender)
                st.success("Patient added")
                st.session_state.patient_id = pid
                st.session_state.show_add = False

# ================= VALIDATION =================
if not st.session_state.patient_id:
    st.warning("Select or add a patient to begin")
    st.stop()

patient = pm.get_patient(st.session_state.patient_id)

# ================= PATIENT INFO =================
st.subheader("Patient Info")

c1, c2, c3 = st.columns(3)
c1.metric("Name", patient.get("name"))
c2.metric("Age", patient.get("age"))
c3.metric("Gender", patient.get("gender"))

st.divider()

# ================= UPLOAD =================
st.subheader("Upload MRI")

uploaded = st.file_uploader("Upload scan", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        st.error("Invalid image")
        st.stop()

    if img.shape[0] < 100 or img.shape[1] < 100:
        st.error("Low resolution image")
        st.stop()

    st.image(img, caption="MRI Scan")

    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):

            success, result = safe_predict(img, model, class_labels)

            if not success:
                st.error(result)
                st.stop()

            label = result["label"]
            confidence = result["confidence"]
            probs = result["probs"]
            level = result["confidence_level"]

            decision, message = clinical_decision(label, level)

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": str(datetime.now()),
            "prediction": label,
            "confidence": confidence,
            "confidence_level": level,
            "decision": decision,
            "probabilities": {class_labels[i]: float(probs[i]) for i in range(len(probs))}
        }

        pm.add_scan_record(st.session_state.patient_id, record)

        st.session_state.record = record
        st.session_state.img = img

# ================= RESULTS =================
record = st.session_state.record

if record:
    st.divider()

    st.info("⚠️ AI-assisted only. Final diagnosis must be made by a radiologist.")

    st.subheader("Results")

    # STATUS
    if record["decision"] == "SAFE":
        st.success(record["decision"])

    elif record["decision"] == "ALERT":
        st.error(record["decision"])

    elif record["decision"] == "CAUTION":
        st.warning(record["decision"])

    else:
        st.warning("UNCERTAIN")

    st.write(record["decision"])
    st.write(record["confidence_level"])

    st.metric("Confidence", f"{record['confidence']*100:.2f}%")
    st.progress(int(record["confidence"] * 100))

    # ================= CHART =================
    df = pd.DataFrame(record["probabilities"].items(), columns=["Class", "Prob"])

    chart = alt.Chart(df).mark_bar().encode(
        x="Class",
        y="Prob",
        color=alt.condition(
            alt.datum.Class == record["prediction"],
            alt.value("#d00000"),
            alt.value("#4e79a7"),
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # ================= GRADCAM =================
    st.subheader("Grad-CAM")

    try:
        processed = preprocess_image(st.session_state.img)

        heatmap = generate_gradcam(model, processed)

        if heatmap is None:
            raise ValueError("Grad-CAM failed internally")

        base = cv2.resize(st.session_state.img, (224, 224))

        overlay = overlay_heatmap(base, heatmap)

        col1, col2 = st.columns(2)
        col1.image(st.session_state.img, caption="Original")
        col2.image(overlay, caption="AI Focus")

    except Exception as e:
        st.warning(f"Grad-CAM unavailable: {e}")
# ================= HISTORY =================
st.divider()
st.subheader("History")

history = pm.get_patient_history(st.session_state.patient_id)

if history:
    df = pd.DataFrame(history)
    df["confidence"] = df["confidence"] * 100
    st.dataframe(df[["timestamp", "prediction", "confidence"]])
else:
    st.info("No history yet")