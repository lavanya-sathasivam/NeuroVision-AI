import streamlit as st
import numpy as np
import cv2
import pandas as pd
import uuid
from datetime import datetime
from PIL import Image
import altair as alt

from utils import load_model_and_labels, preprocess_image, predict_image, classify_confidence, clinical_decision, validate_image
import patient_manager as pm
from gradcam import generate_gradcam, overlay_heatmap

# ================= CONFIG =================
st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def get_model():
    return load_model_and_labels()

model, class_labels = get_model()

# ================= SESSION STATE =================
if "current_patient" not in st.session_state:
    st.session_state.current_patient = None

if "current_scan" not in st.session_state:
    st.session_state.current_scan = None

if "show_add_patient" not in st.session_state:
    st.session_state.show_add_patient = False

if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None

# ================= SIDEBAR =================
with st.sidebar:
    st.title("👨‍⚕️ Doctor Information")

    # Doctor Info
    doctor_name = st.text_input("Name", "Dr. Maya Rao", key="doctor_name")
    doctor_dept = st.text_input("Department", "Radiology", key="doctor_dept")
    doctor_hospital = st.text_input("Hospital", "City General Hospital", key="doctor_hospital")

    st.divider()
    st.caption("NeuroVision AI v2.0 - Clinical Decision Support")

# ================= MAIN PAGE =================

# Add Patient Form
if st.session_state.show_add_patient:
    st.header("Add New Patient")
    with st.form("add_patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", key="new_patient_id")
            patient_name = st.text_input("Full Name", key="new_patient_name")
        with col2:
            patient_age = st.number_input("Age", min_value=0, max_value=120, key="new_patient_age")
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="new_patient_gender")

        submitted = st.form_submit_button("Save Patient")
        if submitted:
            if not patient_id or not patient_name:
                st.error("Patient ID and Name are required.")
            else:
                try:
                    pm.add_patient(patient_id, patient_name, age=patient_age, gender=patient_gender)
                    st.success(f"Patient {patient_name} added successfully!")
                    st.session_state.current_patient = patient_id
                    st.session_state.show_add_patient = False
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

# Scan Details View
elif st.session_state.current_scan and st.session_state.current_patient:
    scan = pm.get_scan_record(st.session_state.current_patient, st.session_state.current_scan)
    if scan:
        st.header("Scan Details")

        # Back button
        if st.button("⬅️ Back to Dashboard", key="back_to_dashboard"):
            st.session_state.current_scan = None
            st.rerun()

        st.info("⚠️ This is AI-assisted analysis. Final diagnosis must be made by a qualified radiologist.")

        col1, col2 = st.columns(2)
        with col1:
            prediction_display = scan['prediction'].capitalize()
            if scan.get('decision') == "UNCERTAIN":
                prediction_display = "UNCERTAIN"
            st.metric("Prediction", prediction_display)
            st.metric("Confidence", f"{scan['confidence']*100:.1f}%")
            st.metric("Confidence Level", scan.get('confidence_level', 'UNKNOWN'))
        with col2:
            decision = scan.get('decision', 'UNKNOWN')
            if decision == "SAFE":
                st.success(f"Decision: {decision}")
            elif decision == "ALERT":
                st.error(f"Decision: {decision}")
            elif decision == "UNCERTAIN":
                st.warning(f"Decision: {decision}")
            elif decision == "CAUTION":
                st.warning(f"Decision: {decision}")
            else:
                st.info(f"Decision: {decision}")
            st.write(scan.get('message', ''))

        # Probability Chart
        if 'probabilities' in scan:
            probs_df = pd.DataFrame(scan['probabilities'].items(), columns=["Class", "Probability"])
            chart = alt.Chart(probs_df).mark_bar().encode(
                x="Class",
                y="Probability",
                color=alt.condition(
                    alt.datum.Class == scan['prediction'],
                    alt.value("#d00000"),
                    alt.value("#4e79a7"),
                )
            )
            st.altair_chart(chart, use_container_width=True)

        # Grad-CAM
        if st.session_state.preprocessed_img is not None and st.session_state.uploaded_img is not None:
            st.subheader("AI Focus Areas (Grad-CAM)")
            try:
                processed = st.session_state.preprocessed_img
                if processed.shape != (1, 224, 224, 3):
                    raise ValueError(f"Unexpected preprocessed image shape: {processed.shape}")

                heatmap = generate_gradcam(model, processed)

                if heatmap is not None:
                    overlay = overlay_heatmap(st.session_state.uploaded_img.copy(), heatmap)
                    if overlay is not None:
                        col1, col2 = st.columns(2)
                        col1.image(st.session_state.uploaded_img, caption="Original Scan", width=300)
                        col2.image(overlay, caption="AI Focus Areas", width=300)
                    else:
                        st.warning("Grad-CAM visualization failed to generate overlay.")
                else:
                    st.warning("Grad-CAM visualization unavailable for this scan.")
            except Exception as e:
                st.warning(f"Grad-CAM visualization failed: {e}")
    else:
        st.error("Scan not found.")
        st.session_state.current_scan = None

# Patient Selection Page
elif not st.session_state.current_patient:
    st.header("🧠 NeuroVision AI - Patient Selection")
    st.markdown("Search and select a patient to begin analysis, or add a new patient.")

    # Search Input
    search_term = st.text_input("Search by Patient ID or Name", key="search_term", placeholder="Enter patient ID or name...")

    # Get filtered patients
    patients = pm.search_patients(search_term)

    # Patient Selection Dropdown
    if patients:
        patient_options = [""] + [f"{p['id']} - {p['name']}" for p in patients]
        selected_patient = st.selectbox("Select Patient", patient_options, key="selected_patient")

        if selected_patient:
            patient_id = selected_patient.split(" - ")[0]
            st.session_state.current_patient = patient_id
            st.session_state.current_scan = None  # Reset scan when switching patient
            st.rerun()
    else:
        st.info("No patients found matching your search.")

    # Add Patient Button
    st.divider()
    if st.button("➕ Add New Patient", key="add_patient_btn"):
        st.session_state.show_add_patient = True

# Patient Dashboard
else:
    patient = pm.get_patient(st.session_state.current_patient)
    if not patient:
        st.error("Patient not found.")
        st.session_state.current_patient = None
        st.rerun()

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"Patient Dashboard - {patient['name']}")
    with col2:
        if st.button("🔄 Switch Patient", key="switch_patient"):
            st.session_state.current_patient = None
            st.session_state.current_scan = None
            st.rerun()

    # Patient Info
    st.subheader("Patient Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ID", st.session_state.current_patient)
    col2.metric("Name", patient['name'])
    col3.metric("Age", patient.get('age', 'N/A'))
    col4.metric("Gender", patient.get('gender', 'N/A'))

    st.divider()

    # New MRI Consultation
    st.subheader("🚀 New MRI Consultation")
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"], key="upload_scan")

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Failed to decode uploaded image. Please try a different file.")
        else:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Validate image
            valid, error_msg = validate_image(img)
            if not valid:
                st.error(f"Image validation failed: {error_msg}")
            else:
                st.image(img, caption="Uploaded MRI Scan", width=300)
                st.session_state.uploaded_img = img

                if st.button("🔍 Run Analysis", key="run_analysis"):
                    with st.spinner("Analyzing MRI scan..."):
                        # Preprocess once and reuse for predict and Grad-CAM
                        preprocessed_img = preprocess_image(img)
                        st.write(f"[DEBUG] preprocessed image shape: {preprocessed_img.shape}")

                        try:
                            label, confidence, probs = predict_image(preprocessed_img, model, class_labels, preprocessed=True)
                            st.write(f"[DEBUG] prediction probabilities: {probs}")
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")
                            st.stop()

                        confidence_level = classify_confidence(confidence)
                        decision, message = clinical_decision(label, confidence_level)

                        scan_record = {
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.now().isoformat(),
                            "prediction": label,
                            "confidence": confidence,
                            "confidence_level": confidence_level,
                            "decision": decision,
                            "message": message,
                            "probabilities": {class_labels[i]: float(probs[i]) for i in range(len(probs))}
                        }

                        # Save to patient
                        pm.add_scan_record(st.session_state.current_patient, scan_record)

                        # Set as current scan
                        st.session_state.current_scan = scan_record["id"]
                        st.success("Analysis complete!")

                        # Keep processed batch for Grad-CAM in state
                        st.session_state.preprocessed_img = preprocessed_img
                        st.rerun()

                        decision, message = clinical_decision(label, level)

                        # Create scan record
                        scan_record = {
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.now().isoformat(),
                            "prediction": label,
                            "confidence": confidence,
                            "confidence_level": level,
                            "decision": decision,
                            "message": message,
                            "probabilities": {class_labels[i]: float(probs[i]) for i in range(len(probs))}
                        }

                        # Save to patient
                        pm.add_scan_record(st.session_state.current_patient, scan_record)

                        # Set as current scan
                        st.session_state.current_scan = scan_record["id"]
                        st.success("Analysis complete!")
                        st.rerun()

    st.divider()

    # Scan History (Collapsible)
    with st.expander("📂 Scan History", expanded=False):
        history = pm.get_patient_history(st.session_state.current_patient)

        if history:
            # Create dataframe
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = df.sort_values('timestamp', ascending=False)

            # Clean data
            df['confidence'] = (df['confidence'] * 100).round(1).astype(str) + '%'
            df['confidence_level'] = df.get('confidence_level', 'UNKNOWN')
            df['decision'] = df.get('decision', 'UNKNOWN')

            # Display table with buttons
            for idx, row in df.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 1, 1])
                timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['timestamp']) else 'UNKNOWN'
                col1.write(timestamp_str)
                col2.write(row['prediction'].capitalize())
                col3.write(f"{row['confidence']} ({row['confidence_level']})")
                col4.write(row['decision'])
                if col5.button("👁️ View", key=f"view_{row['id']}"):
                    st.session_state.current_scan = row['id']
                    st.rerun()
                if col6.button("🗑️ Delete", key=f"delete_{row['id']}"):
                    try:
                        pm.delete_scan_record(st.session_state.current_patient, row['id'])
                        st.success("Scan deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete scan: {e}")
        else:
            st.info("No scans available for this patient.")

