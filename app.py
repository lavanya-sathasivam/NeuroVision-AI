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
import user_manager as um
import email_notifications as email_notif
import report_scheduler as rs

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")

@st.cache_resource
def get_model():
    return load_model_and_labels()

model, class_labels = get_model()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

if "current_patient" not in st.session_state:
    st.session_state.current_patient = None

if "current_scan" not in st.session_state:
    st.session_state.current_scan = None

if "show_add_patient" not in st.session_state:
    st.session_state.show_add_patient = False

if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None

if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.7

if "notifications_enabled" not in st.session_state:
    st.session_state.notifications_enabled = True

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "auto_save_enabled" not in st.session_state:
    st.session_state.auto_save_enabled = True

if "gradcam_cache" not in st.session_state:
    st.session_state.gradcam_cache = {}

if "preprocessed_img" not in st.session_state:
    st.session_state.preprocessed_img = None

# ================= SIDEBAR =================
with st.sidebar:
    if not st.session_state.logged_in:
        st.title("Login")
    else:
        st.title("User Profile")
        user = um.get_user(st.session_state.current_user)
        if user:
            st.write(f"**Name:** {user['full_name']}")
            st.write(f"**Email:** {user['email']}")
            st.write(f"**Department:** {user['department']}")
        st.divider()
        
        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
                st.session_state.show_settings = False
                st.session_state.show_signup = False
        with col2:
            if st.button("Settings", key="nav_settings", use_container_width=True):
                st.session_state.show_settings = True
                st.session_state.current_patient = None
                st.session_state.show_add_patient = False
        st.divider()
        
        if st.button("Logout", key="logout_btn", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.show_settings = False
            st.session_state.current_patient = None
            st.rerun()
        st.divider()
        st.caption("NeuroVision AI v1.0 - Clinical Decision Support")

# ================= MAIN PAGE =================

# LOGIN/SIGNUP PAGE
if not st.session_state.logged_in:
    if st.session_state.show_signup:
        st.header("Create New Account")  
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            with col1:
                user_id = st.text_input("Username", key="signup_user_id", help="Username already exists! Choose a unique username.")
                full_name = st.text_input("Full Name", key="signup_full_name")
                email = st.text_input("Email Address", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
            
            with col2:
                confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
                department = st.text_input("Department", key="signup_department", value="Radiology")
                hospital = st.text_input("Hospital/Organization", key="signup_hospital", value="")
                phone = st.text_input("Phone Number (Optional)", key="signup_phone", value="")
            
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if not user_id or not full_name or not email or not password:
                    st.error("Please fill in all required fields!")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    try:
                        um.create_user(
                            user_id=user_id,
                            password=password,
                            full_name=full_name,
                            email=email,
                            department=department,
                            hospital=hospital,
                            phone=phone
                        )
                        st.success(f"Account created successfully for {full_name}!")
                        st.info("You can now login with your credentials")
                        st.session_state.show_signup = False
                        st.rerun()
                    except ValueError as e:
                        st.error(f"{str(e)}")
        st.divider()
        if st.button("Back to Login", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()
    else:
        # LOGIN PAGE
        st.header("NeuroVision AI - Doctor Portal")
        st.subheader("Clinical Decision Support System")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### Welcome Back!
            
            Access your patient records and AI-powered diagnostic tools.
            
            **Features:**
            - MRI Analysis with AI
            - Grad-CAM Visualization
            - Patient Management
            - Clinical Insights
            """)
        with col2:
            st.markdown("### Login to Your Account")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if not username or not password:
                        st.error("Please enter username and password")
                    elif um.authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        um.update_last_login(username)
                        # Load user settings
                        user_settings = um.load_user_settings(username)
                        st.session_state.confidence_threshold = user_settings.get("model", {}).get("confidence_threshold", 0.7)
                        st.session_state.notifications_enabled = user_settings.get("notifications", {}).get("enabled", True)
                        st.session_state.dark_mode = user_settings.get("display", {}).get("dark_mode", False)
                        st.session_state.auto_save_enabled = user_settings.get("privacy", {}).get("auto_save", True)
                        
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
            st.divider()
            st.markdown("### New User?")
            if st.button("Create an Account", use_container_width=True):
                st.session_state.show_signup = True
                st.rerun()
            st.divider()
            st.info("For account recovery, contact your system administrator")
# SETTINGS PAGE
elif st.session_state.show_settings:
    st.header("Application Settings")
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Display", 
        "Model", 
        "Email Notifications", 
        "Report Scheduling",
        "In-App Alerts", 
        "Privacy & Data", 
        "About"
    ])
    # ===== DISPLAY SETTINGS =====
    with tab1:
        st.subheader("Display Preferences")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.dark_mode = st.checkbox(
                "Dark Mode", 
                value=st.session_state.dark_mode,
                help="Enable dark theme for the interface"
            )
        with col2:
            show_advanced_metrics = st.checkbox(
                "Show Advanced Metrics",
                value=False,
                help="Display detailed statistical information on dashboards"
            )
        st.divider()
        st.write("**Chart Settings**")
        col1, col2 = st.columns(2)
        with col1:
            chart_style = st.selectbox(
                "Chart Style",
                ["Default", "Minimal", "Dark Background"],
                help="Choose visualization style for prediction charts"
            )
        with col2:
            animation_enabled = st.checkbox(
                "Enable Animations",
                value=True,
                help="Animate chart transitions and updates"
            )
        st.divider()
        st.write("**Layout**")
        layout_cols = st.selectbox(
            "Default Sidebar Width",
            ["Narrow", "Medium", "Wide"],
            help="Set the preferred sidebar width"
        )
    # ===== MODEL SETTINGS =====
    with tab2:
        st.subheader("Model & Prediction Settings")
        st.write("**Confidence Thresholds**")
        st.session_state.confidence_threshold = st.slider(
            "Minimum Confidence Threshold for Diagnosis",
            min_value=0.5,
            max_value=0.99,
            value=st.session_state.confidence_threshold,
            step=0.01,
            help="Predictions below this threshold will be marked as UNCERTAIN"
        )
        st.info(f"Current threshold: {st.session_state.confidence_threshold*100:.1f}%")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            alert_threshold = st.slider(
                "ALERT Decision Threshold",
                min_value=0.5,
                max_value=0.99,
                value=0.85,
                step=0.01,
                help="Confidence level above which to trigger ALERT status"
            )
        with col2:
            caution_threshold = st.slider(
                "CAUTION Decision Threshold",
                min_value=0.5,
                max_value=0.99,
                value=0.70,
                step=0.01,
                help="Confidence level above which to trigger CAUTION status"
            )
        st.divider()
        st.write("**Inference Settings**")
        col1, col2 = st.columns(2)
        with col1:
            batch_processing = st.checkbox(
                "Enable Batch Processing",
                value=False,
                help="Process multiple scans simultaneously"
            )
        with col2:
            gpu_acceleration = st.checkbox(
                "GPU Acceleration (if available)",
                value=True,
                help="Use GPU for faster inference"
            )
        
        if st.button("Reset Model to Defaults", key="reset_model_btn"):
            st.session_state.confidence_threshold = 0.7
            st.success("Model settings reset to defaults!")
            st.rerun()
    # ===== EMAIL NOTIFICATIONS SETTINGS =====
    with tab3:
        st.subheader("Email Notification Configuration")
        
        st.write("**SMTP Settings**")
        user_settings = um.load_user_settings(st.session_state.current_user)
        email_config = user_settings.get("email", {})
        
        col1, col2 = st.columns(2)
        with col1:
            smtp_server = st.text_input(
                "SMTP Server",
                value=email_config.get("smtp_server", "smtp.gmail.com"),
                help="Email SMTP server (e.g., smtp.gmail.com)"
            )
            smtp_port = st.number_input(
                "SMTP Port",
                value=email_config.get("smtp_port", 587),
                help="SMTP port (usually 587 for TLS)"
            )
        with col2:
            email_address = st.text_input(
                "Email Address",
                value=email_config.get("email_address", ""),
                help="Email account to send notifications from"
            )
            email_password = st.text_input(
                "Email Password",
                type="password",
                value="",
                help="Email password or app-specific password"
            )
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Test Email Configuration", use_container_width=True):
                if email_address and email_password:
                    success, message = email_notif.test_email_configuration(
                        email_address, email_password, smtp_server, int(smtp_port)
                    )
                    if success:
                        st.success(f"{message}")
                    else:
                        st.error(f"{message}")
                else:
                    st.warning("Please enter email and password to test")
        with col2:
            if st.button("Send Test Email", use_container_width=True):
                if email_address and email_password:
                    success, msg = email_notif.send_email(
                        sender_email=email_address,
                        sender_password=email_password,
                        recipient_email=email_address,
                        subject="NeuroVision AI - Test Email",
                        body="This is a test email from NeuroVision AI. If you received this, your email configuration is working correctly.",
                        smtp_server=smtp_server,
                        smtp_port=int(smtp_port)
                    )
                    if success:
                        st.success("Test email sent successfully!")
                    else:
                        st.error(f"Failed to send email: {msg}")
                else:
                    st.warning("Please configure email settings first")
        
        with col3:
            if st.button("Save Email Settings", use_container_width=True):
                um.update_user_settings(st.session_state.current_user, "email", "smtp_server", smtp_server)
                um.update_user_settings(st.session_state.current_user, "email", "smtp_port", int(smtp_port))
                um.update_user_settings(st.session_state.current_user, "email", "email_address", email_address)
                if email_password:
                    um.update_user_settings(st.session_state.current_user, "email", "email_password", email_password)
                st.success("Email settings saved!")
        st.divider()
        st.write("**Email Notification Preferences**")
        notify_via_email = st.checkbox(
            "Send Predictions via Email",
            value=email_config.get("notifications_via_email", False),
            help="Email scan reports to configured recipients"
        )
        if notify_via_email:
            st.write("**Email Recipients**")
            recipients_text = st.text_area(
                "Recipient Email Addresses (one per line)",
                value="\n".join(email_config.get("recipients", [email_address])),
                help="List of email addresses to receive reports"
            )
            recipients = [r.strip() for r in recipients_text.split("\n") if r.strip()]
    # ===== REPORT SCHEDULING SETTINGS =====
    with tab4:
        st.subheader("Automated Report Scheduling")
        # List existing schedules
        schedules = rs.list_user_schedules(st.session_state.current_user)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Active Schedules:** {len([s for s in schedules if s.get('is_active')])}")
        with col2:
            if st.button("Create New Schedule", use_container_width=True):
                st.session_state.create_schedule = True
        st.divider()
        if st.session_state.get("create_schedule"):
            st.markdown("### Create New Report Schedule")
            with st.form("create_schedule_form"):
                col1, col2 = st.columns(2)
                with col1:
                    schedule_name = st.text_input(
                        "Schedule Name",
                        value="Weekly Report",
                        help="Name for this schedule"
                    )
                    frequency = st.selectbox(
                        "Report Frequency",
                        ["Daily", "Weekly", "Monthly"],
                        help="How often to generate and send reports"
                    )
                with col2:
                    recipients_input = st.text_area(
                        "Email Recipients",
                        value=um.get_user(st.session_state.current_user).get("email", ""),
                        help="Email addresses to receive reports (one per line)"
                    )
                    report_type = st.multiselect(
                        "Report Contents",
                        ["Model Metrics", "Patient Summary", "Prediction Charts", "Confidence Distribution"],
                        default=["Model Metrics", "Prediction Charts"]
                    )
                submitted = st.form_submit_button("Create Schedule", use_container_width=True)
                if submitted and schedule_name:
                    try:
                        schedule_id = f"schedule_{uuid.uuid4().hex[:8]}"
                        recipients = [r.strip() for r in recipients_input.split("\n") if r.strip()]
                        
                        rs.create_report_schedule(
                            schedule_id=schedule_id,
                            user_id=st.session_state.current_user,
                            schedule_name=schedule_name,
                            frequency=frequency,
                            recipients=recipients,
                            include_metrics="Model Metrics" in report_type,
                            include_patient_summary="Patient Summary" in report_type
                        )
                        st.success(f"Schedule '{schedule_name}' created!")
                        st.session_state.create_schedule = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating schedule: {str(e)}")
        st.divider()
        if schedules:
            st.write("**Your Schedules:**")
            for schedule in schedules:
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
                col1.write(schedule['name'])
                col2.write(schedule['frequency'])
                col3.write(", ".join(schedule['recipients'][:2]))
                if col4.checkbox("Active", value=schedule.get('is_active'), key=f"active_{schedule['id']}"):
                    rs.update_schedule(schedule['id'], is_active=True)
                else:
                    rs.update_schedule(schedule['id'], is_active=False)
                if col5.button("Delete", key=f"del_schedule_{schedule['id']}"):
                    rs.delete_schedule(schedule['id'])
                    st.rerun()
        else:
            st.info("No schedules created yet. Create one to get started!")
        st.divider()
        st.write("**Schedule Statistics**")
        stats = rs.get_schedule_statistics()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Schedules", stats['total_schedules'])
        col2.metric("Active", stats['active_schedules'])
        col3.metric("Total Sent", stats['total_reports_sent'])
        col4.metric("Pending", len(rs.get_pending_schedules()))

    # ===== IN-APP NOTIFICATION SETTINGS =====
    with tab5:
        st.subheader("In-App Notification Preferences")        
        st.session_state.notifications_enabled = st.checkbox(
            "Enable All In-App Notifications",
            value=st.session_state.notifications_enabled,
            help="Master switch for all notifications"
        )
        if st.session_state.notifications_enabled:
            st.divider()
            st.write("**Notification Types**")   
            col1, col2 = st.columns(2)
            with col1:
                notify_high_confidence = st.checkbox(
                    "High Confidence Predictions",
                    value=True,
                    help="Alert when prediction confidence is very high"
                )
                notify_alerts = st.checkbox(
                    "ALERT Status Detected",
                    value=True,
                    help="Critical alert - always notify"
                )
            with col2:
                notify_uncertain = st.checkbox(
                    "Uncertain Predictions",
                    value=True,
                    help="Notify when predictions are uncertain"
                )
                notify_analysis_complete = st.checkbox(
                    "Analysis Complete",
                    value=True,
                    help="Notify when scan analysis finishes"
                )
            st.divider()
            st.write("**Notification Sound**")
            sound_enabled = st.checkbox(
                "Enable Notification Sounds",
                value=True,
                help="Play sound for critical alerts"
            )
        if st.button("Send Test Notification", key="test_notif_btn"):
            st.success("Test notification sent!")
    # ===== PRIVACY & DATA SETTINGS =====
    with tab6:
        st.subheader("Privacy & Data Management")
        st.write("**Data Storage**")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.auto_save_enabled = st.checkbox(
                "Auto-Save Scans",
                value=st.session_state.auto_save_enabled,
                help="Automatically save scan results"
            )
        with col2:
            anonymize_exports = st.checkbox(
                "Anonymize Patient Data on Export",
                value=True,
                help="Remove identifying information from exported files"
            )
        st.divider()
        st.write("**Data Deletion & Privacy**")
        col1, col2 = st.columns(2)
        with col1:
            data_retention_days = st.selectbox(
                "Auto-Delete Scan History After (Days)",
                ["Never", "30", "60", "90", "180"],
                help="Automatically delete old scans to save storage"
            )
        with col2:
            log_retention_days = st.selectbox(
                "Log Retention Period",
                ["7 days", "30 days", "60 days", "Never"],
                help="How long to keep activity logs"
            )
        st.divider()
        st.write("**Data Export & Clear Cache**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export All Data", key="export_data_btn", use_container_width=True):
                st.info("Exporting data... This feature would export patient data to CSV/JSON format")
        
        with col2:
            if st.button("Clear Cache", key="clear_cache_btn", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
        
        with col3:
            if st.button("Delete All Data", key="delete_all_data_btn", use_container_width=True):
                st.warning("This action cannot be undone!")
                confirm_delete = st.button("Confirm Deletion", key="confirm_delete_btn")
                if confirm_delete:
                    st.error("All data would be deleted (confirmation step)")
        st.divider()
        st.write("**GDPR & Compliance**")
        st.checkbox("I understand the privacy policy", help="Acknowledge GDPR compliance")
        st.checkbox("Enable HIPAA compliant mode", help="Enforce HIPAA regulations")
    # ===== ABOUT PAGE =====
    with tab7:
        st.subheader("About NeuroVision AI")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Application Information**")
            info_data = {
                "Application": "NeuroVision AI",
                "Version": "2.0.1",
                "Release Date": "March 2024",
                "Model": "EfficientNetB0 (Transfer Learning)",
                "Input Size": "224 × 224 pixels",
                "Test Accuracy": "87%",
                "Framework": "TensorFlow/Keras + Streamlit"
            }
            for key, val in info_data.items():
                st.write(f"**{key}:** {val}")
        with col2:
            st.write("**Classification Classes**")
            classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
            for cls in classes:
                st.write(cls)
        st.divider()
        st.write("**System Requirements**")
        requirements = {
            "Python": "3.8+",
            "RAM": "4GB minimum",
            "Storage": "2GB for model files",
            "GPU": "Optional (NVIDIA CUDA)"
        }
        for req, spec in requirements.items():
            st.write(f"• {req}: {spec}")
        st.divider()
        st.write("**Features**")
        features = [
            "MRI Image Classification (4 Classes)",
            "Real-time Prediction with Confidence Scores",
            "Grad-CAM Visualization for Explainability",
            "Patient Management System",
            "Scan History Tracking",
            "Clinical Decision Support",
            "Detailed Performance Metrics",
        ]
        for feature in features:
            st.write(feature)
        st.divider()
        st.write("**Development Team**")
        st.write("Developed with love for clinical decision support in neuroimaging")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Documentation", key="docs_btn", use_container_width=True)
        with col2:
            st.button("Report Issue", key="issue_btn", use_container_width=True)
        with col3:
            st.button("GitHub Repo", key="github_btn", use_container_width=True)
# ================= PATIENT DASHBOARD (Logged-In Users Only) =================
elif st.session_state.logged_in:
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
            if st.button("Back to Dashboard", key="back_to_dashboard"):
                st.session_state.current_scan = None
            st.rerun()

        st.info("This is AI-assisted analysis. Final diagnosis must be made by a qualified radiologist.")
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
        # Grad-CAM with Caching
        if st.session_state.preprocessed_img is not None and st.session_state.uploaded_img is not None:
            st.subheader("AI Focus Areas (Grad-CAM)")
            try:
                processed = st.session_state.preprocessed_img
                if processed.shape != (1, 224, 224, 3):
                    raise ValueError(f"Unexpected preprocessed image shape: {processed.shape}")

                # Check cache first
                scan_id = st.session_state.current_scan
                if scan_id not in st.session_state.gradcam_cache:
                    with st.spinner("Computing AI Focus Areas (Grad-CAM)..."):
                        st.session_state.gradcam_cache[scan_id] = generate_gradcam(model, processed)
                
                heatmap = st.session_state.gradcam_cache[scan_id]

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
        st.header("NeuroVision AI - Patient Selection")
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
        if st.button("Add New Patient", key="add_patient_btn"):
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
        st.subheader("New MRI Consultation")
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

                    if st.button("Run Analysis", key="run_analysis"):
                        with st.spinner("Analyzing MRI scan..."):
                            # Preprocess once and reuse for predict and Grad-CAM
                            preprocessed_img = preprocess_image(img)
                            try:
                                label, confidence, probs = predict_image(preprocessed_img, model, class_labels, preprocessed=True)
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
        with st.expander("Scan History", expanded=False):
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
                    if col5.button("View", key=f"view_{row['id']}"):
                        st.session_state.current_scan = row['id']
                        st.rerun()
                    if col6.button("Delete", key=f"delete_{row['id']}"):
                        try:
                            pm.delete_scan_record(st.session_state.current_patient, row['id'])
                            st.success("Scan deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete scan: {e}")
            else:
                st.info("No scans available for this patient.")