import streamlit as st
import numpy as np
import cv2
from utils import load_model_and_labels, predict_image, preprocess_image
from gradcam import generate_gradcam, overlay_heatmap

st.set_page_config(page_title="NeuroVision AI", layout="wide")

@st.cache_resource
def load_resources():
    return load_model_and_labels()

model, class_labels = load_resources()

st.title("NeuroVision AI")
st.subheader("Brain Tumor Detection with Explainable AI")

st.markdown("""
### AI-powered MRI Brain Tumor Detection System

Upload an MRI scan to:
- Detect tumor type  
- View confidence scores  
- Visualize model attention using Grad-CAM  
""")

st.sidebar.title("📤 Upload Section")
uploaded_file = st.sidebar.file_uploader("Upload MRI Image", type=["jpg", "png"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded MRI", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing MRI..."):
            label, confidence, preds = predict_image(img, model, class_labels)

        st.success(f"Prediction: {label.upper()}")
        st.metric("Confidence", f"{confidence*100:.2f}%")

        st.markdown("### 📊 Class Probabilities")
        st.bar_chart(preds)

    st.markdown("---")
    st.subheader("Grad-CAM Visualization")

    processed = preprocess_image(img)
    heatmap = generate_gradcam(model, processed)

    overlay = overlay_heatmap(cv2.resize(img, (224, 224)), heatmap)

    col3, col4 = st.columns(2)

    with col3:
        st.image(img, caption="Original MRI", use_column_width=True)

    with col4:
        st.image(overlay, caption="Model Attention (Grad-CAM)", use_column_width=True)

else:
    st.info("Upload an MRI image from the sidebar to start analysis")

    st.markdown("""
    ### How it works:
    1. Upload MRI image  
    2. AI analyzes the scan  
    3. Predicts tumor type  
    4. Highlights important regions  

    ---
    ### Supported Classes:
    - Glioma  
    - Meningioma  
    - Pituitary  
    - No Tumor  
    """)