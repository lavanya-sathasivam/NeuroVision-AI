import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_model_and_labels, predict_image, preprocess_image
from gradcam import generate_gradcam, overlay_heatmap

st.set_page_config(page_title="NeuroVision AI", layout="wide")

st.title("NeuroVision AI")
st.subheader("Brain Tumor Detection with Explainable AI")

st.sidebar.header("Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png"])

@st.cache_resource
def load_resources():
    return load_model_and_labels()

model, class_labels = load_resources()

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded MRI", use_column_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            label, confidence, preds = predict_image(img, model, class_labels)
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        st.write("Class Probabilities:")
        st.bar_chart(preds)

    st.subheader("Grad-CAM Visualization")

    processed = preprocess_image(img)
    heatmap = generate_gradcam(model, processed)

    overlay = overlay_heatmap(cv2.resize(img, (224, 224)), heatmap)

    col3, col4 = st.columns(2)

    with col3:
        st.image(img, caption="Original")

    with col4:
        st.image(overlay, caption="Grad-CAM Heatmap")