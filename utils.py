import numpy as np
import json
import tensorflow as tf
import cv2
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input

# ================= PATH CONFIG =================
MODEL_PATH = Path("model/final_brain_mri_model.keras")
LABELS_PATH = Path("model/class_indices.json")

# ================= LOAD MODEL =================
def load_model_and_labels():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        if not LABELS_PATH.exists():
            raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")

        model = tf.keras.models.load_model(MODEL_PATH)

        with open(LABELS_PATH, "r") as f:
            class_indices = json.load(f)

        # Convert index → label
        class_labels = {int(v): k for k, v in class_indices.items()}

        return model, class_labels

    except Exception as e:
        raise RuntimeError(f"Error loading model or labels: {e}")

# ================= PREPROCESS =================
def preprocess_image(img):
    if img is None:
        raise ValueError("Invalid image")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if len(img.shape) not in [2, 3]:
        raise ValueError("Unsupported image format")

    # Convert grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.expand_dims(img, axis=0)
    img = img / 255.0


    return img

# ================= PREDICT =================
def predict_image(img, model, class_labels):
    try:
        processed = preprocess_image(img)

        preds = model.predict(processed, verbose=0)[0]
        valid,error= validate_prediction(preds)
        
        if not valid:
            raise ValueError(error)
        
        if preds is None or len(preds) == 0:
            raise ValueError("Empty prediction output")

        if np.isnan(preds).any():
            raise ValueError("Model output contains NaN")

        if not np.isclose(np.sum(preds), 1.0, atol=0.05):
            raise ValueError("Invalid probability distribution")

        class_idx = int(np.argmax(preds))
        label = class_labels.get(class_idx, "unknown")

        confidence = float(np.max(preds))

        return label, confidence, preds

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

# ================= OPTIONAL: SAFE WRAPPER =================

def classify_confidence(confidence):
    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.60:
        return "MEDIUM"
    else:
        return "LOW"

def safe_predict(img, model, class_labels):
    try:
        label, confidence, preds = predict_image(img, model, class_labels)

        confidence_level = classify_confidence(confidence)

        return True, {
            "label": label,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "probs": preds
        }

    except Exception as e:
        return False, str(e) 
    
def validate_prediction(preds, threshold=0.5):
    if np.isnan(preds).any():
        return False, "Invalid model output (NaN detected)"

    if not np.isclose(np.sum(preds), 1.0, atol=0.05):
        return False, "Invalid probability distribution"

    confidence = float(np.max(preds))

    if confidence < threshold:
        return False, "Prediction confidence too low"

    return True, None

def clinical_decision(label, confidence_level):
    if confidence_level == "LOW":
        return "UNCERTAIN", "Radiologist review REQUIRED. Do not rely on AI."

    if label == "no tumor" and confidence_level == "HIGH":
        return "SAFE", "No tumor detected. Routine follow-up."

    if label != "no tumor" and confidence_level == "HIGH":
        return "ALERT", "Tumor likely. Immediate specialist consultation."

    return "CAUTION", "Further imaging recommended."