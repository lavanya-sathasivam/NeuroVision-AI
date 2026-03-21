import numpy as np
import json
import tensorflow as tf
import cv2
from pathlib import Path

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
        class_indices_int = {k: int(v) for k, v in class_indices.items()}
        # Validate contiguous mapping
        indices = sorted(class_indices_int.values())
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Invalid class_indices mapping. Expected contiguous 0..{len(indices)-1}, got {indices}"
            )

        class_labels = {v: k for k, v in class_indices_int.items()}

        print("[DEBUG] class_labels mapping:")
        for idx in sorted(class_labels):
            print(f"  {idx} -> {class_labels[idx]}")

        return model, class_labels

    except Exception as e:
        raise RuntimeError(f"Error loading model or labels: {e}")

# ================= PREPROCESS =================
def preprocess_image(img):
    """
    Preprocess image for model input.
    Ensures RGB format, correct size, and proper normalization.
    """
    if img is None:
        raise ValueError("Image is None")

    if not isinstance(img, np.ndarray):
        raise ValueError("Image must be numpy array")

    if img.size == 0:
        raise ValueError("Image is empty")

    if len(img.shape) not in [2, 3]:
        raise ValueError("Unsupported image dimensions")

    # Convert grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Ensure 3 channels
    if img.shape[2] != 3:
        raise ValueError("Image must have 3 color channels")

    # Resize safely
    try:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        raise ValueError(f"Resize failed: {e}")

    # Convert BGR → RGB if needed (OpenCV loads as BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] - remove preprocess_input for consistency
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# ================= PREDICT =================
def predict_image(img, model, class_labels):
    """
    Make prediction with robust error handling.
    """
    try:
        processed = preprocess_image(img)

        preds = model.predict(processed, verbose=0)
        if preds is None or len(preds) == 0:
            raise ValueError("Model returned no predictions")

        preds = preds[0]  # Get first prediction

        if np.isnan(preds).any():
            raise ValueError("Model output contains NaN")

        if not np.isclose(np.sum(preds), 1.0, atol=0.1):
            raise ValueError("Invalid probability distribution")

        class_idx = int(np.argmax(preds))
        if class_idx not in class_labels:
            raise ValueError(f"Invalid class index {class_idx}")

        label = class_labels[class_idx]
        confidence = float(np.max(preds))

        return label, confidence, preds

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

# ================= CONFIDENCE CLASSIFICATION =================
def classify_confidence(confidence):
    """
    Classify confidence level with UNCERTAIN threshold.
    """
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.6:
        return "MEDIUM"
    else:
        return "UNCERTAIN"

# ================= SAFE PREDICT =================
def safe_predict(img, model, class_labels):
    """
    Safe prediction with validation and confidence thresholding.
    """
    try:
        # Validate image first
        valid, error_msg = validate_image(img)
        if not valid:
            return False, error_msg

        label, confidence, preds = predict_image(img, model, class_labels)
        confidence_level = classify_confidence(confidence)

        # Apply confidence threshold
        if confidence_level == "UNCERTAIN":
            return False, "Prediction confidence too low - UNCERTAIN result"

        return True, {
            "label": label,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "probs": preds
        }

    except Exception as e:
        return False, f"Analysis failed: {str(e)}"

# ================= IMAGE VALIDATION =================
def validate_image(img):
    """
    Validate uploaded image for medical use.
    Checks size, brightness, and basic integrity.
    """
    if img is None or img.size == 0:
        return False, "Invalid or empty image"

    if len(img.shape) not in [2, 3]:
        return False, "Unsupported image format"

    height, width = img.shape[:2]
    if height < 100 or width < 100:
        return False, "Image resolution too low (minimum 100x100)"

    if height > 4096 or width > 4096:
        return False, "Image resolution too high (maximum 4096x4096)"

    # Check brightness (avoid completely black/dark images)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mean_brightness = np.mean(gray)
    if mean_brightness < 10:
        return False, "Image appears too dark - please check scan quality"

    return True, None

# ================= CLINICAL DECISION =================
def clinical_decision(label, confidence_level):
    """
    Make clinical decision based on prediction and confidence.
    """
    if confidence_level == "UNCERTAIN":
        return "UNCERTAIN", "Radiologist review REQUIRED. AI confidence too low."

    if label == "notumor" and confidence_level == "HIGH":
        return "SAFE", "No tumor detected. Routine follow-up recommended."

    if label != "notumor" and confidence_level == "HIGH":
        return "ALERT", "Tumor likely. Immediate specialist consultation required."

    return "CAUTION", "Further imaging and specialist review recommended."


# ================= MODEL TEST UTILS =================
def evaluate_known_images(model, class_labels, samples):
    """
    Evaluate model on known labeled samples.

    samples: list of (image_array, expected_label)
    """
    results = []

    for idx, (img, expected_label) in enumerate(samples):
        try:
            label, confidence, preds = predict_image(img, model, class_labels)
            is_correct = label.lower() == expected_label.lower()
            results.append({
                "index": idx,
                "expected": expected_label,
                "predicted": label,
                "confidence": confidence,
                "correct": is_correct,
                "preds": preds.tolist() if isinstance(preds, np.ndarray) else preds
            })
        except Exception as e:
            results.append({
                "index": idx,
                "expected": expected_label,
                "predicted": None,
                "confidence": 0.0,
                "correct": False,
                "error": str(e)
            })

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    uncertain = sum(1 for r in results if r.get("confidence", 0) < 0.6)

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": float(correct / total) if total else 0.0,
        "uncertain": uncertain,
        "uncertain_ratio": float(uncertain / total) if total else 0.0,
        "results": results,
    }

    # Underfit/misclassify check
    if summary["accuracy"] < 0.70:
        summary["status"] = "weak_model"
        summary["recommendation"] = (
            "Low accuracy on known images. Retrain with more data, augmentations, class balance and consistent preprocessing."
        )
    else:
        summary["status"] = "ok"

    return summary
