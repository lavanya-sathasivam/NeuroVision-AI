import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2


def load_model_and_labels():
    model = tf.keras.models.load_model("model/final_brain_mri_model.keras")

    with open("model/class_indices.json") as f:
        class_indices = json.load(f)

    class_labels = {v: k for k, v in class_indices.items()}

    return model, class_labels


def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def predict_image(img, model, class_labels):
    processed = preprocess_image(img)
    preds = model.predict(processed)[0]

    class_idx = int(np.argmax(preds))
    label = class_labels.get(class_idx, "unknown")
    confidence = float(np.max(preds))

    return label, confidence, preds
