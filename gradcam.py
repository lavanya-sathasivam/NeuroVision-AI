import tensorflow as tf
import numpy as np
import cv2


# ================= FIND LAST CONV LAYER =================
def get_last_conv_layer(model):
    """
    Automatically find last convolutional layer
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


# ================= GENERATE GRAD-CAM =================

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")

def generate_gradcam(model, img_array):
    try:
        last_conv_layer_name = find_last_conv_layer(model)

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            if predictions is None:
                raise ValueError("Predictions are None")

            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Gradients are None")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)

        max_val = tf.reduce_max(heatmap)

        if max_val == 0:
            raise ValueError("Heatmap max is zero")

        heatmap /= max_val

        return heatmap.numpy()

    except Exception as e:
        print(f"[GradCAM ERROR]: {e}")
        return None

# ================= OVERLAY HEATMAP =================
def overlay_heatmap(original_img, heatmap):
    if heatmap is None:
        raise ValueError("Heatmap is None")

    if np.isnan(heatmap).any():
        raise ValueError("Heatmap contains NaN")

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return overlay