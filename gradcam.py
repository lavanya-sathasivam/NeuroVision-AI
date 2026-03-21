import tensorflow as tf
import numpy as np
import cv2

# ================= FIND LAST CONV LAYER =================
def find_last_conv_layer(model):
    """
    Automatically find the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model")

# ================= GENERATE GRAD-CAM =================
def generate_gradcam(model, img_array):
    """
    Generate Grad-CAM heatmap for the given image and model.
    Returns None if generation fails or heatmap is invalid.
    """
    try:
        if model is None or img_array is None:
            return None

        if img_array.shape[0] != 1:
            return None  # Must be single image batch

        last_conv_layer_name = find_last_conv_layer(model)

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            if predictions is None or tf.reduce_any(tf.math.is_nan(predictions)):
                return None

            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None or tf.reduce_any(tf.math.is_nan(grads)):
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs * pooled_grads[..., tf.newaxis, tf.newaxis]
        heatmap = tf.reduce_sum(heatmap, axis=-1)

        heatmap = tf.maximum(heatmap, 0)

        max_val = tf.reduce_max(heatmap)
        if max_val == 0 or tf.math.is_nan(max_val) or tf.math.is_inf(max_val):
            return None

        heatmap /= max_val
        heatmap = tf.squeeze(heatmap)

        # Final validation
        if tf.reduce_any(tf.math.is_nan(heatmap)) or tf.reduce_any(tf.math.is_inf(heatmap)):
            return None

        return heatmap.numpy()

    except Exception as e:
        print(f"[GradCAM ERROR]: {e}")
        return None

# ================= OVERLAY HEATMAP =================
def overlay_heatmap(original_img, heatmap):
    """
    Overlay the Grad-CAM heatmap on the original image.
    Returns None if overlay fails.
    """
    try:
        if heatmap is None:
            raise ValueError("Heatmap is None")

        if np.isnan(heatmap).any() or np.isinf(heatmap).any() or heatmap.size == 0:
            raise ValueError("Invalid heatmap")

        if original_img is None or original_img.size == 0:
            raise ValueError("Invalid original image")

        # Ensure heatmap is 2D
        if len(heatmap.shape) != 2:
            raise ValueError("Heatmap must be 2D")

        # Resize heatmap to match original image size safely
        try:
            heatmap_resized = cv2.resize(
                heatmap.astype(np.float32),
                (original_img.shape[1], original_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        except Exception as e:
            raise ValueError(f"Heatmap resize failed: {e}")

        # Validate resized heatmap
        if np.isnan(heatmap_resized).any() or np.isinf(heatmap_resized).any():
            raise ValueError("Resized heatmap contains invalid values")

        # Normalize and convert to uint8
        heatmap_max = np.max(heatmap_resized)
        if heatmap_max == 0 or not np.isfinite(heatmap_max):
            raise ValueError("Heatmap has zero or invalid maximum value")

        heatmap_normalized = heatmap_resized / heatmap_max
        heatmap_uint8 = np.uint8(255 * heatmap_normalized)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Ensure images have same dimensions
        if heatmap_colored.shape[:2] != original_img.shape[:2]:
            raise ValueError("Dimension mismatch between heatmap and original image")

        # Overlay on original image
        overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

        return overlay

    except Exception as e:
        print(f"[Overlay ERROR]: {e}")
        return None