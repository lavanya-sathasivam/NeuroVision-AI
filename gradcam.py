import tensorflow as tf
import numpy as np
import cv2


# ================= FIND LAST CONV LAYER =================
def find_last_conv_layer(model):
    """
    Robustly find last Conv2D layer even inside nested models.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        # check nested model (EfficientNet case)
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name

    raise ValueError("No Conv2D layer found in model")


# ================= GENERATE GRAD-CAM =================
def generate_gradcam(model, img_array):
    """
    Generate Grad-CAM heatmap.
    Returns None only if something is truly broken.
    """
    try:
        if model is None or img_array is None:
            print("[GradCAM] Invalid model or image")
            return None

        if len(img_array.shape) != 4 or img_array.shape[0] != 1:
            print("[GradCAM] Input must be batch of 1")
            return None

        # 🔥 FIND LAST CONV LAYER
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"[GradCAM] Using layer: {last_conv_layer_name}")

        # 🔥 BUILD GRAD MODEL
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

        # 🔥 FORWARD + GRADIENT
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            if predictions is None or tf.reduce_any(tf.math.is_nan(predictions)):
                print("[GradCAM] Invalid predictions")
                return None

            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("[GradCAM] Gradients are None")
            return None

        if tf.reduce_any(tf.math.is_nan(grads)):
            print("[GradCAM] NaN gradients")
            return None

        # 🔥 GLOBAL AVERAGE POOLING
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 🔥 FEATURE MAP
        conv_outputs = conv_outputs[0]

        # 🔥 CORRECT BROADCASTING (CRITICAL FIX)
        pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))

        heatmap = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(heatmap, axis=-1)

        # 🔥 RELU
        heatmap = tf.maximum(heatmap, 0)

        # 🔥 NORMALIZE
        max_val = tf.reduce_max(heatmap)

        if not tf.math.is_finite(max_val) or max_val < 1e-6:
            print("[GradCAM] Invalid heatmap max")
            return None

        heatmap = heatmap / max_val
        heatmap = tf.squeeze(heatmap)

        if tf.reduce_any(tf.math.is_nan(heatmap)):
            print("[GradCAM] NaN heatmap")
            return None

        return heatmap.numpy()

    except Exception as e:
        print(f"[GradCAM ERROR]: {e}")
        return None


# ================= OVERLAY HEATMAP =================
def overlay_heatmap(original_img, heatmap):
    """
    Overlay Grad-CAM heatmap on original image.
    """
    try:
        if heatmap is None:
            print("[Overlay] Heatmap is None")
            return None

        if original_img is None or original_img.size == 0:
            print("[Overlay] Invalid original image")
            return None

        # Ensure heatmap is 2D
        if len(heatmap.shape) != 2:
            print("[Overlay] Heatmap must be 2D")
            return None

        # Resize heatmap
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32),
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize
        heatmap_resized = np.maximum(heatmap_resized, 0)
        max_val = np.max(heatmap_resized)

        if max_val < 1e-6:
            print("[Overlay] Heatmap max too small")
            return None

        heatmap_resized /= max_val
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Convert original to BGR if needed
        if original_img.shape[-1] == 3:
            overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        else:
            print("[Overlay] Unexpected image format")
            return None

        return overlay

    except Exception as e:
        print(f"[Overlay ERROR]: {e}")
        return None