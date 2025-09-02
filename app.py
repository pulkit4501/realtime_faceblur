import gradio as gr
import cv2
from deepface import DeepFace
import numpy as np
import tensorflow as tf
import time

# --- 1. Verify GPU and Prepare Model ---
print("GPU available: ", tf.config.list_physical_devices('GPU'))
print("DeepFace is ready.")

TARGET_FPS = 5                       # <-- Adjust this (e.g., 4 or 5)
MIN_INTERVAL = 1.0 / TARGET_FPS

# --- 2. The Processing Function (with throttling) ---
def blur_frame_throttled(frame, state):
    """
    Throttle processing to ~TARGET_FPS. Only reprocess if MIN_INTERVAL elapsed.
    Otherwise, return the last processed frame to minimise latency.
    """
    if frame is None:
        return None, state

    now = time.time()
    last_t = state.get("last_t", 0.0)
    last_output = state.get("last_output", None)

    # If not enough time has passed, reuse the last processed output (if available)
    if (now - last_t) < MIN_INTERVAL and last_output is not None:
        return last_output, state

    # Otherwise process a fresh frame
    original = frame.copy()

    # Resize down for speed (trade-off between speed and quality)
    h, w, _ = original.shape
    scale = 320 / w if w > 0 else 1.0
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized_frame = cv2.resize(original, (new_w, new_h))

    try:
        detected_faces = DeepFace.extract_faces(
            img_path=resized_frame,
            detector_backend='centerface',
            enforce_detection=False
        )

        for face_obj in detected_faces:
            box = face_obj['facial_area']
            # Scale coordinates back to the original frame size
            x = int(box['x'] / scale)
            y = int(box['y'] / scale)
            w_box = int(box['w'] / scale)
            h_box = int(box['h'] / scale)

            # Clamp to frame bounds
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(original.shape[1], x + w_box)
            y1 = min(original.shape[0], y + h_box)

            if x1 > x0 and y1 > y0:
                face_roi = original[y0:y1, x0:x1]
                if face_roi.size > 0:
                    blurred_face = cv2.medianBlur(face_roi, 39)
                    original[y0:y1, x0:x1] = blurred_face

    except Exception as e:
        print(f"Error processing frame: {e}")

    # Update state
    state["last_t"] = now
    state["last_output"] = original

    return original, state

# --- 3. Create and Launch the Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## 📹 Live Webcam Face Blurrer (Throttled to ~5 fps)")

    video_stream = gr.Image(sources="webcam", streaming=True, label="Real-Time Stream")
    state = gr.State({"last_t": 0.0, "last_output": None})

    # Stream: returns both the updated frame and the updated state
    video_stream.stream(
        fn=blur_frame_throttled,
        inputs=[video_stream, state],
        outputs=[video_stream, state]
    )

demo.queue().launch(debug=True)
