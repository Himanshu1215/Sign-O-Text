"""
Sign-O-Text Flask Web App
==========================
Real-time sign language detection from browser webcam.

Flow:
  1. Browser captures webcam frame -> sends as base64 JPEG to /predict
  2. Server runs MediaPipe HolisticLandmarker -> extracts 126-dim hand landmarks
  3. Server maintains a sliding window buffer (30 frames)
  4. When buffer is full, runs LSTM model -> returns prediction + confidence
  5. Browser displays the result with animated UI

Usage:
    python app.py
    Open http://localhost:5000 in your browser.
"""

import base64
import threading
import uuid
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

import mediapipe as mp
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASSES = ['hello', 'thanks', 'Father', 'Mother', 'Yes', 'No', 'Help']
SEQUENCE_LENGTH = 30
HAND_DIM = 63       # 21 landmarks x 3 (x, y, z)
FEATURE_LENGTH = 126  # left + right hand

MODEL_PATH = Path("model/action_wlasl_7.keras")
HOLISTIC_PATH = Path("model/holistic_landmarker.task")

CONFIDENCE_THRESHOLD = 0.5     # minimum confidence to show a prediction
STABILITY_FRAMES = 3           # same prediction N times before switching
SMOOTHING_ALPHA = 0.3          # EMA factor for confidence smoothing

# ---------------------------------------------------------------------------
# Lazy-loaded model & detector (thread-safe)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()
_detector = None
_detector_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import tensorflow as tf
                _model = tf.keras.models.load_model(str(MODEL_PATH))
                print(f"[app] Model loaded: {MODEL_PATH.name} ({_model.count_params():,} params)")
    return _model


def get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                base = mp.tasks.BaseOptions(model_asset_path=str(HOLISTIC_PATH))
                opts = vision.HolisticLandmarkerOptions(
                    base_options=base,
                    running_mode=vision.RunningMode.IMAGE,
                )
                _detector = vision.HolisticLandmarker.create_from_options(opts)
                print(f"[app] HolisticLandmarker loaded")
    return _detector


# ---------------------------------------------------------------------------
# Per-session buffers
# ---------------------------------------------------------------------------
sessions = {}


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "buffer": [],
            "smoothed": None,
            "last_prediction": None,
            "stable_count": 0,
            "no_hand_frames": 0,
        }
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------
def extract_hand_keypoints(result) -> np.ndarray:
    lh = np.zeros(HAND_DIM, dtype=np.float32)
    rh = np.zeros(HAND_DIM, dtype=np.float32)
    if result.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks],
            dtype=np.float32,
        ).flatten()
    if result.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks],
            dtype=np.float32,
        ).flatten()
    return np.concatenate([lh, rh])


def hands_visible(result) -> bool:
    return bool(result.left_hand_landmarks or result.right_hand_landmarks)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "sign-o-text-secret-change-in-production"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image data"}), 400

    try:
        header, encoded = image_data.split(",", 1)
    except ValueError:
        header, encoded = "", image_data

    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    detector = get_detector()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    kp = extract_hand_keypoints(result)
    has_hands = hands_visible(result)

    sess = get_session(session_id)
    buffer = sess["buffer"]
    buffer.append(kp)
    if len(buffer) > SEQUENCE_LENGTH * 2:
        buffer[:] = buffer[-(SEQUENCE_LENGTH + 5):]

    response = {
        "buffer_size": len(buffer),
        "buffer_capacity": SEQUENCE_LENGTH,
        "has_hands": has_hands,
        "prediction": None,
        "confidence": 0.0,
        "all_probs": {},
    }

    if len(buffer) >= SEQUENCE_LENGTH and has_hands:
        model = get_model()
        input_seq = np.array(buffer[-SEQUENCE_LENGTH:], dtype=np.float32)
        input_seq = np.expand_dims(input_seq, axis=0)
        raw_probs = model.predict(input_seq, verbose=0)[0]

        if sess["smoothed"] is None:
            sess["smoothed"] = raw_probs.copy()
        else:
            sess["smoothed"] = (
                SMOOTHING_ALPHA * raw_probs + (1 - SMOOTHING_ALPHA) * sess["smoothed"]
            )

        probs = sess["smoothed"]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        predicted_class = CLASSES[class_idx]

        if (
            sess["last_prediction"] is not None
            and sess["last_prediction"]["class"] == predicted_class
        ):
            sess["stable_count"] += 1
        else:
            sess["stable_count"] = 1

        sess["last_prediction"] = {"class": predicted_class, "confidence": confidence}

        if sess["stable_count"] >= STABILITY_FRAMES and confidence >= CONFIDENCE_THRESHOLD:
            response["prediction"] = predicted_class
            response["confidence"] = round(confidence, 4)
            response["all_probs"] = {
                CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))
            }
            response["stable_count"] = sess["stable_count"]

    if not has_hands:
        sess["no_hand_frames"] += 1
    else:
        sess["no_hand_frames"] = 0
    response["no_hand_frames"] = sess["no_hand_frames"]

    return jsonify(response)


@app.route("/reset", methods=["POST"])
def reset_buffer():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "ok"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_exists": MODEL_PATH.exists()})


if __name__ == "__main__":
    print(f"[app] Sign-O-Text starting...")
    print(f"[app] Classes: {CLASSES}")
    print(f"[app] Open http://localhost:5000 in your browser")
    print(f"[app] Hold a hand up to the camera to start detecting signs!\n")

    print("[app] Pre-warming MediaPipe detector...")
    get_detector()

    print("[app] Warming up model (first inference may be slow)...")
    warmup = np.zeros((1, SEQUENCE_LENGTH, FEATURE_LENGTH), dtype=np.float32)
    get_model().predict(warmup, verbose=0)
    print("[app] Model ready!")

    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
