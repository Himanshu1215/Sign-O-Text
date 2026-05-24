# Sign-O-Text: Real-Time Sign Language Recognition

Sign-O-Text is a high-performance, real-time sign language detection web application. It uses **MediaPipe Holistic** for landmark extraction and a **Stacked LSTM (Long Short-Term Memory)** neural network to recognize temporal sign sequences from a browser webcam.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10+-green.svg)

---

## 🚀 Key Features

*   **Real-Time Browser Inference**: Captures webcam frames at ~10 FPS and predicts signs with low latency.
*   **Optimized Landmark Pipeline**: Extracts only hand landmarks (126 dims) instead of full-body (1662 dims), reducing data footprint by 90% and improving speed.
*   **Temporal Sequence Modeling**: Uses a 30-frame sliding window to capture the movement dynamics of signs.
*   **Polished UI**: Dark-themed dashboard with real-time confidence bars, probability distributions, and prediction history.
*   **Robust Detection**: Implements Exponential Moving Average (EMA) smoothing and stability filtering to reduce flicker.

---

## 🛠️ Technical Architecture

### 1. Data Pipeline & Input/Output Specs

#### **A. Image Capture (Frontend)**
*   **Input**: Browser Webcam Stream (MediaDevices API).
*   **Format**: 640x480 RGB frames.
*   **Frequency**: 100ms interval (~10 FPS).
*   **Output**: Base64 encoded JPEG sent via POST to `/predict`.

#### **B. Feature Extraction (MediaPipe)**
*   **Engine**: MediaPipe Tasks - `HolisticLandmarker`.
*   **Input**: Raw RGB Image.
*   **Logic**: Extracts 21 landmarks for the Left Hand and 21 for the Right Hand.
*   **Feature Vector**: 126 total features (42 landmarks × 3 coordinates [x, y, z]).
*   **Normalization**: Coordinates are normalized (0.0 to 1.0) relative to the image dimensions by MediaPipe.

#### **C. Temporal Sequence (LSTM)**
*   **Input Shape**: `(Batch, 30, 126)`
    *   `30`: Number of consecutive frames (Temporal Window).
    *   `126`: Features per frame.
*   **Model Architecture**:
    *   `LSTM (64 units)` + BatchNormalization + Dropout (0.3).
    *   `LSTM (128 units)` + BatchNormalization + Dropout (0.3).
    *   `LSTM (64 units)` + BatchNormalization.
    *   `Dense (64 units, ReLU)` -> `Dense (32 units, ReLU)`.
    *   `Dense (7 units, Softmax)` (Final Classification).

---

## 📊 Classes & Accuracy

The current model is trained on **7 WLASL-derived classes**:
`hello`, `thanks`, `Father`, `Mother`, `Yes`, `No`, `Help`

| Metric | Value |
| :--- | :--- |
| **Sequence Length** | 30 Frames (~3 seconds of motion) |
| **Feature Dimension** | 126 (Hands Only) |
| **Prediction Confidence** | Thresholded at 0.5 |
| **Stability Filter** | Requires 3 consecutive identical predictions |

---

## 📂 Project Structure

```text
Sign-O-Text/
├── app.py                # Flask Web Server (Inference Engine)
├── model/
│   ├── action_wlasl_7.keras  # Trained LSTM Model
│   └── holistic_landmarker.task # MediaPipe Model Bundle
├── static/
│   ├── app.js            # Frontend logic (Webcam + UI)
│   └── style.css         # Polished Dark Theme
├── templates/
│   └── index.html        # UI Layout
├── scripts/
│   ├── process_wlasl_to_npy.py  # Data extraction script
│   └── optimized_dataloader.py  # Efficient TF training pipeline
└── notebooks/
    └── Retrain_WLASL_7_Classes.ipynb # Training Research
```

---

## ⚡ Installation & Usage

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/Sign-O-Text.git
cd Sign-O-Text
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
*   Wait for `[app] Model ready!` in the terminal.
*   Open `http://localhost:5000` in your browser.
*   **Important**: Allow camera access and sign with your **Right Hand** for best results.

---

## 📝 Future Improvements
*   **Global Normalization**: Implement wrist-relative coordinate normalization to make the model invariant to camera distance.
*   **Class Expansion**: Extend to the full WLASL-100 or ASL Alphabet dataset.
*   **Data Augmentation**: Add rotation and translation noise during training to improve real-world robustness.

---

## 🤝 Acknowledgments
*   **MediaPipe** for the lightning-fast holistic landmarking.
*   **WLASL Dataset** for provide the foundational sign language data.
*   **TensorFlow/Keras** for the temporal modeling capabilities.
