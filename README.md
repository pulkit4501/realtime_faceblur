---
title: Realtime Faceblur
emoji: 🐠
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
---

# 📹 Live Webcam Face Blurrer

A real-time, browser-based facial censorship application using Gradio's streaming functionality and DeepFace for facial detection. This application processes your live webcam feed to automatically detect and blur faces on the fly.

## Features

- **Real-Time Streaming**: Captures frames directly from your webcam using a live Gradio stream.
- **Throttled Processing**: To minimize latency and provide a smooth user experience, background processing is intelligently throttled to ~5 frames per second. Previous processed frames are seamlessly interpolated during detection intervals.
- **High-Performance Detection**: Uses the `centerface` backend via DeepFace for rapid detection optimized for real-time environments.
- **Privacy First**: Fully local processing within the Python backend; applies a heavy median blur to obscure identities safely.

## How it Works

1. The app captures your webcam input using a `gr.Image(streaming=True)` Gradio component.
2. The incoming stream is passed to a backend function that maintains a lightweight processing state.
3. Every 1/5th of a second (~5 FPS targeted), the newest frame is computationally downscaled for fast coordinate detection using DeepFace.
4. Any bounding box identified is scaled back to original resolution and subjected to a median blur via OpenCV.
5. The processed, blurred output is instantly streamed back to the browser window.

## Running Locally

### Prerequisites

You need Python 3.8+ installed, alongside a functioning webcam.

### Installation

1. Clone the repository and navigate into the `realtime_faceblur` directory.
2. Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### Usage

Run the app from the terminal:

```bash
python app.py
```

Click the provided local link (usually `http://127.0.0.1:7860`) to open the app in your browser, allow camera permissions when prompted, and observe the live blurring.

## Built With

- **[Gradio](https://gradio.app/)**: Handling bidirectional real-time streaming interfaces.
- **[DeepFace](https://github.com/serengil/deepface)**: Face detection model management using the CenterFace backend.
- **[TensorFlow / Keras](https://www.tensorflow.org/)**: Underlying ML processing backend for the active models.
- **[OpenCV](https://opencv.org/)**: Frame interpolation, resizing, and bounding-box blurring.
