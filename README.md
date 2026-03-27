# Weapon Detection using AI (Image/Video + Real-time)

An end-to-end demo that detects **weapon-like objects** in **images** and **videos**, annotates results with bounding boxes, and supports **real-time** detection scenarios.

## Features

- Image weapon detection (upload → annotated output)
- Video weapon detection (upload → annotated output MP4)
- Real-time detection from webcam or a video source
- Simple web UI built with **HTML/CSS/JavaScript**

## Tech

- **Backend**: Python, FastAPI, OpenCV, Ultralytics YOLO
- **Frontend**: HTML, CSS, JavaScript

## Setup

Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the web app

```bash
uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

> First run will download YOLO weights automatically (internet required once).

## Real-time webcam detection

```bash
python -m backend.realtime --source 0
```

## Notes / Accuracy

This demo uses a general-purpose detector and filters detections by weapon-related labels.
For higher accuracy in production, you typically train/fine-tune on a dedicated weapon dataset,
apply augmentations (blur/low light/angles), and calibrate confidence thresholds.

