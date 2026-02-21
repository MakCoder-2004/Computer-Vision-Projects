# Cars Counter

## Overview

Cars Counter is a computer vision project that performs real-time vehicle detection, tracking, and counting using YOLO object detection models and the SORT (Simple Online and Realtime Tracking) algorithm.

The system processes a video file, detects vehicles such as cars, buses, trucks, and motorcycles, tracks them across frames, and counts them when they cross predefined virtual lines. The processed output is exported as a new video with visual overlays.

This project is intended for educational and research purposes.

---

## Features

* Vehicle detection using YOLO models
* Real-time multi-object tracking using SORT
* Bidirectional vehicle counting (up and down directions)
* Unique ID assignment for each tracked vehicle
* Robust logic to prevent double counting
* Automatic GPU utilization if available (CUDA support)
* Export of processed video with bounding boxes, IDs, and counters

---

## How It Works

### 1. Detection

Each frame of the input video is processed using a YOLO model to detect vehicles.

### 2. Tracking

Detected objects are passed to the SORT tracker, which assigns a unique ID to each vehicle and maintains identity consistency across frames.

### 3. Counting

When a tracked vehicle crosses a predefined counting line, it is counted once. Previous positions are analyzed to prevent duplicate counting and reduce ID-switching errors.

### 4. Visualization

Bounding boxes, tracking IDs, and vehicle counts are drawn on each frame. The final processed video is saved as `output.mp4`.

---

## Project Structure

```
Cars-Counter/
│
├── CarCounter.py      # Main script (detection, tracking, counting, visualization)
└── sort.py            # SORT tracking algorithm implementation
```

---

## Requirements

* Python 3.8 or higher
* numpy
* torch
* ultralytics
* cvzone
* opencv-python

Install dependencies using:

```bash
pip install opencv-python cvzone ultralytics torch numpy
```

---

## Usage

### Step 1: Add Input Video

Place your input video in the project directory and name it:

```
video.mp4
```

### Step 2: Add YOLO Model Weights

Place the YOLO model weights inside the `models/` directory.

Example:

```
models/yolo26l.pt
```

### Step 3: Run the Application

```bash
python CarCounter.py
```

### Step 4: Output

After execution, the processed video will be saved as:

```
output.mp4
```

---

## Customization

You can modify the following parameters inside `CarCounter.py`:

* Counting line positions
* Tracker parameters (`max_age`, `min_hits`, `iou_threshold`)
* YOLO model selection (e.g., nano or large variants)

---
