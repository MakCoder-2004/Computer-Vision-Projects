import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import torch
import time

# ---------------------------
# Check Using GPU
# ---------------------------
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# ---------------------------
# Load Models
# ---------------------------
# YOLO
model = YOLO("./models/yolo26l.pt")
model.to('cuda')  # Force model to use GPU

# SORT Tracker
tracker = Sort(max_age=12, min_hits=3, iou_threshold=0.5)

# Line Limits for Counting
line1_limit = [(250, 660), (650, 660)]
line2_limit = [(760, 520), (1020, 520)]

# Counters
total_count_down = []
total_count_up = []

# Track previous center positions for each ID
prev_centers = {}

# ---------------------------
# Initialize Webcam
# ---------------------------
cap = cv2.VideoCapture("./video.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Set resolution
cap.set(3, 1280)
cap.set(4, 720)

# Initialize VideoWriter for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, out_fps, (frame_width, frame_height))

prev_time = 0

# ---------------------------
# Initialize Main Loop
# ---------------------------
try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # ---------------------------
        # Object Detection
        # ---------------------------
        detections= np.empty((0, 5))

        results = model(frame, stream=True)
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Confidence
                conf = box.conf[0].item()
                # Class
                cls = result.names[int(box.cls[0].item())]
                # centers
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cls in ["car", "truck", "bus", "motorcycle"] and conf > 0.5:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                    cvzone.putTextRect(
                        frame,
                        f"{cls} {conf:.2f}",
                        (max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1
                    )

        # Always update tracker, even if no detections
        TrackingResults = tracker.update(detections)
        for result in TrackingResults:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # Draw Line Limits
            cv2.line(frame, line1_limit[0], line1_limit[1], (0, 255, 0), 2)
            cv2.line(frame, line2_limit[0], line2_limit[1], (0, 0, 255), 2)

            # Track previous center for each ID
            if id in prev_centers:
                prev_cx, prev_cy = prev_centers[id]
                # Down line crossing (y increases past line1)
                if prev_cy < line1_limit[0][1] and cy >= line1_limit[0][1]:
                    if id not in total_count_down:
                        total_count_down.append(id)
                        cv2.line(frame, line1_limit[0], line1_limit[1], (0, 255, 255), 5)
                # Up line crossing (y decreases past line2)
                if prev_cy > line2_limit[0][1] and cy <= line2_limit[0][1]:
                    if id not in total_count_up:
                        total_count_up.append(id)
                        cv2.line(frame, line2_limit[0], line2_limit[1], (255, 0, 255), 5)
            prev_centers[id] = (cx, cy)

        # Display counts professionally
        cvzone.putTextRect(frame, f'Count Down: {len(total_count_down)}', (50, 50), scale=2, thickness=2, colorR=(0,255,0), colorT=(0,0,0))
        cvzone.putTextRect(frame, f'Count Up: {len(total_count_up)}', (50, 120), scale=2, thickness=2, colorR=(0,0,255), colorT=(0,0,0))

        # ---------------------------
        # Visualizing The Detections
        # ---------------------------
        cv2.imshow("Cars Counter Detection", frame)
        out.write(frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

except Exception as e:
    print("Error occurred:", e)

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Camera released. Program closed safely.")