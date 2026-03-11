import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision

# ==========================================================
# Load Detection Model
# ==========================================================
def load_detection_model(model_path, callback=None):

    HandLandmarkerOptions = vision.HandLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = vision.RunningMode
    HandLandmarker = vision.HandLandmarker

    if callback:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=callback,
            num_hands=2, # Ensure both hands can be tracked simultaneously
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
    else:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )

    landmarker = HandLandmarker.create_from_options(options)
    return landmarker

# ==========================================================
# Detect Frame
# ==========================================================
def detect_frame(landmarker, frame, timestamp_ms, async_mode=False):

    # Convert BGR (OpenCV) → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    if async_mode:
        landmarker.detect_async(mp_image, timestamp_ms)
        return None
    else:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        return result

# ==========================================================
# Visualizing Frame
# ==========================================================

def visualize_detection(frame, detection_result):

    if detection_result is None or not detection_result.hand_landmarks:
        return frame

    height, width = frame.shape[:2]

    FINGER_CONNECTIONS = {
        "Thumb": {"connections": [(0,1),(1,2),(2,3),(3,4)], "color": (0,0,255)},
        "Index": {"connections": [(0,5),(5,6),(6,7),(7,8)], "color": (255,0,0)},
        "Middle": {"connections": [(0,9),(9,10),(10,11),(11,12)], "color": (0,255,0)},
        "Ring": {"connections": [(0,13),(13,14),(14,15),(15,16)], "color": (0,255,255)},
        "Pinky": {"connections": [(0,17),(17,18),(18,19),(19,20)], "color": (255,0,255)},
        "Palm": {"connections": [(5,9),(9,13),(13,17)], "color": (200,200,200)}
    }

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):

        landmarks = np.array(
            [(lm.x * width, lm.y * height) for lm in hand_landmarks],
            dtype=np.int32
        )

        # =========================
        # Draw Colored Fingers
        # =========================
        for finger_data in FINGER_CONNECTIONS.values():
            color = finger_data["color"]
            for start_idx, end_idx in finger_data["connections"]:
                cv2.line(
                    frame,
                    tuple(landmarks[start_idx]),
                    tuple(landmarks[end_idx]),
                    color,
                    3
                )

        for point in landmarks:
            cv2.circle(frame, tuple(point), 4, (255,255,255), -1)

    return frame