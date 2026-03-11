from HandLandmark_mediapipe_utils import *
import pyautogui
import time
import threading


# ==========================================================
# Detect Jump / Crouch Gestures
# ==========================================================
is_jumping = False
is_crouching = False
jump_cooldown = 0.2
crouch_cooldown = 0.2
last_jump_time = 0
last_crouch_time = 0

def is_jump_gesture(hand_landmarks):
    Thumb_Tip = hand_landmarks[4]
    Index_Finger_Tip = hand_landmarks[8]
    Middle_Finger_Tip = hand_landmarks[12]
    Ring_Finger_Tip = hand_landmarks[16]
    Pinky_Tip = hand_landmarks[20]

    fingers_tips = [Index_Finger_Tip, Middle_Finger_Tip, Ring_Finger_Tip, Pinky_Tip]
    return all(finger.y > Thumb_Tip.y for finger in fingers_tips)

def is_crouch_gesture(hand_landmarks):
    Wrist = hand_landmarks[0]
    Index_Finger_Tip = hand_landmarks[8]
    # Crouch logic: Index finger tip is lower than the wrist
    return Index_Finger_Tip.y > Wrist.y

def handle_gestures(result: vision.HandLandmarkerResult):
    global is_jumping, is_crouching, last_jump_time, last_crouch_time

    if not result or not result.hand_landmarks or not result.handedness:
        # If no hands are detected, release crouch if active
        if is_crouching:
            pyautogui.keyUp('down')
            is_crouching = False
        return

    current_time = time.time()

    # Process each detected hand
    current_hand_closed_left = False
    current_hand_closed_right = False

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        # Get handedness label (Left or Right)
        # Note: MediaPipe handedness is mirrored for front-facing camera usually
        label = result.handedness[idx][0].category_name

        hand_closed = is_jump_gesture(hand_landmarks)

        if label == "Left":
            current_hand_closed_left = hand_closed
        elif label == "Right":
            current_hand_closed_right = hand_closed

    # Handle Jump (Left Hand Closed)
    if current_hand_closed_left:
        if not is_jumping and (current_time - last_jump_time > jump_cooldown):
            pyautogui.press('space')
            print("Left Hand Closed: Jump!")
            is_jumping = True
            last_jump_time = current_time
    else:
        is_jumping = False

    # Handle Crouch (Right Hand Closed)
    if current_hand_closed_right:
        if not is_crouching:
            pyautogui.keyDown('down')
            print("Right Hand Closed: Crouch Start")
            is_crouching = True
    else:
        if is_crouching:
            pyautogui.keyUp('down')
            print("Right Hand Closed: Crouch End")
            is_crouching = False

# ==========================================================
# Global Variable for Thread-Safe Detection Result
# ==========================================================
result_lock = threading.Lock()
latest_result = None
last_processed_timestamp = -1

def detection_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, last_processed_timestamp
    with result_lock:
        # Only update if this result is newer than what we've already processed
        if timestamp_ms > last_processed_timestamp:
            latest_result = result
            last_processed_timestamp = timestamp_ms

# ==========================================================
# Load Model
# ==========================================================
model_path = "../model/hand_landmarker.task"
landmarker = load_detection_model(model_path, callback=detection_callback)

# ==========================================================
# Main Video Capture
# ==========================================================
cap = cv2.VideoCapture(0)
# Optimization: Lower resolution to speed up detection and reduce processing latency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_frame_time = 0

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Camera is not found.")
        break

    # Optimization: Resize frame before detection to reduce MediaPipe workload
    # Convert to RGB and resize in one step for efficiency
    detection_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256))

    # Non-blocking detection with current system time
    timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detection_frame)
    landmarker.detect_async(mp_image, timestamp_ms)

    # Use the latest available result from the callback
    current_result = None
    with result_lock:
        current_result = latest_result

    if current_result:
        frame = visualize_detection(frame, current_result)
        handle_gestures(current_result)

    # Calculate and display FPS
    new_frame_time = time.time()
    if new_frame_time - prev_frame_time > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(frame, f'FPS: {fps}', (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)



    cv2.imshow('Play Dino Game using your hand', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()