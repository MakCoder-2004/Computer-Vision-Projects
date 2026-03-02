import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================
# Configuration
# ==============================

MODEL_PATH = "model/selfie_segmenter.tflite"
THRESHOLD = 0.2
BLUR_BACKGROUND = True   # False = solid background

BG_COLOR = (192, 192, 192)
SAVE_DIR = "captures"
BG_IMAGE_PATH = "images/background.jpg"  # Path to custom background image

os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# Load MediaPipe Segmenter
# ==============================
def load_segmenter(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True
    )

    return vision.ImageSegmenter.create_from_options(options)

# ==============================
# Apply Segmentation Effect
# ==============================
def apply_effect(frame, segmenter, use_image_bg=False, bg_image=None):
    # Resize frame for faster processing
    process_size = (320, 240)
    small_frame = cv2.resize(frame, process_size)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    start_seg = time.time()
    result = segmenter.segment(mp_image)
    seg_time = time.time() - start_seg

    mask = result.category_mask.numpy_view().astype(np.float32)
    mask = mask / 255.0
    mask = 1.0 - mask
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask_3ch = np.stack((mask,) * 3, axis=-1)

    # Resize mask back to original frame size
    mask_3ch = cv2.resize(mask_3ch, (frame.shape[1], frame.shape[0]))

    if use_image_bg and bg_image is not None:
        bg = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
        output = frame * mask_3ch + bg * (1 - mask_3ch)
    elif BLUR_BACKGROUND:
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)  # smaller kernel for speed
        output = frame * mask_3ch + blurred * (1 - mask_3ch)
    else:
        bg = np.full(frame.shape, BG_COLOR, dtype=np.uint8)
        output = frame * mask_3ch + bg * (1 - mask_3ch)

    # Print segmentation time for profiling
    print(f"Segmentation time: {seg_time:.3f}s")

    return output.astype(np.uint8)

# ==============================
# Main Webcam Loop
# ==============================
def main():
    print("🔄 Loading model...")
    segmenter = load_segmenter(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    else:
        print("Webcam opened successfully")

    # Load background image
    bg_image = None
    if os.path.exists(BG_IMAGE_PATH):
        bg_image = cv2.imread(BG_IMAGE_PATH)
        print(f"Loaded background image: {BG_IMAGE_PATH}")
    else:
        print(f"Background image not found: {BG_IMAGE_PATH}")

    use_image_bg = False
    prev_time = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = apply_effect(frame, segmenter, use_image_bg, bg_image)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Display FPS
        cv2.putText(
            output,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Real-Time Selfie Segmentation", output)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Save frame
        if key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"capture_{frame_count}.png")
            cv2.imwrite(filename, output)
            print(f"📸 Saved: {filename}")
            frame_count += 1

        # Toggle background mode
        if key == ord('r'):
            use_image_bg = not use_image_bg
            print(f"Background mode: {'Image' if use_image_bg else 'Blur/Solid'}")

    cap.release()
    cv2.destroyAllWindows()
    segmenter.close()
    print("✅ Clean exit")

if __name__ == "__main__":
    main()